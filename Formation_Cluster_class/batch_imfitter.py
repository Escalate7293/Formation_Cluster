import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

# 引入你的基础类和相关辅助函数 (假设在此同一目录下)
from .Formation_Cluster import Formation_Cluster, create_cutout_from_coords, mad_std, casa_imfit_manually, SedFluxer

class BatchImfitter:
    def __init__(self, output_dir, distance_pc, cluster_name="Target"):
        """
        通用化批量二维高斯拟合流水线
        :param output_dir: 结果、切图、拟合日志的输出根目录
        :param distance_pc: 天体的距离(单位：pc)，传递给 Formation_Cluster
        :param cluster_name: 目标星团名称（仅用于文件命名标识）
        """
        self.output_dir = output_dir
        self.distance_pc = distance_pc
        self.cluster_name = cluster_name
        self.cutout_dir = os.path.join(self.output_dir, "cutouts")
        os.makedirs(self.cutout_dir, exist_ok=True)
        
    def _sum_flux_sedfluxer(self, working_dir, instance_this, ra_center, dec_center):
        """内部测光辅助函数: 寻找 CASA fit 日志并依靠 SedFluxer 测光"""
        try:
            fitlog_summary_file = os.path.join(working_dir, "fit_summary_log.dat")
            if not os.path.exists(fitlog_summary_file):
                raise FileNotFoundError(f"Fit summary missing: {fitlog_summary_file}")
                
            df = pd.read_csv(fitlog_summary_file, index_col=False, header=0, delim_whitespace=True, skiprows=1)
            if df.empty:
                raise ValueError("Fit summary is empty.")
                
            fitlog_data = df.shift(axis=1) 
            ra_center_fit = fitlog_data["LongICRS"][0]
            dec_center_fit = fitlog_data["LatICRS"][0]
                        
            fwhm_pix = (np.sqrt(fitlog_data["ConMaj"][0] * fitlog_data["ConMin"][0]) / instance_this.PIXEL_SCALE.value)
            sigma_pix = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            
            central_coords = SkyCoord(ra=ra_center_fit*u.deg, dec=dec_center_fit*u.deg, frame='icrs')
            fluxer = SedFluxer(instance_this.hdu[0])
            aper_rad = 3 * sigma_pix * instance_this.PIXEL_SCALE.value # 3 sigma circle
            
            flux_obj = fluxer.get_flux(central_coords, aper_rad, aper_rad, aper_rad*2)
            return flux_obj.flux_bkgsub, flux_obj.fluc_error, fitlog_data
        except Exception as e:
            # 静默返回 None，表示拟合未正确收敛产生数据
            return None, None, None

    def run_pipeline(self, 
                     source_list_csv, 
                     fits_normal, 
                     std_normal, 
                     fits_rbm05=None, 
                     std_rbm05=None,
                     cutout_size=(100, 100),
                     snr_threshold=20):
        """
        执行核心清洗管线
        :param source_list_csv: 提供源列表的CSV文件 (需包含列：ID, RA, DEC)
        :param fits_normal: 标准 continuous 图 (Robust +0.5) 绝对路径
        :param std_normal: 标准图的底噪声
        :param fits_rbm05: 可选，高分辨率 robust-0.5 图路径。如果提供了会同时处理测光
        :param std_rbm05: 可选，高分辨率图噪声
        """
        # 1. 读表 (不再依赖 venn_code)
        sources_df = pd.read_csv(source_list_csv)
        if not all(col in sources_df.columns for col in ['ID', 'RA', 'DEC']):
            raise ValueError("CSV must contain 'ID', 'RA', and 'DEC' columns")

        # 2. 初始化天文图像基类
        print(f"[{self.cluster_name}] Loading normal FITS: {fits_normal}")
        fc_norm = Formation_Cluster(fits_normal, distance=self.distance_pc)
        
        has_rbm05 = fits_rbm05 is not None and std_rbm05 is not None
        if has_rbm05:
            print(f"[{self.cluster_name}] Loading rbm05 FITS: {fits_rbm05}")
            fc_rbm05 = Formation_Cluster(fits_rbm05, distance=self.distance_pc)

        num_sources = len(sources_df)
        results = {
            'ID': sources_df['ID'].values,
            'RA': sources_df['RA'].values,
            'DEC': sources_df['DEC'].values,
            'SNR_normal': np.zeros(num_sources),
            'Flux_normal': np.zeros(num_sources),
            'Flux_err_normal': np.zeros(num_sources),
        }
        if has_rbm05:
            results['Flux_rbm05'] = np.zeros(num_sources)
            results['Flux_err_rbm05'] = np.zeros(num_sources)

        # 3. 循环每个源进行切分和 imfit
        print(f"[{self.cluster_name}] Starting batch fitting...")
        for i, row in tqdm(sources_df.iterrows(), total=num_sources):
            src_id, ra, dec = row['ID'], row['RA'], row['DEC']
            
            # --- 建立独立切图目录 ---
            output_dir_src = os.path.join(self.cutout_dir, f"source_{src_id}")
            os.makedirs(output_dir_src, exist_ok=True)
            
            save_path_norm = os.path.join(output_dir_src, f"cutout_norm.fits")
            
            # --- 切图与峰值查找 (Normal) ---
            cutout_norm = create_cutout_from_coords(
                ra, dec, fc_norm.img, fc_norm.wcs.celestial, fc_norm.head,
                freq=fc_norm.Freq, cutout_size=cutout_size, std_val=std_normal,
                save_path=save_path_norm, source_id=src_id, show=False
            )
            
            # 根据你的原逻辑，去中心搜寻局部峰值计算 SNR (省略复杂 WCS 折算，示意核心概念)
            cen_x, cen_y = cutout_size[1]//2, cutout_size[0]//2
            local_img = cutout_norm.data[cen_y-5:cen_y+5, cen_x-5:cen_x+5]
            ipeak = np.nanmax(local_img)
            snr_this = ipeak / std_normal
            results['SNR_normal'][i] = snr_this
            
            # --- 分支：大框 vs 小框 ---
            if snr_this <= snr_threshold:
                box_set = f"{cen_x-10},{cen_y-10},{cen_x+10},{cen_y+10}"
                use_pt = False
            else:
                box_set = "30,30,70,70"
                use_pt = None
                
            # --- 执行 CASA Imfit (利用原始的 Try-except 退网机制) ---
            try:
                casa_log_norm = casa_imfit_manually(
                    save_path_norm, fc_norm, box_set=box_set, 
                    RMS=std_normal, show_fitting_result=False
                    # fcen_ra, IPeak 等引数如果需要可继续追加
                )
                flux, err, log_df = self._sum_flux_sedfluxer(output_dir_src, fc_norm, ra, dec)
                if flux is not None:
                    results['Flux_normal'][i] = flux
                    results['Flux_err_normal'][i] = err
            except Exception as e:
                pass # 拟合彻底失败，维持 0
                
            # --- (可选) 执行 Rbm05 测光 ---
            if has_rbm05:
                save_path_rbm05 = os.path.join(output_dir_src, f"cutout_rbm05.fits")
                # 创建 Rbm05 切图
                create_cutout_from_coords(
                    ra, dec, fc_rbm05.img, fc_rbm05.wcs.celestial, fc_rbm05.head,
                    freq=fc_rbm05.Freq, cutout_size=cutout_size, std_val=std_rbm05,
                    save_path=save_path_rbm05, source_id=src_id, show=False
                )
                try:
                    casa_imfit_manually(
                        save_path_rbm05, fc_rbm05, box_set=box_set, 
                        RMS=std_rbm05, show_fitting_result=False
                    )
                    flux_r, err_r, _ = self._sum_flux_sedfluxer(output_dir_src, fc_rbm05, ra, dec)
                    if flux_r is not None:
                        results['Flux_rbm05'][i] = flux_r
                        results['Flux_err_rbm05'][i] = err_r
                except Exception as e:
                    pass

        # 4. 把数组存储为好读的 CSV 或 Picke
        plt.close('all')
        res_df = pd.DataFrame(results)
        final_csv = os.path.join(self.output_dir, f"{self.cluster_name}_imfit_results.csv")
        res_df.to_csv(final_csv, index=False)
        print(f"[{self.cluster_name}] All done! Results saved to {final_csv}")
        return res_df