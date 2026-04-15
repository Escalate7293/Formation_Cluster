import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import mad_std
from matplotlib.colors import LinearSegmentedColormap
from photutils.aperture import EllipticalAperture
from photutils.background import Background2D
from astropy.stats import SigmaClip,sigma_clip
from matplotlib.colors import LogNorm,PowerNorm
from matplotlib.patches import Ellipse,Rectangle,Circle
from sedcreator import SedFluxer

# 引入你的基础类和相关辅助函数 (假设在此同一目录下)
from .Formation_Cluster import Formation_Cluster, create_cutout_from_coords, casa_imfit_manually, replace_fits_data

#基本颜色映射
# 读取图片
image_path = "/home/esker7293/IRAS07299_highres/continuum/ColorTableHueSatValue2.png"  # 确保路径正确
img = mpimg.imread(image_path)

# 取中间一行的颜色数据
color_samples = img[img.shape[0] // 2, :, :]

# 归一化索引
num_colors = color_samples.shape[0]
normalized_indices = np.linspace(0, 1, num_colors)

# 创建 colormap
hue_sat_value2_cmap = LinearSegmentedColormap.from_list("HueSatValue2", list(zip(normalized_indices, color_samples)))


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
        self.results_dir = os.path.join(self.output_dir, "results")
        os.makedirs(self.cutout_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

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
                     clustername,
                     source_list_csv, 
                     fits_normal, 
                     std_normal, 
                     fits_rbm05=None, 
                     std_rbm05=None,
                     cutout_size=(100, 100),
                     bg_complex_ratio_threshold=1.2,
                     snr_threshold=20,
                     cmap=hue_sat_value2_cmap,
                     show_plots=True
                     ):
        """
        执行核心清洗管线
        :param source_list_csv: 提供源列表的CSV文件 (需包含列：RA, DEC)
        :param fits_normal: 标准 continuous 图 (Robust +0.5) 绝对路径
        :param std_normal: 标准图的底噪声
        :param fits_rbm05: 可选，高分辨率 robust-0.5 图路径。如果提供了会同时处理测光
        :param std_rbm05: 可选，高分辨率图噪声
        """

        # 1. 读表 (不再依赖 venn_code)
        sources_df = pd.read_csv(source_list_csv)
        if not all(col in sources_df.columns for col in ['RA', 'DEC']):
            raise ValueError("CSV must contain 'RA', and 'DEC' columns")

        # 2. 初始化天文图像基类
        print(f"[{self.cluster_name}] Loading normal FITS: {fits_normal}")
        fc_norm = Formation_Cluster(fits_normal, distance=self.distance_pc)
        
        has_rbm05 = fits_rbm05 is not None and std_rbm05 is not None
        if has_rbm05:
            print(f"[{self.cluster_name}] Loading rbm05 FITS: {fits_rbm05}")
            fc_rbm05 = Formation_Cluster(fits_rbm05, distance=self.distance_pc)

        num_sources = len(sources_df)
        results = {
            'RA': sources_df['RA'].values,
            'DEC': sources_df['DEC'].values,
            'IMFIT_logs_norm': [],  # 这里可以存储每个源的拟合日志或参数字典
            'LOCAL_mad_std': np.zeros(num_sources),
            'LOCAL_complex_bool': np.zeros(num_sources, dtype=bool),
            'SNR_normal': np.zeros(num_sources),
            'Sum_Flux_normal': np.zeros(num_sources),
            'Sum_Flux_err_normal': np.zeros(num_sources),
        }
        if has_rbm05:
            results['IMFIT_logs_rbm05'] = []
            results['Sum_Flux_rbm05'] = np.zeros(num_sources)
            results['Sum_Flux_err_rbm05'] = np.zeros(num_sources)

        # 3. 循环每个源进行切分和 imfit
        print(f"[{self.cluster_name}] Starting batch fitting...")
        for i, row in tqdm(sources_df.iterrows(), total=num_sources):
            ra, dec = row['RA'], row['DEC']
            
            # --- 建立独立切图目录 ---
            output_dir_src = os.path.join(self.cutout_dir, f"source_{i+1}")
            os.makedirs(output_dir_src, exist_ok=True)
            
            save_path_norm = os.path.join(output_dir_src, f"cutout_norm.fits")

            # 基础概览图，所有instance，暂时空出
            
            # --- 切图与峰值查找 (Normal) ---
            # cutout_norm = create_cutout_from_coords(
            #     ra, dec, fc_norm.img, fc_norm.wcs.celestial, fc_norm.head,
            #     freq=fc_norm.Freq, cutout_size=cutout_size, std_val=std_normal,
            #     save_path=save_path_norm, source_id=src_id, show=False
            # )

            cutout_normal = create_cutout_from_coords(
                ra, dec, fc_norm.img, fc_norm.wcs.celestial, fc_norm.head, 
                freq=fc_norm.Freq,
                cutout_size=cutout_size, 
                std_val=std_normal, 
                bt_func=fc_norm.Brightness_Temperature, 
                cmap=cmap, 
                show=False,  # 如果不需要每次都弹窗两个图，建议设为False，或者仅在下面的rmb05开启
                save_path=save_path_norm, 
                source_id=i+1
            )
            
            # 计算 Normal 的环境噪音
            std_surrounding_normal = mad_std(cutout_normal.data)

            # --- (可选) 执行 Rbm05 测光 ---
            if has_rbm05:
                save_path_rbm05 = os.path.join(output_dir_src, f"cutout_rbm05.fits")
                cutout_rbm05 = create_cutout_from_coords(
                    ra, dec, fc_rbm05.img, fc_rbm05.wcs.celestial, fc_rbm05.head, 
                    freq=fc_rbm05.Freq,
                    cutout_size=cutout_size, 
                    std_val=std_rbm05, 
                    bt_func=fc_rbm05.Brightness_Temperature, 
                    cmap=cmap, 
                    show=False, # 这里可以让其显示，或者根据需求调整
                    save_path=save_path_rbm05, 
                    source_id=i+1
                )


            def fit_and_mask_single_source(
                    instance_obj, cutout_obj, save_path_file, std_val, 
                    peak_x, peak_y, 
                    flux_arr, flux_err_arr,  # 接收结果数组
                    array_idx, suffix_name
                ):
                """
                逻辑：
                1. 尝试使用小盒子 (Small Box) 进行 imfit 拟合。
                2. 解析参数、SedFluxer 测光、生成 Mask。
                3. 如果上述任意步骤报错 (通常是 to_mask 报错)，进入 except。
                4. 在 except 中使用大盒子 (Large Box) 重试一遍。
                5. 返回: (source_mask, log_result, fit_params_dict)
                """
                
                # 定义 Box 范围
                box_small = f'{peak_x-5},{peak_y-5},{peak_x+5},{peak_y+5}'
                box_large = f'{peak_x-10},{peak_y-10},{peak_x+10},{peak_y+10}'
                
                # 初始化返回变量
                mask_img = np.zeros(cutout_obj.data.shape, dtype=bool)
                log_res = None
                fit_params = None

                # --- 尝试 1: 小盒子 ---
                try:
                    # 1. Imfit
                    log_res = casa_imfit_manually(
                        save_path_file, instance_obj,
                        manual_estimate=None, show_fitting_result=show_plots, zero_level=True,
                        box_set=box_small,
                        show_one_dim_result=show_plots, idx=peak_y, idy=peak_x,
                        RMS=std_val, savepath=self.results_dir,
                        fig_basename=clustername + f'_source_{i+1}_raw_{suffix_name}'
                    )

                    # 2. 解析几何参数 (如果拟合失败，这里取值可能会报错，或者得到NaN)
                    conmaj_sigma = log_res['ConMaj'][0] / (2 * np.sqrt(2 * np.log(2))) / instance_obj.PIXEL_SCALE.value
                    conmin_sigma = log_res['ConMin'][0] / (2 * np.sqrt(2 * np.log(2))) / instance_obj.PIXEL_SCALE.value
                    conPA = log_res['ConPA'][0]
                    ra_c = log_res["LongICRS"][0]
                    dec_c = log_res["LatICRS"][0]
                    fwhm_pix = (np.sqrt(log_res["ConMaj"][0] * log_res["ConMin"][0]) / instance_obj.PIXEL_SCALE.value)
                    sigma_pix = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))

                    # 3. SedFluxer 测光 (不去背景)
                    central_coords = SkyCoord(ra=ra_c*u.deg, dec=dec_c*u.deg, frame='icrs')
                    fluxer = SedFluxer(instance_obj.hdu[0])
                    aper_rad = 3 * sigma_pix * instance_obj.PIXEL_SCALE.value 
                    
                    flux_obj = fluxer.get_flux(central_coords, aper_rad, aper_rad, aper_rad*2)
                    if show_plots:
                        flux_obj.plot(cmap='jet') 
                    
                    # 写入数组
                    flux_arr[array_idx] = flux_obj.flux_bkgsub 
                    flux_err_arr[array_idx] = flux_obj.fluc_error

                    # 4. 生成 Mask (这是最容易报错的一步，如果参数离谱)
                    ra_pix_c, dec_pix_c = cutout_obj.wcs.celestial.all_world2pix(ra_c, dec_c, 0)
                    ap = EllipticalAperture((ra_pix_c, dec_pix_c), conmaj_sigma * 3, conmin_sigma * 3, np.radians(conPA+90))
                    mask_img = ap.to_mask().to_image(cutout_obj.data.shape)
                    mask_img = mask_img.astype(bool)

                    # 记录成功的拟合参数用于后续绘图
                    fit_params = {
                        'ra_pix': ra_pix_c, 'dec_pix': dec_pix_c,
                        'conmaj_sigma': conmaj_sigma, 'conmin_sigma': conmin_sigma, 'conPA': conPA
                    }

                except Exception as e:
                    # print(f"[{suffix_name}] Small box fit failed/invalid ({e}). Retrying with large box...")
                    
                    # --- 尝试 2: 大盒子 (逻辑完全相同，只是 Box 变大) ---
                    try:
                        log_res = casa_imfit_manually(
                            save_path_file, instance_obj,
                            manual_estimate=None, show_fitting_result=show_plots, zero_level=True,
                            box_set=box_large,  # <--- Change here
                            show_one_dim_result=show_plots, idx=peak_y, idy=peak_x,
                            RMS=std_val, savepath=self.results_dir,
                            fig_basename=clustername + f'_source_{i+1}_raw_{suffix_name}'
                        )

                        conmaj_sigma = log_res['ConMaj'][0] / (2 * np.sqrt(2 * np.log(2))) / instance_obj.PIXEL_SCALE.value
                        conmin_sigma = log_res['ConMin'][0] / (2 * np.sqrt(2 * np.log(2))) / instance_obj.PIXEL_SCALE.value
                        conPA = log_res['ConPA'][0]
                        ra_c = log_res["LongICRS"][0]
                        dec_c = log_res["LatICRS"][0]
                        fwhm_pix = (np.sqrt(log_res["ConMaj"][0] * log_res["ConMin"][0]) / instance_obj.PIXEL_SCALE.value)
                        sigma_pix = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))

                        central_coords = SkyCoord(ra=ra_c*u.deg, dec=dec_c*u.deg, frame='icrs')
                        fluxer = SedFluxer(instance_obj.hdu[0])
                        aper_rad = 3 * sigma_pix * instance_obj.PIXEL_SCALE.value 
                        
                        flux_obj = fluxer.get_flux(central_coords, aper_rad, aper_rad, aper_rad*2)
                        if show_plots:
                            flux_obj.plot(cmap='jet') 
                        
                        flux_arr[array_idx] = flux_obj.flux_bkgsub 
                        flux_err_arr[array_idx] = flux_obj.fluc_error

                        ra_pix_c, dec_pix_c = cutout_obj.wcs.celestial.all_world2pix(ra_c, dec_c, 0)
                        ap = EllipticalAperture((ra_pix_c, dec_pix_c), conmaj_sigma * 3, conmin_sigma * 3, np.radians(conPA+90))
                        mask_img = ap.to_mask().to_image(cutout_obj.data.shape)
                        mask_img = mask_img.astype(bool)

                        fit_params = {
                            'ra_pix': ra_pix_c, 'dec_pix': dec_pix_c,
                            'conmaj_sigma': conmaj_sigma, 'conmin_sigma': conmin_sigma, 'conPA': conPA
                        }
                    
                    except Exception as e2:
                        print(f"[{suffix_name}] Fit totally failed for source {i+1}: {e2}")
                        # 保持 mask_img 为全 False, fit_params 为 None
                
                return mask_img, log_res, fit_params
            
            # 进入分支：根据环境噪声与全局噪声的比值判断是否复杂背景
            if std_surrounding_normal > std_normal * bg_complex_ratio_threshold:
                sim_cen_x_pix, sim_cen_y_pix = fc_norm.wcs.celestial.all_world2pix(ra, dec, 0)
                sim_cen_x_pix, sim_cen_y_pix = int(sim_cen_x_pix), int(sim_cen_y_pix)
                
                # 在中心 10x10 区域找最大值
                img_vicinity = fc_norm.img[sim_cen_y_pix-5:sim_cen_y_pix+5, sim_cen_x_pix-5:sim_cen_x_pix+5]
                Ipeak = img_vicinity.max()
                
                # 获取 Peak 的像素坐标 (相对于 Vicinity)
                dec_peak_local, ra_peak_local = np.where(img_vicinity == Ipeak)
                
                # 转换为全局像素坐标
                ra_peak_pix_global = ra_peak_local[0] + (sim_cen_x_pix - 5)
                dec_peak_pix_global = dec_peak_local[0] + (sim_cen_y_pix - 5)
                
                # 转换为世界坐标 (RA/DEC)
                ra_peak, dec_peak = fc_norm.wcs.celestial.all_pix2world(ra_peak_pix_global, dec_peak_pix_global, 0)
                
                # 记录 SNR (基于 Normal)
                snr_this = Ipeak / std_normal
                # source_snr_array[i] = snr_this
                results['SNR_normal'][i] = snr_this

                # 计算在 Cutout 中的像素坐标 (Normal 和 Rmb05 的 WCS 和尺寸一致，统一使用 Normal 的坐标)
                ra_peak_pix_cutout, dec_peak_pix_cutout = cutout_normal.wcs.celestial.all_world2pix(ra_peak, dec_peak, 0)  
                ra_peak_pix_cutout = int(ra_peak_pix_cutout)
                dec_peak_pix_cutout = int(dec_peak_pix_cutout)

                # 1. 处理 Normal (独立 Try-Except)
                # ------------------------------------------------------------------------------
                source_mask_normal, log_raw_normal, fit_params_normal = fit_and_mask_single_source(
                    fc_norm, cutout_normal, save_path_norm, std_normal,
                    ra_peak_pix_cutout, dec_peak_pix_cutout,
                    results['Sum_Flux_normal'], results['Sum_Flux_err_normal'],
                    i, 'normal'
                )

                # 3. 背景估计 (Background2D) - 带中心Mask容错
                # ==========================================
                SigmaClip_set = SigmaClip(sigma=3.0, maxiters=None, stdfunc=mad_std)

                def get_robust_background(data, specific_mask, sigma_clip, box_size=(5, 5)):
                    """
                    尝试使用指定的 specific_mask 计算背景。
                    如果失败，则自动回退到 Mask 掉中心 20x20 区域再次尝试。
                    如果依然失败，返回全 0 背景。
                    """
                    # --- 尝试 1: 使用传入的特定 Mask (源的拟合形状) ---
                    if np.any(specific_mask):
                        try:
                            # exclude_percentile=50 允许 Box 中有一半像素被 Mask
                            bkg = Background2D(data, box_size, mask=specific_mask, 
                                            sigma_clip=sigma_clip, exclude_percentile=50)
                            return bkg.background
                        except Exception:
                            # 捕获 ValueError (All boxes contain > ...) 或其他错误
                            # print("Specific mask failed, trying fallback center mask...")
                            pass # 进入下方 fallback 逻辑
                    
                    # --- 尝试 2: Fallback - Mask 掉中心 20x20 区域 ---
                    try:
                        fallback_mask = np.zeros_like(data, dtype=bool)
                        ny, nx = data.shape
                        cy, cx = ny // 2, nx // 2
                        
                        # 定义中心 20x20 区域 (半径 10)
                        # 确保索引不越界
                        y_start = max(0, cy - 10)
                        y_end = min(ny, cy + 10)
                        x_start = max(0, cx - 10)
                        x_end = min(nx, cx + 10)
                        
                        fallback_mask[y_start:y_end, x_start:x_end] = True
                        
                        bkg = Background2D(data, box_size, mask=fallback_mask, 
                                        sigma_clip=sigma_clip, exclude_percentile=50)
                        return bkg.background
                        
                    except Exception:
                        # --- 尝试 3: 彻底失败，返回全 0 ---
                        # print(f"Background estimation failed completely. Returning 0.")
                        return np.zeros_like(data)

                # --- 计算 Normal 背景 ---
                bgmap_normal = get_robust_background(cutout_normal.data, source_mask_normal, SigmaClip_set)

                # --- 为了能在画图中用到 Rmb05 数据，我们先把 Rmb05 的拟合和背景计算提前 --- 
                if has_rbm05:
                    source_mask_rbm05, log_raw_rbm05, fit_params_rbm05 = fit_and_mask_single_source(
                        fc_rbm05, cutout_rbm05, save_path_rbm05, std_rbm05,
                        ra_peak_pix_cutout, dec_peak_pix_cutout,
                        results['Sum_Flux_rbm05'], results['Sum_Flux_err_rbm05'],
                        i, 'rmb05'
                    )
                    # 计算 Rmb05 背景
                    bgmap_rbm05 = get_robust_background(cutout_rbm05.data, source_mask_rbm05, SigmaClip_set)

                # =================== 画图模块 ===================
                if show_plots:
                    # 根据是否使用 Rmb05 动态分配画布行数和高度
                    nrows = 2 if has_rbm05 else 1
                    fig_height = 12 if has_rbm05 else 6
                    
                    fig, axes = plt.subplots(nrows, 3, figsize=(18, fig_height))
                    
                    # 如果只有 1 行，plt.subplots 给出的是 1D 数组 [ax1, ax2, ax3]
                    # 为了统一 axes[0] 和 axes[1] 的调用口补一个维度外壳
                    axes_2d = axes if has_rbm05 else [axes]
                    
                    # 准备 Colormap
                    current_cmap = plt.cm.jet.copy() if cmap is None else cmap.copy()
                    current_cmap.set_under(current_cmap(0.0))
                    current_cmap.set_over(current_cmap(1.0))

                    # --- 定义单行绘图逻辑 (内部帮助函数) ---
                    def plot_single_row(ax_row, cutout_obj, bgmap_arr, fit_params, title_prefix):
                        ax1, ax2, ax3 = ax_row
                        
                        vmax = np.nanmax(cutout_obj.data)
                        if vmax <= 0: vmax = 1e-3
                        norm = LogNorm(vmin=1e-5, vmax=vmax)

                        # Subplot 1: 原始数据
                        im1 = ax1.imshow(cutout_obj.data, origin='lower', cmap=current_cmap, norm=norm)
                        ax1.set_title(f"{title_prefix}\nOriginal Data", fontsize=10)

                        # Subplot 2: 背景模型 + Mask椭圆
                        im2 = ax2.imshow(bgmap_arr, origin='lower', cmap=current_cmap, norm=norm)
                        ax2.set_title("Background Model + Mask", fontsize=10)
                        
                        if fit_params is not None:
                            ellipse_mask = Ellipse(
                                xy=(fit_params['ra_pix'], fit_params['dec_pix']), 
                                width=fit_params['conmaj_sigma'] * 4, 
                                height=fit_params['conmin_sigma'] * 4,
                                angle=90 + fit_params['conPA'], 
                                edgecolor='black', facecolor='none', linewidth=2.0
                            )   
                            ax2.add_patch(ellipse_mask)

                        # Subplot 3: 扣除背景后的数据
                        subtracted_data = cutout_obj.data - bgmap_arr
                        im3 = ax3.imshow(subtracted_data, origin='lower', cmap=current_cmap, norm=norm)
                        ax3.set_title("Background Subtracted", fontsize=10)

                        fig.colorbar(im3, ax=ax_row.tolist(), orientation='horizontal', fraction=0.07, pad=0.1, extend='both')

                    # --- 绘制第一行: Normal ---
                    params_n = fit_params_normal if 'fit_params_normal' in locals() else None
                    title_norm = getattr(fc_norm, 'filename', 'Normal') 
                    plot_single_row(axes_2d[0], cutout_normal, bgmap_normal, params_n, f"Normal: {title_norm}")

                    # --- 绘制第二行: Rmb05 ---
                    if has_rbm05:
                        params_r = fit_params_rbm05 if 'fit_params_rbm05' in locals() else None
                        title_rbm = getattr(fc_rbm05, 'filename', 'Rmb05')
                        plot_single_row(axes_2d[1], cutout_rbm05, bgmap_rbm05, params_r, f"Rmb05: {title_rbm}")

                    plt.tight_layout()

                    # --- 保存合并后的图像 ---
                    if self.results_dir is not None:
                        # 针对不同模式命名
                        save_name = clustername + f'_source_{i+1}_maskbg_normal_rbm05.png' if has_rbm05 else clustername + f'_source_{i+1}_maskbg_normal.png'
                        save_full_path = os.path.join(self.results_dir, save_name)
                        fig.savefig(save_full_path, dpi=300, bbox_inches='tight')

                replace_file_path_normal = save_path_norm.replace('.fits','_bgmap.fits')
                replace_fits_data(
                    original_fits_path=save_path_norm,
                    new_data=cutout_normal.data - bgmap_normal,
                    output_path=replace_file_path_normal
                )

                if has_rbm05:
                    replace_file_path_rbm05 = save_path_rbm05.replace('.fits','_bgmap.fits')
                    replace_fits_data(
                        original_fits_path=save_path_rbm05,
                        new_data=cutout_rbm05.data - bgmap_rbm05,
                        output_path=replace_file_path_rbm05
                    )

                # Refit on background subtracted image
                try:
                    log_normal = casa_imfit_manually(
                        replace_file_path_normal,
                        fc_norm,
                        manual_estimate=None,
                        box_set=f'{ra_peak_pix_cutout-20},{dec_peak_pix_cutout-20},{ra_peak_pix_cutout+20},{dec_peak_pix_cutout+20}',
                        show_fitting_result=show_plots,
                        show_one_dim_result=show_plots, idx=dec_peak_pix_cutout, idy=ra_peak_pix_cutout,
                        RMS=std_normal,
                        savepath=self.results_dir,
                        fig_basename=clustername + f'_source_{i+1}_maskbg_sub_normal'
                    )
                except Exception as e:
                    log_normal = casa_imfit_manually(
                        replace_file_path_normal,
                        fc_norm,
                        manual_estimate=None,
                        box_set=f'{ra_peak_pix_cutout-5},{dec_peak_pix_cutout-5},{ra_peak_pix_cutout+5},{dec_peak_pix_cutout+5}',
                        show_fitting_result=show_plots,
                        show_one_dim_result=show_plots, idx=dec_peak_pix_cutout, idy=ra_peak_pix_cutout,
                        RMS=std_normal,
                        savepath=self.results_dir,
                        fig_basename=clustername + f'_source_{i+1}_maskbg_sub_normal'
                    )

                results['IMFIT_logs_norm'].append(log_normal)
                results['LOCAL_mad_std'][i] = std_surrounding_normal
                results['LOCAL_complex_bool'][i] = 1   # 标记为背景复杂

                # Refit on background subtracted image
                if has_rbm05:
                    try:
                        log_rbm05 = casa_imfit_manually(
                            replace_file_path_rbm05,
                            fc_rbm05,
                            manual_estimate=None,
                            box_set=f'{ra_peak_pix_cutout-20},{dec_peak_pix_cutout-20},{ra_peak_pix_cutout+20},{dec_peak_pix_cutout+20}',
                            show_fitting_result=show_plots,
                            show_one_dim_result=show_plots, idx=dec_peak_pix_cutout, idy=ra_peak_pix_cutout,
                            RMS=std_rbm05,
                            savepath=self.results_dir,
                            fig_basename=clustername + f'_source_{i+1}_maskbg_sub_rbm05'
                        )
                    except Exception as e:
                        log_rbm05 = casa_imfit_manually(
                            replace_file_path_rbm05,
                            fc_rbm05,
                            manual_estimate=None,
                            box_set=f'{ra_peak_pix_cutout-5},{dec_peak_pix_cutout-5},{ra_peak_pix_cutout+5},{dec_peak_pix_cutout+5}',
                            show_fitting_result=show_plots,
                            show_one_dim_result=show_plots, idx=dec_peak_pix_cutout, idy=ra_peak_pix_cutout,
                            RMS=std_rbm05,
                            savepath=self.results_dir,
                            fig_basename=clustername + f'_source_{i+1}_maskbg_sub_rbm05'
                        )

                    results['IMFIT_logs_rbm05'].append(log_rbm05)
