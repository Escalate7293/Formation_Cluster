import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import mad_std, sigma_clipped_stats
from matplotlib.colors import LinearSegmentedColormap
from photutils.aperture import EllipticalAperture
from photutils.background import Background2D
from astropy.stats import SigmaClip,sigma_clip
from matplotlib.colors import LogNorm,PowerNorm
from matplotlib.patches import Ellipse,Rectangle,Circle
from sedcreator import SedFluxer
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import pandas as pd

# 引入你的基础类和相关辅助函数 (假设在此同一目录下)
from .Formation_Cluster import Formation_Cluster, create_cutout_from_coords, casa_imfit_manually, replace_fits_data, _scientific_plot_rc

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
            'SNR_complex': np.full(num_sources, np.nan),
            'SNR_complex_local': np.full(num_sources, np.nan),
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
            output_dir_result_src = os.path.join(self.results_dir, f"source_{i+1}")
            os.makedirs(output_dir_result_src, exist_ok=True)
            
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
                        RMS=std_val, savepath=output_dir_result_src, #self.results_dir,
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
                    # if show_plots:
                    #     flux_obj.plot(cmap='jet') 
                    
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
                            RMS=std_val, savepath=output_dir_result_src, #self.results_dir,
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
                        # if show_plots:
                        #     flux_obj.plot(cmap='jet') 
                        
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
                    nrows = 2 if has_rbm05 else 1
                    ncols = 3

                    with plt.rc_context(_scientific_plot_rc(labelsize=10, axes_labelsize=11)):
                        fig, axes = plt.subplots(
                            nrows, ncols, figsize=(13.5, 4.3 * nrows),
                            sharex=True, sharey=True, constrained_layout=False
                        )
                        axes_2d = np.atleast_2d(axes)
                        fig.subplots_adjust(left=0.06, right=0.905, bottom=0.08, top=0.97,
                                            wspace=0.0, hspace=0.0)

                        current_cmap = plt.cm.jet.copy() if cmap is None else cmap.copy()
                        current_cmap.set_under(current_cmap(0.0))
                        current_cmap.set_over(current_cmap(1.0))

                        def add_panel_text(ax, text):
                            ax.text(
                                0.05, 0.95, text, transform=ax.transAxes,
                                va='top', ha='left', fontsize=10, color='black',
                                bbox=dict(facecolor='white', alpha=0.72, edgecolor='none', pad=2.0)
                            )

                        def style_joined_axis(ax, row_idx, col_idx):
                            is_bottom = row_idx == nrows - 1
                            is_left = col_idx == 0

                            ax.set_aspect('equal', adjustable='box')
                            ax.minorticks_on()
                            ax.tick_params(
                                axis='x', which='both', direction='in',
                                top=True, bottom=True,
                                labeltop=False, labelbottom=is_bottom
                            )
                            ax.tick_params(
                                axis='y', which='both', direction='in',
                                left=True, right=True,
                                labelleft=is_left, labelright=False
                            )

                            # Keep the full in-panel frame and inward ticks on every panel.
                            for spine in ax.spines.values():
                                spine.set_visible(True)

                            ax.set_xlabel('Pixel offset' if is_bottom else '')
                            ax.set_ylabel('Pixel offset' if is_left else '')

                        def add_row_colorbar(ax, image):
                            fig.canvas.draw()
                            ax_pos = ax.get_position().frozen()

                            divider = make_axes_locatable(ax)
                            cax = divider.append_axes('right', size='3%', pad=0.02)

                            # Freeze the image panel after creating the divider; otherwise
                            # append_axes compresses the third column and breaks alignment.
                            ax.set_axes_locator(None)
                            ax.set_position(ax_pos)
                            cax.set_axes_locator(None)

                            pad = 0.02 / fig.get_size_inches()[0]
                            cbar_width = max(ax_pos.width * 0.03, 0.006)
                            cax.set_position([ax_pos.x1 + pad, ax_pos.y0, cbar_width, ax_pos.height])

                            cbar = fig.colorbar(image, cax=cax, orientation='vertical', extend='both')
                            cbar.set_label('Intensity (Jy/beam)', fontsize=10)
                            cbar.ax.tick_params(axis='y', which='both', direction='in', labelsize=9)

                        # --- 定义单行绘图逻辑 (内部帮助函数) ---
                        def plot_single_row(ax_row, cutout_obj, bgmap_arr, fit_params, row_label, row_idx):
                            ax1, ax2, ax3 = ax_row

                            vmax = np.nanmax(cutout_obj.data)
                            if not np.isfinite(vmax) or vmax <= 0:
                                vmax = 1e-3
                            norm = LogNorm(vmin=1e-5, vmax=vmax)

                            ny, nx = cutout_obj.data.shape
                            x_center = 0.5 * (nx - 1)
                            y_center = 0.5 * (ny - 1)
                            extent = [
                                -0.5 - x_center, nx - 0.5 - x_center,
                                -0.5 - y_center, ny - 0.5 - y_center,
                            ]

                            im1 = ax1.imshow(cutout_obj.data, origin='lower', cmap=current_cmap,
                                             norm=norm, extent=extent)
                            im2 = ax2.imshow(bgmap_arr, origin='lower', cmap=current_cmap,
                                             norm=norm, extent=extent)

                            if fit_params is not None:
                                ellipse_mask = Ellipse(
                                    xy=(fit_params['ra_pix'] - x_center, fit_params['dec_pix'] - y_center),
                                    width=fit_params['conmaj_sigma'] * 4,
                                    height=fit_params['conmin_sigma'] * 4,
                                    angle=90 + fit_params['conPA'],
                                    edgecolor='black', facecolor='none', linewidth=1.4
                                )
                                ax2.add_patch(ellipse_mask)

                            subtracted_data = cutout_obj.data - bgmap_arr
                            im3 = ax3.imshow(subtracted_data, origin='lower', cmap=current_cmap,
                                             norm=norm, extent=extent)

                            add_panel_text(ax1, f'{row_label}\nOriginal Data')
                            add_panel_text(ax2, f'{row_label}\nBackground Model + Mask')
                            add_panel_text(ax3, f'{row_label}\nBackground Subtracted')

                            for col_idx, ax_this in enumerate(ax_row):
                                style_joined_axis(ax_this, row_idx, col_idx)

                            add_row_colorbar(ax3, im3)

                        # --- 绘制第一行: Normal ---
                        params_n = fit_params_normal if 'fit_params_normal' in locals() else None
                        plot_single_row(axes_2d[0], cutout_normal, bgmap_normal, params_n, 'Normal', 0)

                        # --- 绘制第二行: Rmb05 ---
                        if has_rbm05:
                            params_r = fit_params_rbm05 if 'fit_params_rbm05' in locals() else None
                            plot_single_row(axes_2d[1], cutout_rbm05, bgmap_rbm05, params_r, 'Rmb05', 1)

                    # --- 保存合并后的图像 ---
                    # if self.results_dir is not None:
                    if output_dir_result_src is not None:
                        # 针对不同模式命名
                        save_name = clustername + f'_source_{i+1}_maskbg_normal_rbm05.png' if has_rbm05 else clustername + f'_source_{i+1}_maskbg_normal.png'
                        save_full_path = os.path.join(output_dir_result_src, save_name)
                        fig.savefig(save_full_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)

                bgsub_normal = cutout_normal.data - bgmap_normal
                replace_file_path_normal = save_path_norm.replace('.fits','_bgmap.fits')
                replace_fits_data(
                    original_fits_path=save_path_norm,
                    new_data=bgsub_normal,
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
                        savepath=output_dir_result_src, #self.results_dir,
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
                        savepath=output_dir_result_src, #self.results_dir,
                        fig_basename=clustername + f'_source_{i+1}_maskbg_sub_normal'
                    )

                try:
                    fit_row = log_normal.iloc[0]
                    peak_complex = float(fit_row['Peak'])
                    if np.isfinite(peak_complex) and np.isfinite(std_normal) and std_normal > 0:
                        results['SNR_complex'][i] = peak_complex / std_normal
                    conmaj_sigma = (
                        float(fit_row['ConMaj'])
                        / (2.0 * np.sqrt(2.0 * np.log(2.0)))
                        / fc_norm.PIXEL_SCALE.value
                    )
                    conmin_sigma = (
                        float(fit_row['ConMin'])
                        / (2.0 * np.sqrt(2.0 * np.log(2.0)))
                        / fc_norm.PIXEL_SCALE.value
                    )
                    source_x, source_y = cutout_normal.wcs.celestial.all_world2pix(
                        float(fit_row['LongICRS']), float(fit_row['LatICRS']), 0
                    )
                    source_aperture = EllipticalAperture(
                        (source_x, source_y),
                        3.0 * conmaj_sigma,
                        3.0 * conmin_sigma,
                        np.radians(float(fit_row['ConPA']) + 90.0),
                    )
                    source_mask = source_aperture.to_mask().to_image(bgsub_normal.shape).astype(bool)
                    noise_pixels = bgsub_normal[np.isfinite(bgsub_normal) & ~source_mask]
                    _, _, std_complex = sigma_clipped_stats(
                        noise_pixels, sigma=3.0, maxiters=None
                    )
                    if np.isfinite(peak_complex) and np.isfinite(std_complex) and std_complex > 0:
                        results['SNR_complex_local'][i] = peak_complex / std_complex
                except Exception as exc:
                    print(f"[{clustername}] source {i + 1} complex SNR failed: {exc}")

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
                            savepath=output_dir_result_src, #self.results_dir,
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
                            savepath=output_dir_result_src, #self.results_dir,
                            fig_basename=clustername + f'_source_{i+1}_maskbg_sub_rbm05'
                        )

                    results['IMFIT_logs_rbm05'].append(log_rbm05)
            
            else: # 环境简单，直接标记并记录 SNR 和 Flux，跳过复杂的拟合和背景处理
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
                # source_snr_array[idx] = snr_this
                results['SNR_normal'][i] = snr_this

                # ... (前文：获取 sim_cen_x_pix, 计算 Ipeak, ra_peak, dec_peak 等逻辑保持不变) ...
                # ... (前文：计算 snr_this, source_snr_array[idx] = snr_this) ...

                # 计算在 Cutout 中的像素坐标 (Normal 和 Rmb05 的 WCS 和尺寸一致，统一使用 Normal 的坐标)
                ra_peak_pix_cutout, dec_peak_pix_cutout = cutout_normal.wcs.celestial.all_world2pix(ra_peak, dec_peak, 0)  
                ra_peak_pix_cutout = int(ra_peak_pix_cutout)
                dec_peak_pix_cutout = int(dec_peak_pix_cutout)

                if snr_this <= snr_threshold:
                    aaa_cutout = Formation_Cluster(save_path_norm)
                    ra_peak_pix_cutout, dec_peak_pix_cutout = aaa_cutout.wcs.celestial.all_world2pix(ra_peak, dec_peak, 0)  
                    ra_peak_pix_cutout = int(ra_peak_pix_cutout)
                    dec_peak_pix_cutout = int(dec_peak_pix_cutout)
                    
                    try:
                        log_normal = casa_imfit_manually(
                            save_path_norm,
                            fc_norm,
                            manual_estimate=None,
                            show_fitting_result=show_plots,
                            # zero_level=True,
                            box_set=f'{ra_peak_pix_cutout-10},{dec_peak_pix_cutout-10},{ra_peak_pix_cutout+10},{dec_peak_pix_cutout+10}',
                            show_one_dim_result=show_plots, idx=dec_peak_pix_cutout, idy=ra_peak_pix_cutout,
                            RMS=std_normal,
                            fcen_ra=ra_peak, fcen_dec=dec_peak, Ipeak=Ipeak, point_source=False,
                            savepath=output_dir_result_src,
                            fig_basename=clustername + f'_source_{i+1}_low_snr_normal'
                        )
                    except:
                        # 如果拟合失败，返回空日志
                        log_normal = None
                    
                    if has_rbm05:
                        try:
                            log_rbm05 = casa_imfit_manually(
                                save_path_rbm05,
                                fc_rbm05,
                                manual_estimate=None,
                                show_fitting_result=show_plots,
                                # zero_level=True,
                                box_set=f'{ra_peak_pix_cutout-10},{dec_peak_pix_cutout-10},{ra_peak_pix_cutout+10},{dec_peak_pix_cutout+10}',
                                show_one_dim_result=show_plots, idx=dec_peak_pix_cutout, idy=ra_peak_pix_cutout,
                                RMS=std_rbm05,
                                fcen_ra=ra_peak, fcen_dec=dec_peak, Ipeak=Ipeak, point_source=False,
                                savepath=output_dir_result_src,
                                fig_basename=clustername + f'_source_{i+1}_low_snr_rbm05'
                            )
                        except:
                            log_rbm05 = None

                else:
                    log_normal = casa_imfit_manually(
                        save_path_norm,
                        fc_norm,
                        manual_estimate=None,
                        show_fitting_result=show_plots,
                        # zero_level=True,
                        box_set='30,30,70,70',
                        # box_set=f'{ra_peak_pix_cutout-5},{dec_peak_pix_cutout-5},{ra_peak_pix_cutout+5},{dec_peak_pix_cutout+5}',
                        show_one_dim_result=show_plots, idx=50, idy=50,
                        RMS=std_normal,
                        # fcen_ra=ra_peak, fcen_dec=dec_peak, Ipeak=Ipeak
                        savepath=output_dir_result_src, #self.results_dir,
                        fig_basename=clustername + f'_source_{i+1}_high_snr_normal'
                    )
                    
                    if has_rbm05:
                        log_rbm05 = casa_imfit_manually(
                            save_path_rbm05,
                            fc_rbm05,
                            manual_estimate=None,
                            show_fitting_result=show_plots,
                            # zero_level=True,
                            box_set='30,30,70,70',
                            # box_set=f'{ra_peak_pix_cutout-5},{dec_peak_pix_cutout-5},{ra_peak_pix_cutout+5},{dec_peak_pix_cutout+5}',
                            show_one_dim_result=show_plots, idx=50, idy=50,
                            RMS=std_rbm05,
                            # fcen_ra=ra_peak, fcen_dec=dec_peak, Ipeak=Ipeak
                            savepath=output_dir_result_src, #self.results_dir,
                            fig_basename=clustername + f'_source_{i+1}_high_snr_rbm05'
                        )

                def sum_flux_sedfluxer(save_path_this, instance_this, sum_flux_array, sum_flux_err_array, idx):
                    working_dir = save_path_this.replace('.fits','')
                    fitlog_file = os.path.join(working_dir, "fit_log.dat")
                    fitlog_summary_file = os.path.join(working_dir, "fit_summary_log.dat")

                    df = pd.read_csv(fitlog_summary_file, index_col=False, header=0, delim_whitespace=True, skiprows=1)
                    fitlog_data = df.shift(axis=1) # 保持原代码逻辑
                    ra_center_fit = fitlog_data["LongICRS"][0]
                    dec_center_fit = fitlog_data["LatICRS"][0]
                                
                    fwhm_pix = (np.sqrt(fitlog_data["ConMaj"][0] * fitlog_data["ConMin"][0]) / instance_this.PIXEL_SCALE.value)
                    sigma_pix = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))

                    central_coords = SkyCoord(ra=ra_center_fit*u.deg, dec=dec_center_fit*u.deg, frame='icrs')
                    fluxer = SedFluxer(instance_this.hdu[0])
                    aper_rad = 3 * sigma_pix * instance_this.PIXEL_SCALE.value # 3 sigma circle
                    
                    flux_obj = fluxer.get_flux(central_coords, aper_rad, aper_rad, aper_rad*2)
                    # if show_plots:
                    #     flux_obj.plot(cmap='jet') # 原代码写死 jet
                        
                    # sum_flux_array[idx] = flux_obj.flux_bkgsub # 注意使用 idx
                    # sum_flux_err_array[idx] = flux_obj.fluc_error
                    sum_flux_array[idx] = flux_obj.flux_bkgsub # 注意使用 idx
                    sum_flux_err_array[idx] = flux_obj.fluc_error
                    return None

                try:
                    sum_flux_sedfluxer(save_path_norm, fc_norm, results['Sum_Flux_normal'], results['Sum_Flux_err_normal'], i)
                # except:
                except Exception as exc:
                    print(f"[{clustername}] source {i + 1} normal aperture failed: {exc}")
                    results['Sum_Flux_normal'][i] = 0.0
                    results['Sum_Flux_err_normal'][i] = 0.0
                if has_rbm05:
                    try:
                        sum_flux_sedfluxer(save_path_rbm05, fc_rbm05, results['Sum_Flux_rbm05'], results['Sum_Flux_err_rbm05'], i)
                    # except:
                    except Exception as exc:
                        print(f"[{clustername}] source {i + 1} rbm05 aperture failed: {exc}")
                        results['Sum_Flux_rbm05'][i] = 0.0
                        results['Sum_Flux_err_rbm05'][i] = 0.0
                # sum_bool_array[idx] = 1
                results['LOCAL_complex_bool'][i] = 0  # 标记为背景简单
                results['LOCAL_mad_std'][i] = std_surrounding_normal  # 以normal的背景为标准
                results['IMFIT_logs_norm'].append(log_normal)

                if has_rbm05:
                    results['IMFIT_logs_rbm05'].append(log_rbm05)

        plt.close('all')
        self.results = results  # 将结果保存在实例属性中，方便后续访问
        return results

    def save_results_to_csv(self, output_dir=None):
        """
        Saves the processed results from the pipeline into CSV files.
        Separates into normal and rbm05 tables if both exist.
        """
        
        if not hasattr(self, 'results') or not self.results:
            print("No results found. Run pipeline first.")
            return

        out_dir = output_dir if output_dir else self.output_dir
        os.makedirs(out_dir, exist_ok=True)
        
        # 提取共有属性
        base_data = {
            'Source_Index': range(1, len(self.results['RA']) + 1),
            'Raw_RA': self.results['RA'],
            'Raw_DEC': self.results['DEC'],
            'LOCAL_complex_bool': self.results['LOCAL_complex_bool'],
            'LOCAL_mad_std': self.results['LOCAL_mad_std'],
            'SNR_normal': self.results['SNR_normal'],
            'SNR_complex': self.results['SNR_complex'],
            'SNR_complex_local': self.results['SNR_complex_local'],
        }
        
        # 构建 normal 表格数据
        norm_data = base_data.copy()
        norm_data['Sum_Flux'] = self.results['Sum_Flux_normal']
        norm_data['Sum_Flux_err'] = self.results['Sum_Flux_err_normal']
        norm_data['Imfit_Flux'] = np.zeros(len(self.results['RA']))  # 占位，后续填充
        norm_data['Imfit_Flux_err'] = np.zeros(len(self.results['RA']))
        norm_data['Peak_Intensity'] = np.zeros(len(self.results['RA']))
        norm_data['Peak_Intensity_err'] = np.zeros(len(self.results['RA']))
        norm_data['deconmajFWHM'] = np.zeros(len(self.results['RA']))
        norm_data['deconmajFWHM_err'] = np.zeros(len(self.results['RA']))
        norm_data['deconminFWHM'] = np.zeros(len(self.results['RA']))
        norm_data['deconminFWHM_err'] = np.zeros(len(self.results['RA']))
        norm_data['deconPA'] = np.zeros(len(self.results['RA']))
        norm_data['deconPA_err'] = np.zeros(len(self.results['RA']))
        norm_data['imfit_ra'] = np.zeros(len(self.results['RA']))
        norm_data['imfit_ra_err'] = np.zeros(len(self.results['RA']))
        norm_data['imfit_dec'] = np.zeros(len(self.results['RA']))
        norm_data['imfit_dec_err'] = np.zeros(len(self.results['RA']))
        norm_data['validfit_flag'] = np.zeros(len(self.results['RA']), dtype=int)  # 0: invalid, 1: valid


        required_fit_fields = ["I", "Peak", "LongICRS", "LatICRS"]

        for index, log in enumerate(self.results['IMFIT_logs_norm']):
            try:
                row = log.iloc[0] # pipeline中默认当成单源处理，取第一行
                if not np.isfinite(
                    row[required_fit_fields].astype(float)
                ).all():
                    continue
                norm_data['Imfit_Flux'][index] = row['I']
                norm_data['Imfit_Flux_err'][index] = row['Ierr']
                norm_data['Peak_Intensity'][index] = row['Peak']
                norm_data['Peak_Intensity_err'][index] = row['PeakErr']
                norm_data['deconmajFWHM'][index] = row['DeconMaj']
                norm_data['deconmajFWHM_err'][index] = row['DeconMajErr']
                norm_data['deconminFWHM'][index] = row['DeconMin']
                norm_data['deconminFWHM_err'][index] = row['DeconMinErr']
                norm_data['deconPA'][index] = row['DeconPA']
                norm_data['deconPA_err'][index] = row['DeconPAErr']
                norm_data['imfit_ra'][index] = row['LongICRS']
                norm_data['imfit_ra_err'][index] = row['LongICRSerr']
                norm_data['imfit_dec'][index] = row['LatICRS']
                norm_data['imfit_dec_err'][index] = row['LatICRSerr']
                norm_data['validfit_flag'][index] = 1  # 标记为有效拟合
            except:
                # 如果日志无效或格式不对，保持默认值（0或NaN）
                continue

        df_normal = pd.DataFrame(norm_data)
        csv_path_norm = os.path.join(out_dir, f"{self.cluster_name}_results_normal.csv")
        df_normal.to_csv(csv_path_norm, index=False)
        print(f"Normal results saved to: {csv_path_norm}")
        
        # 若存在 rbm05 数据，构建 rbm05 表格
        if 'Sum_Flux_rbm05' in self.results and any(x is not None for x in self.results['Sum_Flux_rbm05']):
            rbm05_data = base_data.copy()
            rbm05_data['Sum_Flux'] = self.results['Sum_Flux_rbm05']
            rbm05_data['Sum_Flux_err'] = self.results['Sum_Flux_err_rbm05']
            rbm05_data['Imfit_Flux'] = np.zeros(len(self.results['RA']))  # 占位，后续填充
            rbm05_data['Imfit_Flux_err'] = np.zeros(len(self.results['RA']))
            rbm05_data['Peak_Intensity'] = np.zeros(len(self.results['RA']))  # 占位，后续填充
            rbm05_data['Peak_Intensity_err'] = np.zeros(len(self.results['RA']))
            rbm05_data['deconmajFWHM'] = np.zeros(len(self.results['RA']))  # 占位，后续填充
            rbm05_data['deconmajFWHM_err'] = np.zeros(len(self.results['RA']))
            rbm05_data['deconminFWHM'] = np.zeros(len(self.results['RA']))  # 占位，后续填充
            rbm05_data['deconminFWHM_err'] = np.zeros(len(self.results['RA']))
            rbm05_data['deconPA'] = np.zeros(len(self.results['RA']))  # 占位，后续填充
            rbm05_data['deconPA_err'] = np.zeros(len(self.results['RA']))
            rbm05_data['imfit_ra'] = np.zeros(len(self.results['RA']))  # 占位，后续填充
            rbm05_data['imfit_ra_err'] = np.zeros(len(self.results['RA']))
            rbm05_data['imfit_dec'] = np.zeros(len(self.results['RA']))  # 占位，后续填充
            rbm05_data['imfit_dec_err'] = np.zeros(len(self.results['RA']))
            rbm05_data['validfit_flag'] = np.zeros(len(self.results['RA']), dtype=int)  # 0: invalid, 1: valid

            for index, log in enumerate(self.results['IMFIT_logs_rbm05']):
                try:
                    row = log.iloc[0] # pipeline中默认当成单源处理，取第一行
                    if not np.isfinite(
                        row[required_fit_fields].astype(float)
                    ).all():
                        continue
                    rbm05_data['Imfit_Flux'][index] = row['I']
                    rbm05_data['Imfit_Flux_err'][index] = row['Ierr']
                    rbm05_data['Peak_Intensity'][index] = row['Peak']
                    rbm05_data['Peak_Intensity_err'][index] = row['PeakErr']
                    rbm05_data['deconmajFWHM'][index] = row['DeconMaj']
                    rbm05_data['deconmajFWHM_err'][index] = row['DeconMajErr']
                    rbm05_data['deconminFWHM'][index] = row['DeconMin']
                    rbm05_data['deconminFWHM_err'][index] = row['DeconMinErr']
                    rbm05_data['deconPA'][index] = row['DeconPA']
                    rbm05_data['deconPA_err'][index] = row['DeconPAErr']
                    rbm05_data['imfit_ra'][index] = row['LongICRS']
                    rbm05_data['imfit_ra_err'][index] = row['LongICRSerr']
                    rbm05_data['imfit_dec'][index] = row['LatICRS']
                    rbm05_data['imfit_dec_err'][index] = row['LatICRSerr']
                    rbm05_data['validfit_flag'][index] = 1  # 标记为有效拟合
                except:
                    # 如果日志无效或格式不对，保持默认值（0或NaN）
                    continue
            
            df_rbm05 = pd.DataFrame(rbm05_data)
                
            csv_path_rbm05 = os.path.join(out_dir, f"{self.cluster_name}_results_rbm05.csv")
            df_rbm05.to_csv(csv_path_rbm05, index=False)
            print(f"RBM05 results saved to: {csv_path_rbm05}")
