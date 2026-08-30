import numpy as np  
from astropy import units as u
from astropy.modeling.models import Gaussian2D
from astrodendro import Dendrogram
import matplotlib.pyplot as plt


class DendrogramTools:    
    def __init__(self,img_array, BEAM_MAJOR, BEAM_MINOR, BEAM_PA, PIXEL_SCALE, distance):
        self.img_array = img_array
        self.BEAM_MAJOR = BEAM_MAJOR # astropy.units.Quantity
        self.BEAM_MINOR = BEAM_MINOR # astropy.units.Quantity
        self.BEAM_PA = BEAM_PA # astropy.units.Quantityß
        self.PIXEL_SCALE = PIXEL_SCALE # astropy.units.Quantity，表示一个pixel是多少arcsec
        self.distance = distance

    # dendrogram analysis
    @staticmethod
    def create_gaussian_source_image(
        peak_factor: float,
        rms_noise: float,
        beam_maj: u.Quantity,
        beam_min: u.Quantity,
        beam_pa: u.Quantity,
        pixel_scale: u.Quantity,
        image_size_pix: int = 100
    ) -> np.ndarray:
        """
        生成一张包含单个理想高斯点源的二维图像。
        
        这个高斯源的形状由指定的波束决定，其峰值亮度是噪声水平的倍数。
        这张图像是理想的，不包含额外添加的噪声。

        参数 (Parameters)
        ----------
        peak_factor : float
            源的峰值亮度是噪声水平的多少倍。例如，对于一个 6σ 的源，此值为 6.0。
        rms_noise : float
            地图的 1σ 噪声水平值 (RMS noise)。源的实际峰值振幅将是 peak_factor * rms_noise。
        beam_maj : astropy.units.Quantity
            波束的主轴 (BMAJ)，必须是带有单位的 astropy Quantity 对象 (例如 1.5 * u.arcsec)。
        beam_min : astropy.units.Quantity
            波束的次轴 (BMIN)，必须是 astropy Quantity 对象。
        beam_pa : astropy.units.Quantity
            波束的位置角 (BPA)，必须是 astropy Quantity 对象 (例如 30 * u.deg)。
        pixel_scale : astropy.units.Quantity
            单个像素的角大小，必须是 astropy Quantity 对象 (例如 0.2 * u.arcsec / u.pix)。
        image_size_pix : int, optional
            生成的正方形图像的边长（以像素为单位），默认为 100。

        返回 (Returns)
        -------
        numpy.ndarray
            一个二维 NumPy 数组，代表了包含高斯源的理想图像。
        """
        FWHM_TO_STDDEV = 1 / (2 * np.sqrt(2 * np.log(2)))
        # 1. 计算源的实际峰值振幅
        amplitude = peak_factor * rms_noise

        # 2. 将波束的 FWHM (半高全宽) 转换为高斯函数的标准差 (sigma)
        #    同时将单位从角秒等转换为像素
        stddev_maj_pix = (beam_maj / pixel_scale).to_value(u.dimensionless_unscaled) * FWHM_TO_STDDEV
        stddev_min_pix = (beam_min / pixel_scale).to_value(u.dimensionless_unscaled) * FWHM_TO_STDDEV
        
        # 3. 确定源在图像中的中心位置
        center_x = image_size_pix / 2
        center_y = image_size_pix / 2

        # 4. 创建一个 astropy Gaussian2D 模型实例
        #    注意：astropy 的 Gaussian2D 模型的 theta (角度) 是从正 x 轴逆时针旋转的，
        #    而天文学的 BPA 通常是从正 y 轴 (北) 向东 (逆时针) 旋转。
        #    因此，我们需要进行转换：theta_astropy = (90 - BPA_astro)
        #    我们在这里直接使用 astropy 单位库来处理角度，更安全。
        theta_for_model = (90 * u.deg - beam_pa).to(u.rad) # 转换为弧度以供模型使用

        gaussian_model = Gaussian2D(
            amplitude=amplitude,
            x_mean=center_x,
            y_mean=center_y,
            x_stddev=stddev_maj_pix,
            y_stddev=stddev_min_pix,
            theta=theta_for_model.value
        )

        # 5. 创建图像的像素网格
        y, x = np.mgrid[0:image_size_pix, 0:image_size_pix]

        # 6. 在网格上计算高斯模型的值，生成图像
        image_data = gaussian_model(x, y)
        
        return image_data

    def get_area2beam_ratio(self,RMS_NOISE_LEVEL,PEAK_SIGMA_FACTOR = 6.0,image_size_pix=100,show=False):
        ideal_image = self.create_gaussian_source_image(
            peak_factor=PEAK_SIGMA_FACTOR,
            rms_noise=RMS_NOISE_LEVEL,
            beam_maj=self.BEAM_MAJOR,
            beam_min=self.BEAM_MINOR,
            beam_pa=self.BEAM_PA,
            pixel_scale=self.PIXEL_SCALE,
            image_size_pix=image_size_pix  # 使用一个稍大的图像以便看得更清楚
        )
        self.RMS_NOISE_LEVEL = RMS_NOISE_LEVEL
        # --- 3. 可视化结果 (可选，但强烈推荐) ---
        if show:
            fig,ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(ideal_image, origin='lower', cmap='viridis')
            plt.colorbar(im,label=f'Flux (in units of {RMS_NOISE_LEVEL})')
            ax.set_xlabel('X Pixel')
            ax.set_ylabel('Y Pixel')
            ax.invert_xaxis()
            ax.grid(False)
            plt.show()
            
        d_ideal = Dendrogram.compute(ideal_image, min_value=5 * RMS_NOISE_LEVEL
                , min_delta=0)
        majpix = self.BEAM_MAJOR / self.PIXEL_SCALE
        minpix = self.BEAM_MINOR / self.PIXEL_SCALE
        Theta_beam = (np.pi * majpix * minpix) / (4 * np.log(2))
        ratio_ab = d_ideal.trunk[0].get_npix() / Theta_beam.value
        self.ratio_ab = ratio_ab
        self.theta_beam = Theta_beam.value
        return ratio_ab
    
    def run_Dendrogram(self,min_value_f=5.0,min_delta_f=1.0,min_npix=None,show=False):
        img_usage = self.img_array #getattr(self, "img_cutout", self.img)
        if min_npix is None:
            min_npix = self.ratio_ab * self.theta_beam
        d = Dendrogram.compute(img_usage, min_value=min_value_f * self.RMS_NOISE_LEVEL
                , min_delta=min_delta_f * self.RMS_NOISE_LEVEL
                , min_npix=min_npix)
        self.d = d
        metadata = {}
        metadata['data_unit'] = u.Jy / u.beam  #beam 是 IrreducibleUnit (不可简化的单位)
        metadata['spatial_scale'] =  self.yresol * 3600 * u.arcsec
        metadata['beam_major'] =  self.majFWHM * 3600 * u.arcsec # FWHM
        metadata['beam_minor'] =  self.minFWHM * 3600 * u.arcsec # FWHM
        # cat = pp_catalog(d, metadata)
        self.metadata = metadata
        # self.cat = cat
        if show:
            v = d.viewer()
            v.show()
        return d

class DendrogramTools_v2:    
    def __init__(self, img_array, BEAM_MAJOR, BEAM_MINOR=None, BEAM_PA=0*u.deg,
                 PIXEL_SCALE=None, distance=None, data_unit=u.Jy/u.beam, wcs=None):
        self.img_array = np.asarray(img_array, dtype=float)
        if self.img_array.ndim != 2:
            raise ValueError("img_array 需要是二维 array。")

        if PIXEL_SCALE is None:
            raise ValueError("请给 PIXEL_SCALE，例如 0.02*u.arcsec。")

        if distance is not None:
            distance_pc = distance.to_value(u.pc) if hasattr(distance, "to") else float(distance)
        else:
            distance_pc = None

        try:
            self.BEAM_MAJOR = BEAM_MAJOR.to(u.arcsec)
        except u.UnitConversionError:
            if distance_pc is None:
                raise ValueError("BEAM_MAJOR 如果用 au 等长度单位，需要同时给 distance。")
            self.BEAM_MAJOR = (BEAM_MAJOR.to_value(u.au) / distance_pc) * u.arcsec

        if BEAM_MINOR is None:
            BEAM_MINOR = BEAM_MAJOR
        try:
            self.BEAM_MINOR = BEAM_MINOR.to(u.arcsec)
        except u.UnitConversionError:
            if distance_pc is None:
                raise ValueError("BEAM_MINOR 如果用 au 等长度单位，需要同时给 distance。")
            self.BEAM_MINOR = (BEAM_MINOR.to_value(u.au) / distance_pc) * u.arcsec

        self.BEAM_PA = BEAM_PA.to(u.deg)
        try:
            self.PIXEL_SCALE = PIXEL_SCALE.to(u.arcsec) # 一个 pixel 对应多少 arcsec
        except u.UnitConversionError:
            if distance_pc is None:
                raise ValueError("PIXEL_SCALE 如果用 au 等长度单位，需要同时给 distance。")
            self.PIXEL_SCALE = (PIXEL_SCALE.to_value(u.au) / distance_pc) * u.arcsec

        self.distance = distance.to(u.pc) if hasattr(distance, "to") else distance
        self.data_unit = data_unit
        self.wcs = wcs

        if self.distance is not None:
            self.PIXEL_SCALE_AU = self.PIXEL_SCALE.to_value(u.arcsec) * distance_pc * u.au
            self.BEAM_MAJOR_AU = self.BEAM_MAJOR.to_value(u.arcsec) * distance_pc * u.au
            self.BEAM_MINOR_AU = self.BEAM_MINOR.to_value(u.arcsec) * distance_pc * u.au
        else:
            self.PIXEL_SCALE_AU = None
            self.BEAM_MAJOR_AU = None
            self.BEAM_MINOR_AU = None

    # dendrogram analysis
    @staticmethod
    def create_gaussian_source_image(
        peak_factor: float,
        rms_noise: float,
        beam_maj: u.Quantity,
        beam_min: u.Quantity,
        beam_pa: u.Quantity,
        pixel_scale: u.Quantity,
        image_size_pix: int = 100
    ) -> np.ndarray:
        """
        生成一张包含单个理想高斯点源的二维图像。
        
        这个高斯源的形状由指定的波束决定，其峰值亮度是噪声水平的倍数。
        这张图像是理想的，不包含额外添加的噪声。

        参数 (Parameters)
        ----------
        peak_factor : float
            源的峰值亮度是噪声水平的多少倍。例如，对于一个 6σ 的源，此值为 6.0。
        rms_noise : float
            地图的 1σ 噪声水平值 (RMS noise)。源的实际峰值振幅将是 peak_factor * rms_noise。
        beam_maj : astropy.units.Quantity
            波束的主轴 (BMAJ)，必须是带有单位的 astropy Quantity 对象 (例如 1.5 * u.arcsec)。
        beam_min : astropy.units.Quantity
            波束的次轴 (BMIN)，必须是 astropy Quantity 对象。
        beam_pa : astropy.units.Quantity
            波束的位置角 (BPA)，必须是 astropy Quantity 对象 (例如 30 * u.deg)。
        pixel_scale : astropy.units.Quantity
            单个像素的角大小，必须是 astropy Quantity 对象 (例如 0.2 * u.arcsec / u.pix)。
        image_size_pix : int, optional
            生成的正方形图像的边长（以像素为单位），默认为 100。

        返回 (Returns)
        -------
        numpy.ndarray
            一个二维 NumPy 数组，代表了包含高斯源的理想图像。
        """
        FWHM_TO_STDDEV = 1 / (2 * np.sqrt(2 * np.log(2)))
        # 1. 计算源的实际峰值振幅
        amplitude = peak_factor * rms_noise

        # 2. 将波束的 FWHM (半高全宽) 转换为高斯函数的标准差 (sigma)
        #    同时将单位从角秒等转换为像素
        stddev_maj_pix = (beam_maj / pixel_scale).to_value(u.dimensionless_unscaled) * FWHM_TO_STDDEV
        stddev_min_pix = (beam_min / pixel_scale).to_value(u.dimensionless_unscaled) * FWHM_TO_STDDEV
        
        # 3. 确定源在图像中的中心位置
        center_x = image_size_pix / 2
        center_y = image_size_pix / 2

        # 4. 创建一个 astropy Gaussian2D 模型实例
        #    注意：astropy 的 Gaussian2D 模型的 theta (角度) 是从正 x 轴逆时针旋转的，
        #    而天文学的 BPA 通常是从正 y 轴 (北) 向东 (逆时针) 旋转。
        #    因此，我们需要进行转换：theta_astropy = (90 - BPA_astro)
        #    我们在这里直接使用 astropy 单位库来处理角度，更安全。
        theta_for_model = (90 * u.deg - beam_pa).to(u.rad) # 转换为弧度以供模型使用

        gaussian_model = Gaussian2D(
            amplitude=amplitude,
            x_mean=center_x,
            y_mean=center_y,
            x_stddev=stddev_maj_pix,
            y_stddev=stddev_min_pix,
            theta=theta_for_model.value
        )

        # 5. 创建图像的像素网格
        y, x = np.mgrid[0:image_size_pix, 0:image_size_pix]

        # 6. 在网格上计算高斯模型的值，生成图像
        image_data = gaussian_model(x, y)
        
        return image_data

    def get_area2beam_ratio(self, RMS_NOISE_LEVEL, PEAK_SIGMA_FACTOR=6.0,
                            image_size_pix=100, min_value_f=5.0, show=False):
        ideal_image = self.create_gaussian_source_image(
            peak_factor=PEAK_SIGMA_FACTOR,
            rms_noise=RMS_NOISE_LEVEL,
            beam_maj=self.BEAM_MAJOR,
            beam_min=self.BEAM_MINOR,
            beam_pa=self.BEAM_PA,
            pixel_scale=self.PIXEL_SCALE,
            image_size_pix=image_size_pix  # 使用一个稍大的图像以便看得更清楚
        )
        self.RMS_NOISE_LEVEL = RMS_NOISE_LEVEL
        # --- 3. 可视化结果 (可选，但强烈推荐) ---
        if show:
            fig,ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(ideal_image, origin='lower', cmap='viridis')
            plt.colorbar(im,label=f'Flux (in units of {RMS_NOISE_LEVEL})')
            ax.set_xlabel('X Pixel')
            ax.set_ylabel('Y Pixel')
            ax.invert_xaxis()
            ax.grid(False)
            plt.show()
            
        d_ideal = Dendrogram.compute(ideal_image, min_value=min_value_f * RMS_NOISE_LEVEL
                , min_delta=0)
        if len(d_ideal.trunk) == 0:
            raise ValueError("假源没有超过 min_value，请调高 PEAK_SIGMA_FACTOR 或降低 min_value_f。")
        majpix = self.BEAM_MAJOR / self.PIXEL_SCALE
        minpix = self.BEAM_MINOR / self.PIXEL_SCALE
        Theta_beam = (np.pi * majpix * minpix) / (4 * np.log(2))
        ratio_ab = d_ideal.trunk[0].get_npix() / Theta_beam.value
        self.ratio_ab = ratio_ab
        self.theta_beam = Theta_beam.value
        self.min_npix_auto = int(np.ceil(ratio_ab * Theta_beam.value))
        return ratio_ab
    
    def convolve_with_beam(self, img_array=None, normalize_kernel=True):
        if img_array is None:
            img_array = self.img_array
        FWHM_TO_STDDEV = 1 / (2 * np.sqrt(2 * np.log(2)))
        sigma_maj_pix = (self.BEAM_MAJOR / self.PIXEL_SCALE).to_value(u.dimensionless_unscaled) * FWHM_TO_STDDEV
        sigma_min_pix = (self.BEAM_MINOR / self.PIXEL_SCALE).to_value(u.dimensionless_unscaled) * FWHM_TO_STDDEV
        theta_rad = (90 * u.deg + self.BEAM_PA).to_value(u.rad)
        beam_kernel = Gaussian2DKernel(
            x_stddev=sigma_maj_pix,
            y_stddev=sigma_min_pix,
            theta=theta_rad
        )
        return convolve_fft(
            img_array,
            beam_kernel,
            normalize_kernel=normalize_kernel,
            nan_treatment='interpolate',
            preserve_nan=True
        )

    def run_Dendrogram(self, min_value_f=5.0, min_delta_f=1.0, min_npix=None,
                       RMS_NOISE_LEVEL=None, show=False):
        img_usage = self.img_array
        if RMS_NOISE_LEVEL is not None:
            self.RMS_NOISE_LEVEL = RMS_NOISE_LEVEL
        if not hasattr(self, "RMS_NOISE_LEVEL"):
            raise ValueError("请先给 RMS_NOISE_LEVEL，或者先运行 get_area2beam_ratio(RMS_NOISE_LEVEL)。")
        if min_npix is None:
            if (not hasattr(self, "ratio_ab")) or (not hasattr(self, "theta_beam")):
                self.get_area2beam_ratio(self.RMS_NOISE_LEVEL, min_value_f=min_value_f)
            min_npix = int(np.ceil(self.ratio_ab * self.theta_beam))

        d = Dendrogram.compute(img_usage, min_value=min_value_f * self.RMS_NOISE_LEVEL
                , min_delta=min_delta_f * self.RMS_NOISE_LEVEL
                , min_npix=min_npix)
        self.d = d
        metadata = {}
        metadata['data_unit'] = self.data_unit  # beam 是 IrreducibleUnit (不可简化的单位)
        metadata['spatial_scale'] = self.PIXEL_SCALE
        metadata['beam_major'] = self.BEAM_MAJOR # FWHM
        metadata['beam_minor'] = self.BEAM_MINOR # FWHM
        cat = pp_catalog(d, metadata)
        self.metadata = metadata
        self.cat = cat
        self.dendrogram_settings = {
            'RMS_NOISE_LEVEL': self.RMS_NOISE_LEVEL,
            'min_value': min_value_f * self.RMS_NOISE_LEVEL,
            'min_delta': min_delta_f * self.RMS_NOISE_LEVEL,
            'min_npix': min_npix,
            'theta_beam_pix': self.theta_beam,
            'ratio_ab': self.ratio_ab,
            'pixel_scale_arcsec': self.PIXEL_SCALE.to_value(u.arcsec),
            'beam_major_arcsec': self.BEAM_MAJOR.to_value(u.arcsec),
            'beam_minor_arcsec': self.BEAM_MINOR.to_value(u.arcsec),
            'beam_pa_deg': self.BEAM_PA.to_value(u.deg),
            'pixel_scale_au': None if self.PIXEL_SCALE_AU is None else self.PIXEL_SCALE_AU.to_value(u.au),
            'beam_major_au': None if self.BEAM_MAJOR_AU is None else self.BEAM_MAJOR_AU.to_value(u.au),
            'beam_minor_au': None if self.BEAM_MINOR_AU is None else self.BEAM_MINOR_AU.to_value(u.au),
        }
        if show:
            v = d.viewer()
            v.show()
        return d

    def plot_core(self, d=None, img_array=None, RMS_NOISE_LEVEL=None,
                  ra_pix=None, dec_pix=None, xlim=None, ylim=None,
                  contour_sigma=5.0, image_vmin_sigma=3.0,
                  vmin=None, vmax=None, cmap='winter', norm='log',
                  norm_kwargs=None,
                  core_color='#ff2f2f', leaf_color=None, core_lw=1.5,
                  contour_color='white', contour_lw=1.0, contour_alpha=0.75,
                  point_color='red', figsize=(10, 10), ax=None,
                  add_colorbar=True, colorbar_label=None,
                  colorbar_size='4%', colorbar_pad=0.03,
                  colorbar_kwargs=None, show=True):
        if d is None:
            d = self.d
        if img_array is None:
            img_array = self.img_array
        if RMS_NOISE_LEVEL is None:
            RMS_NOISE_LEVEL = getattr(self, "RMS_NOISE_LEVEL", None)
        if leaf_color is None:
            leaf_color = core_color
        norm_kwargs = {} if norm_kwargs is None else dict(norm_kwargs)
        colorbar_kwargs = {} if colorbar_kwargs is None else dict(colorbar_kwargs)

        if ax is None:
            fig = plt.figure(figsize=figsize)
            if self.wcs is not None:
                ax = fig.add_subplot(1, 1, 1, projection=self.wcs.celestial)
            else:
                ax = fig.add_subplot(1, 1, 1)
        else:
            fig = ax.figure

        finite_data = img_array[np.isfinite(img_array)]
        if finite_data.size == 0:
            raise ValueError("img_array does not contain finite values.")
        if vmax is None:
            vmax = np.nanmax(finite_data)
        if vmin is None:
            if RMS_NOISE_LEVEL is not None:
                vmin = image_vmin_sigma * RMS_NOISE_LEVEL
            else:
                positive_data = finite_data[finite_data > 0]
                vmin = np.nanpercentile(positive_data, 1) if positive_data.size > 0 else np.nanmin(finite_data)

        image_norm = None
        if norm is not None:
            if isinstance(norm, str):
                norm_name = norm.lower()
                if norm_name in ('none', 'linear-none'):
                    image_norm = None
                elif norm_name == 'log':
                    if vmin <= 0:
                        positive_data = finite_data[finite_data > 0]
                        if positive_data.size > 0:
                            vmin = np.nanmin(positive_data)
                        else:
                            norm_name = 'linear'
                    if norm_name == 'log':
                        norm_kwargs.setdefault('clip', True)
                        image_norm = LogNorm(vmin=vmin, vmax=vmax, **norm_kwargs)
                if norm_name in ('linear', 'normalize'):
                    norm_kwargs.setdefault('clip', False)
                    image_norm = Normalize(vmin=vmin, vmax=vmax, **norm_kwargs)
                elif norm_name in ('power', 'powernorm'):
                    norm_kwargs.setdefault('gamma', 0.5)
                    norm_kwargs.setdefault('clip', False)
                    image_norm = PowerNorm(vmin=vmin, vmax=vmax, **norm_kwargs)
                elif image_norm is None and norm_name not in ('none', 'linear-none'):
                    raise ValueError("norm must be one of log, power, linear, None, or a matplotlib norm object.")
            else:
                image_norm = norm

        if image_norm is None:
            im = ax.imshow(img_array, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, zorder=1)
        else:
            im = ax.imshow(img_array, origin='lower', cmap=cmap, norm=image_norm, zorder=1)

        if RMS_NOISE_LEVEL is not None and contour_sigma is not None:
            level = contour_sigma * RMS_NOISE_LEVEL
            if np.nanmin(img_array) < level < np.nanmax(img_array):
                ax.contour(img_array, levels=[level], colors=contour_color,
                           linewidths=contour_lw, alpha=contour_alpha, zorder=3)

        if ra_pix is not None and dec_pix is not None:
            ax.plot(ra_pix, dec_pix, marker='x', color=point_color,
                    markersize=3, linestyle='None', zorder=5)

        p = d.plotter()
        for structure in d.leaves:
            p.plot_contour(ax, structure=structure.idx, linewidths=core_lw,
                           colors=leaf_color, zorder=4)

        if add_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size=colorbar_size,
                                      pad=colorbar_pad, axes_class=plt.Axes)
            cbar = fig.colorbar(im, cax=cax, **colorbar_kwargs)
            if colorbar_label is None:
                colorbar_label = f'Flux density ({self.data_unit})'
            cbar.set_label(colorbar_label)
            cbar.ax.tick_params(direction='in')

        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        if self.wcs is not None:
            ax.set_xlabel('R.A.', fontsize=20)
            ax.set_ylabel('Dec.', fontsize=20)
            ax.coords[0].display_minor_ticks(True)
            ax.coords[1].display_minor_ticks(True)
        else:
            ax.set_xlabel('X Pixel', fontsize=20)
            ax.set_ylabel('Y Pixel', fontsize=20)

        ax.set_aspect('equal', adjustable='datalim')
        ax.tick_params(axis='both', which='major', direction='in',
                       length=10, width=2, labelsize=20)
        ax.tick_params(axis='both', which='minor', direction='in', length=5)

        if show:
            plt.show()
        return fig, ax

