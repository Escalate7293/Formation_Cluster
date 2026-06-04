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

    
