import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.io import fits
import os
from astropy.wcs import WCS
from astropy.visualization import wcsaxes
from matplotlib.colors import LogNorm,PowerNorm
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib.patheffects import withStroke
from astropy.stats import sigma_clipped_stats
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.image as mpimg
import astropy.constants as cons
from astropy.coordinates import Angle
from astropy import units as u
from lmfit.models import GaussianModel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Ellipse,Rectangle,Circle
from matplotlib.ticker import MaxNLocator,FormatStrFormatter,FuncFormatter,LogLocator
import matplotlib.ticker as ticker
from astropy.nddata import Cutout2D
import pandas as pd
import matplotlib.font_manager as fm
from astropy.coordinates import SkyCoord
import glob
from uncertainties import ufloat,unumpy
from regions import Regions,PixCoord,RectanglePixelRegion
import sqlite3
from astropy.table import Table
import sep
import re
from astropy.visualization import simple_norm
from scipy.optimize import curve_fit
from tqdm import tqdm

from astrodendro import Dendrogram,pp_catalog
from astrodendro.scatter import Scatter
from astropy.modeling.models import Gaussian2D
from astrodendro.analysis import PPStatistic

from pywavan import powspec
from pywavan import fan_trans

from lifelines import KaplanMeierFitter

import networkx as nx

from pathlib import Path

from adjustText import adjust_text

from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.nddata import Cutout2D

from photutils.aperture import EllipticalAperture
from astropy.stats import mad_std
from matplotlib.lines import Line2D

from itertools import combinations

from photutils.aperture import EllipticalAperture
from photutils.background import Background2D
from astropy.stats import SigmaClip,sigma_clip

from photutils.aperture import EllipticalAperture
from photutils.background import Background2D
from astropy.stats import SigmaClip,sigma_clip

from photutils.aperture import CircularAperture, aperture_photometry
from scipy.ndimage import map_coordinates

from sedcreator import SedFluxer
from astropy.coordinates import SkyCoord


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

# # 测试 colormap
# fig, ax = plt.subplots(figsize=(6, 1))
# cb = plt.colorbar(plt.cm.ScalarMappable(cmap=hue_sat_value2_cmap), cax=ax, orientation='horizontal')
# plt.show()


class Formation_Cluster:
    def __init__(self,url,distance=None):
        filename = Path(url).name
        self.filename = filename
        hdu = fits.open(url)
        self.hdu = hdu
        self.head = hdu[0].header
        head = self.head
        wcs = WCS(head)
        self.wcs = wcs
        self.xaxis0 = head['CRVAL1']
        self.xresol = head['CDELT1']
        self.yaxis0 = head['CRVAL2']
        self.yresol = head['CDELT2']
        self.Freq = head['CRVAL3']
        self.ra = head['OBSRA']
        self.dec = head['OBSDEC']
        if head['NAXIS'] == 4:
            self.img = hdu[0].data[0][0]
            width_ful,length_ful = np.shape(self.img)
        elif head['NAXIS'] == 2:
            self.img = hdu[0].data
            width_ful,length_ful = np.shape(self.img)
        # self.img[np.isnan(self.img)] = 0
        self.length_ful = length_ful
        self.width_ful = width_ful
        #图的像素大小，源在中间方便索引
        self.xnpixels = head['NAXIS1']
        self.ynpixels = head['NAXIS2']
        #添加该图的psf信息
        self.majFWHM = head['BMAJ']#好像是以角秒作为单位啊
        self.minFWHM = head['BMIN']
        self.psfPA = head['BPA']#以度作为单位
        #观测时间
        self.data_obs = head['DATE-OBS']
        #beam 对应的 solid angle
        sr_pixel = np.abs(self.xresol * self.yresol) * (np.pi/180)**2
        sr_beam = (np.abs(np.pi * (self.majFWHM/2) * (self.minFWHM/2)) * (np.pi/(180))**2) / (np.log(2))
        self.sr_beam = sr_beam
        self.sr_pixel = sr_pixel
        self.distance = distance  # in pc

        #in astropy units
        BEAM_MAJOR = self.majFWHM * 3600 * u.arcsec
        BEAM_MINOR = self.minFWHM * 3600 * u.arcsec
        BEAM_PA = self.psfPA * u.deg
        PIXEL_SCALE = self.yresol * 3600 * u.arcsec
        self.BEAM_MAJOR = BEAM_MAJOR
        self.BEAM_MINOR = BEAM_MINOR
        self.BEAM_PA = BEAM_PA
        self.PIXEL_SCALE = PIXEL_SCALE

        self.getsf_cat = {}
        self.getsf_ra_cen = {}
        self.getsf_dec_cen = {}
        self.getsf_majFWHM = {}
        self.getsf_minFWHM = {}

    
    def Brightness_Temperature(self,intensity,freq):
        T_bright = 10 **(-26) * intensity * cons.c.value**2 / (2 * cons.k_B.value * freq**2 * self.sr_beam)
        return T_bright

    def get_cutout_ins(self,ra_min=113.040875,ra_max=113.0406958,dec_min=-16.970138889,dec_max=-16.96994444):
        #ax.imshow(self.img,origin='lower')
        #ax.show()
        '''xlower = int(input('xlower:'))
        xupper = int(input('xupper:'))
        ylower = int(input('ylower:'))
        yupper = int(input('yupper:'))'''
        # 设置要切片的赤经和赤纬范围
        # ra_min, ra_max = 113.040875, 113.0406958  # 设置赤经范围（单位：度）
        # dec_min, dec_max = -16.970138889, -16.96994444  # 设置赤纬范围（单位：度）
        # 将赤经和赤纬范围转换为对应的像素坐标
        print(self.wcs.all_world2pix(ra_min, dec_min, 0,0,0))
        x_min, y_min,useless1,useless2 = self.wcs.all_world2pix(ra_min, dec_min, 0,0,0)
        x_max, y_max,useless3,useless4 = self.wcs.all_world2pix(ra_max, dec_max, 0,0,0)
        print(x_min,y_min,x_max,y_max)
        # 将像素坐标转换为整数，获取切片的索引
        xlower = int(np.floor(x_min))
        xupper = int(np.ceil(x_max))
        ylower = int(np.floor(y_min))
        yupper = int(np.ceil(y_max))
        self.wcs.wcs.crpix[0] -= xlower  # 更新原点横坐标
        self.wcs.wcs.crpix[1] -= ylower
        # 切片
        '''xlower = int(self.xnpixels/2 - 50)
        xupper = int(self.xnpixels/2 + 50)
        ylower = int(self.ynpixels/2 - 50)
        yupper = int(self.ynpixels/2 + 50)'''
        self.cutoutsize = np.array([xlower,xupper,ylower,yupper])
        self.img_cutout = self.img[ylower:yupper,xlower:xupper]
        mean, median, std = sigma_clipped_stats(self.img_cutout, sigma=3.0)
        self.cut_out_std = std
        self.cut_out_max = np.max(self.img_cutout)
        self.region_ins = np.array([xlower,xupper,ylower,yupper])
        self.ra_lim = np.array([ra_min,ra_max])
        self.dec_lim = np.array([dec_min,dec_max])
        return None

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
        img_usage = getattr(self, "img_cutout", self.img)
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
        cat = pp_catalog(d, metadata)
        self.metadata = metadata
        self.cat = cat
        if show:
            v = d.viewer()
            v.show()
        return d
    
    def plot_leaves(self,cmap='Blues', fontsize = 25,show_ellipse=True,vmin=None,vmax=None):
        fig, ax = plt.subplots(figsize=(20,12),subplot_kw={'projection': self.wcs.celestial})         # X、Y轴标签字体大小
        plt.rcParams['xtick.labelsize'] = fontsize
        plt.rcParams['ytick.labelsize'] = fontsize
        # X、Y轴刻度标签字体大小
        plt.rcParams['axes.labelsize'] = fontsize
        T_img = self.Brightness_Temperature(self.img,self.Freq)
        mean,median,std = sigma_clipped_stats(T_img,sigma=3.0)
        if vmin == None:
            vmin = 3 * std
        if vmax == None:
            vmax = T_img.max()
        norm1 = ImageNormalize(stretch=LogStretch(),vmin=3 * std,vmax=vmax)
        imshow1 = ax.imshow(T_img,norm=norm1,origin='lower',cmap=cmap)
        for i,j in enumerate(self.d.leaves):
            y,x = np.array(j.get_peak()[0])
            cat_this = self.cat[self.d.leaves[i].idx]
            x_cen,y_cen = cat_this['x_cen'],cat_this['y_cen']
            if show_ellipse == True:
                pps = PPStatistic(j, self.metadata)
                ellipse_this = pps.to_mpl_ellipse()
                ellipse_this.set_width(ellipse_this.width / self.PIXEL_SCALE.value)
                ellipse_this.set_height(ellipse_this.height / self.PIXEL_SCALE.value)
                ellipse_this.set_edgecolor('red')
                ellipse_this.set_facecolor('none')
                ellipse_this.set_linewidth(0.5)
                ax.add_patch(ellipse_this)
            ax.plot(x_cen,y_cen,marker='+',color='black',markersize=5,linestyle='None'
                        ,markerfacecolor='black',markeredgewidth=0.5)

        effect = withStroke(linewidth=2, foreground='red')
        wcsaxes.add_beam(ax=ax,header=self.head,pad=2,path_effects=[effect])
        ax.set_xlabel('R.A.')
        ax.set_ylabel('Dec.')
        ax.tick_params(axis='both', length=8, width=2,direction='in')
        ax.minorticks_on()
        ax.set_aspect('equal')
        cb1 = plt.colorbar(imshow1, pad=0, aspect=20,fraction=0.1)
        cb1.set_ticks(LogLocator(base=10.0))  # 使 colorbar 刻度以 log 方式显示
        cb1.ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=[0.2,0.3,0.4, 0.5,0.6, 0.7,0.8,0.9], numticks=10))
        cb1.ax.tick_params(labelsize=fontsize, length=8, width=2)  #设置色标刻度字体大小。
        cb1.ax.tick_params(direction='in')
        cb1.ax.xaxis.set_ticks_position('bottom')  # 让刻度移动到下方
        cb1.ax.tick_params(axis="both", which="major", width=2, length=8, direction='in')
        cb1.ax.tick_params(axis="both", which="minor", width=1.5, length=4, direction='in')
        font = {'family' : 'serif','color'  : 'darkred','weight' : 'normal','size'   : fontsize}
        cb1.set_label('Brightness Temperature (K)',fontdict=font) #设置colorbar的标签字体及其大小
        plt.show()
        return None
    
    # 整理收集 leaves 的coords
    # def get_leaves(self,region=None):
    #     leaf_x_coords = np.array([])
    #     leaf_y_coords = np.array([])
    #     for i,j in enumerate(self.d.leaves):
    #         y,x = np.array(j.get_peak()[0])
    #         cat_this = self.cat[self.d.leaves[i].idx]
    #         x_cen,y_cen = cat_this['x_cen'],cat_this['y_cen']
    #         leaf_x_coords = np.append(leaf_x_coords,x_cen)
    #         leaf_y_coords = np.append(leaf_y_coords,y_cen)
    #     self.leaf_x_coords = leaf_x_coords
    #     self.leaf_y_coords = leaf_y_coords

    #     if region is not None:
    #         regs_main = Regions.read(region)
    #         box_main = regs_main.regions[0]
    #         box_main_pixel = box_main.to_pixel(self.wcs.celestial)
    #         leaves_pixel= PixCoord(leaf_x_coords,leaf_y_coords)
    #         ins_in = box_main_pixel.contains(leaves_pixel)
    #         ins_out = ~ins_in
    #         leaf_x_out = leaf_x_coords[ins_out]
    #         leaf_y_out = leaf_y_coords[ins_out]
    #         self.leaf_x_out = leaf_x_out
    #         self.leaf_y_out = leaf_y_out
    #         self.box_main_pixel = box_main_pixel
    #         self.box_main = box_main
    #     elif region is None:
    #         self.leaf_x_out = leaf_x_coords
    #         self.leaf_y_out = leaf_y_coords

    #     return None

    def get_leaves_old(self, region=None):
        """
        提取所有 leaves 的坐标, 并识别出在所有给定区域之外的 leaves。

        参数:
        region (str, list, or None): 
            - None: 不进行区域筛选，所有 leaves 都被视为 'out'。
            - str: 单个 region 文件的路径。
            - list: 多个 region 文件路径的列表。
        """
        
        # --- 第 1 部分：提取所有 leaf 的坐标 ---
        # (这里使用了更高效的列表推导式，而不是循环 np.append)
        
        leaf_x_list = []
        leaf_y_list = []

        for leaf in self.d.leaves:
            # y,x = np.array(leaf.get_peak()[0]) # 你原来的代码，似乎没有使用 y,x
            cat_this = self.cat[leaf.idx]
            x_cen, y_cen = cat_this['x_cen'], cat_this['y_cen']
            leaf_x_list.append(x_cen)
            leaf_y_list.append(y_cen)
            
        self.leaf_x_coords = np.array(leaf_x_list)
        self.leaf_y_coords = np.array(leaf_y_list)

        # --- 第 2 部分：根据区域进行筛选 ---

        # 1. 初始化存储区域的字典
        self.box_main = {}
        self.box_main_pixel = {}

        # 2. 规范化 region 输入，统一为列表
        region_list = []
        if region is None:
            region_list = []
        elif isinstance(region, str):
            region_list = [region] # 单个字符串也转为列表
        elif isinstance(region, (list, tuple)):
            region_list = list(region) # 已经是列表或元组
        else:
            print(f"警告: 'region' 参数类型不支持 ({type(region)})。将忽略区域筛选。")
            region_list = [] # 类型错误，视为空列表

        try:
            leaves_pixel = PixCoord(self.leaf_x_coords, self.leaf_y_coords)
        except Exception as e:
            print(f"错误: 创建 PixCoord 失败 (可能是 self.wcs 未定义): {e}")
            # 无法创建 PixCoord，就无法进行 contains 检查
            self.leaf_x_out = self.leaf_x_coords
            self.leaf_y_out = self.leaf_y_coords
            return None

        # 5. 初始化一个“在任意区域内”的掩码，开始时全为 False
        total_inside_mask = np.zeros(len(self.leaf_x_coords), dtype=bool)

        # 6. 遍历所有 region 文件
        for reg_file_path in region_list:
            try:
                # 读取 region 文件
                regs = Regions.read(reg_file_path)
                
                if not regs:
                    print(f"警告: Region 文件 '{reg_file_path}' 为空。跳过。")
                    continue

                # 按照你之前的逻辑，我们只取文件中的第一个区域
                box_main = regs[0]
                box_main_pixel = box_main.to_pixel(self.wcs.celestial)
                
                # 使用文件名 (不含路径) 作为字典的键
                key = os.path.basename(reg_file_path)
                
                # 存入字典
                self.box_main[key] = box_main
                self.box_main_pixel[key] = box_main_pixel

                # 检查哪些 leaves 在 *这个* 区域内
                ins_in_this_region = box_main_pixel.contains(leaves_pixel)
                
                # 更新总的“内部”掩码
                # 只要点在任何一个区域内，它就为 True
                total_inside_mask = total_inside_mask | ins_in_this_region

            except Exception as e:
                print(f"错误: 处理 region 文件 '{reg_file_path}' 失败: {e}")
                continue # 跳过这个损坏的或无法处理的文件

        # 7. 计算最终的“外部”掩码
        # 'ins_out' 为 True 的点，是那些在 total_inside_mask 中为 False 的点
        ins_out = ~total_inside_mask

        # 8. 应用掩码，获取所有区域之外的 leaves
        self.leaf_x_out = self.leaf_x_coords[ins_out]
        self.leaf_y_out = self.leaf_y_coords[ins_out]

        return None
    
    def get_leaves(self, region=None):
        """
        1. 即使没有 dendrogram (self.d)，也会读取并存储 region 信息。
        2. 如果有 self.d，提取所有 leaves 的坐标, 并识别出在所有给定区域之外的 leaves。

        参数:
        region (str, list, or None): 
            - None: 不进行区域筛选。
            - str: 单个 region 文件的路径。
            - list: 多个 region 文件路径的列表。
        """

        # --- 第 1 部分：处理 Region 文件 (独立于 self.d) ---

        # 1. 初始化存储区域的字典
        self.box_main = {}
        self.box_main_pixel = {}

        # 2. 规范化 region 输入，统一为列表
        region_list = []
        if region is None:
            region_list = []
        elif isinstance(region, str):
            region_list = [region]
        elif isinstance(region, (list, tuple)):
            region_list = list(region)
        else:
            print(f"警告: 'region' 参数类型不支持 ({type(region)})。")
            region_list = []

        # 3. 读取 Region 文件并存入属性
        # 注意：这里需要 self.wcs 存在。如果 self.wcs 也不存在，这里也会报错，
        # 但通常 WCS 比 dendrogram 更基础。
        valid_regions = [] # 用于后续筛选 leaves
        
        for reg_file_path in region_list:
            try:
                # 读取 region 文件
                regs = Regions.read(reg_file_path)
                
                if not regs:
                    print(f"警告: Region 文件 '{reg_file_path}' 为空。跳过。")
                    continue

                # 只取第一个 region
                box_main = regs[0]
                
                # 尝试转换为像素坐标 (依赖 self.wcs)
                if hasattr(self, 'wcs') and self.wcs is not None:
                    try:
                        box_main_pixel = box_main.to_pixel(self.wcs.celestial)
                    except Exception as wcs_e:
                        print(f"警告: 无法将 region 转为像素坐标 (WCS问题): {wcs_e}")
                        box_main_pixel = None
                else:
                    box_main_pixel = None

                # 存入字典
                key = os.path.basename(reg_file_path)
                self.box_main[key] = box_main
                self.box_main_pixel[key] = box_main_pixel
                
                # 只有转换成功的 region 才能用于筛选 leaves
                if box_main_pixel is not None:
                    valid_regions.append(box_main_pixel)

            except Exception as e:
                print(f"错误: 处理 region 文件 '{reg_file_path}' 失败: {e}")
                continue

        # --- 第 2 部分：提取 Leaves 并筛选 (依赖 self.d) ---
        
        # 检查 self.d 是否存在
        if not hasattr(self, 'd') or self.d is None:
            print("提示: 未找到 dendrogram (self.d)。仅处理了 Region，未提取 Leaves。")
            # 可以在这里初始化为空数组，防止后续调用报错
            self.leaf_x_coords = np.array([])
            self.leaf_y_coords = np.array([])
            self.leaf_x_out = np.array([])
            self.leaf_y_out = np.array([])
            return None

        # 如果 self.d 存在，继续执行原来的逻辑
        
        # 提取坐标
        leaf_x_list = []
        leaf_y_list = []

        for leaf in self.d.leaves:
            cat_this = self.cat[leaf.idx]
            x_cen, y_cen = cat_this['x_cen'], cat_this['y_cen']
            leaf_x_list.append(x_cen)
            leaf_y_list.append(y_cen)
            
        self.leaf_x_coords = np.array(leaf_x_list)
        self.leaf_y_coords = np.array(leaf_y_list)

        # 创建 PixCoord 对象
        if len(self.leaf_x_coords) > 0:
            leaves_pixel = PixCoord(self.leaf_x_coords, self.leaf_y_coords)
            
            # 初始化 mask
            total_inside_mask = np.zeros(len(self.leaf_x_coords), dtype=bool)

            # 使用刚才解析好的 valid_regions 进行筛选
            for region_pixel in valid_regions:
                ins_in_this_region = region_pixel.contains(leaves_pixel)
                total_inside_mask = total_inside_mask | ins_in_this_region

            ins_out = ~total_inside_mask
            self.leaf_x_out = self.leaf_x_coords[ins_out]
            self.leaf_y_out = self.leaf_y_coords[ins_out]
        else:
            self.leaf_x_out = np.array([])
            self.leaf_y_out = np.array([])

        return None
    
    # 对于allchan为了防止中央区域拥挤，又防止加了region限制导致遗漏，旋转留下的leaves是trunk=leaf的那些
    def get_trunk_leaves(self):
        """
        从 dendrogram 的 trunk 中筛选出本身就是 leaf 的结构，
        并提取其中心像素坐标，存为 self.leaf_x_out / self.leaf_y_out。

        与基于 region 的 get_leaves() 为互斥逻辑分支。
        """

        # 1. 初始化存储区域的字典
        self.box_main = {}
        self.box_main_pixel = {}

        # --- 0. 检查 dendrogram 是否存在 ---
        if not hasattr(self, 'd') or self.d is None:
            print("提示: 未找到 dendrogram (self.d)，无法提取 trunk-level leaves。")
            self.leaf_x_out = np.array([])
            self.leaf_y_out = np.array([])
            return None

        # --- 1. 从 trunk 中筛选 leaf ---
        trunk_leaves = []
        for j in self.d.trunk:
            if j.is_leaf:
                trunk_leaves.append(j)

        # --- 2. 提取中心坐标（依赖 self.cat） ---
        leaf_x_list = []
        leaf_y_list = []

        for leaf in trunk_leaves:
            cat_this = self.cat[leaf.idx]
            leaf_x_list.append(cat_this['x_cen'])
            leaf_y_list.append(cat_this['y_cen'])

        # --- 3. 存为 numpy array ---
        self.leaf_x_out = np.array(leaf_x_list)
        self.leaf_y_out = np.array(leaf_y_list)

        return None

    def get_trunk_leaves_2(self):
        """
        从 dendrogram 的 trunk 中筛选出顶层叶子：
        - trunk 本身是 leaf
        - trunk 是 branch，但其直接子节点都是 leaf（没有 branch 子节点）
        
        提取这些 leaf 的中心像素坐标，存为 self.leaf_x_out / self.leaf_y_out。
        
        参数:
        region (str, list, 或 None): 为保持接口一致，传入但不使用。
        """

        # --- 0. 初始化区域字典（保持与 get_leaves 一致） ---
        self.box_main = {}
        self.box_main_pixel = {}

        # --- 1. 检查 dendrogram 是否存在 ---
        if not hasattr(self, 'd') or self.d is None:
            print("提示: 未找到 dendrogram (self.d)，无法提取 trunk-level leaves。")
            self.leaf_x_coords = np.array([])
            self.leaf_y_coords = np.array([])
            self.leaf_x_out = np.array([])
            self.leaf_y_out = np.array([])
            return None

        # --- 2. 筛选 trunk 下的 leaf ---
        trunk_leaf = []

        for j in self.d.trunk:
            if j.is_leaf:
                trunk_leaf.append(j)
            elif j.is_branch:
                # 检查子节点是否还有 branch
                branch_list = [k for k in j.children if k.is_branch]
                if len(branch_list) == 0:
                    # 所有子节点都是 leaf
                    trunk_leaf.extend(j.children)

        # --- 3. 提取中心坐标（依赖 self.cat） ---
        leaf_x_list = [self.cat[leaf.idx]['x_cen'] for leaf in trunk_leaf]
        leaf_y_list = [self.cat[leaf.idx]['y_cen'] for leaf in trunk_leaf]

        # --- 4. 存为 numpy array ---
        self.leaf_x_coords = np.array(leaf_x_list)
        self.leaf_y_coords = np.array(leaf_y_list)
        self.leaf_x_out = np.array(leaf_x_list)
        self.leaf_y_out = np.array(leaf_y_list)

        return None

    def cross_match_outer_region(self, other=None, thre_arcsec=None):
        leaf_ra_out,leaf_dec_out = self.wcs.celestial.all_pix2world(self.leaf_x_out,self.leaf_y_out,0)
        if other is not None:
            leaf_ra_out_other,leaf_dec_out_other = other.wcs.celestial.all_pix2world(other.leaf_x_out,other.leaf_y_out,0)
            leaf_out = SkyCoord(ra=leaf_ra_out*u.deg,dec=leaf_dec_out*u.deg)
            leaf_out_other = SkyCoord(ra=leaf_ra_out_other*u.deg,dec=leaf_dec_out_other*u.deg)
            idx, d2d, d3d = leaf_out.match_to_catalog_sky(leaf_out_other)
            if thre_arcsec is None:
                d2d_thre = np.min([self.BEAM_MAJOR.value,self.BEAM_MINOR.value]) / 2 / 3600
            else:
                d2d_thre = thre_arcsec / 3600
            mask_matched = d2d.value < d2d_thre
            leaf_out_matched_ra_array = leaf_out.ra[mask_matched].degree
            leaf_out_matched_dec_array = leaf_out.dec[mask_matched].degree

            mask_unmatched1 = ~mask_matched
            leaf_out_unmatched_ra = leaf_out.ra[mask_unmatched1].degree
            leaf_out_unmatched_dec = leaf_out.dec[mask_unmatched1].degree
                
            idx_p, d2d_p, d3d_p = leaf_out_other.match_to_catalog_sky(leaf_out)
            mask_matched_p = d2d_p.value < d2d_thre
            mask_unmatched_p = ~mask_matched_p
            leaf_out_unmatched_ra_other = leaf_out_other.ra[mask_unmatched_p].degree
            leaf_out_unmatched_dec_other = leaf_out_other.dec[mask_unmatched_p].degree

            self.leaf_out_matched_ra_array = leaf_out_matched_ra_array
            self.leaf_out_matched_dec_array = leaf_out_matched_dec_array
            self.leaf_out_unmatched_ra = leaf_out_unmatched_ra
            self.leaf_out_unmatched_dec = leaf_out_unmatched_dec
            self.leaf_out_unmatched_ra_other = leaf_out_unmatched_ra_other
            self.leaf_out_unmatched_dec_other = leaf_out_unmatched_dec_other
            self.d2d_thre = d2d_thre
            
            leaf_out_all_ra = np.concatenate([leaf_out_matched_ra_array,leaf_out_unmatched_ra,leaf_out_unmatched_ra_other])
            leaf_out_all_dec = np.concatenate([leaf_out_matched_dec_array,leaf_out_unmatched_dec,leaf_out_unmatched_dec_other])
            self.leaf_out_all_ra = leaf_out_all_ra
            self.leaf_out_all_dec = leaf_out_all_dec

        else:
            self.leaf_out_matched_ra_array = np.array([])
            self.leaf_out_matched_dec_array = np.array([])
            self.leaf_out_unmatched_ra = leaf_ra_out
            self.leaf_out_unmatched_dec = leaf_dec_out
            self.leaf_out_unmatched_ra_other = np.array([])
            self.leaf_out_unmatched_dec_other = np.array([])

            leaf_out_all_ra = self.leaf_out_unmatched_ra
            leaf_out_all_dec = self.leaf_out_unmatched_dec
            self.leaf_out_all_ra = leaf_out_all_ra
            self.leaf_out_all_dec = leaf_out_all_dec

        return None
    
    @staticmethod
    def copy_rectangle_pixel_region(region):
        return RectanglePixelRegion(
            PixCoord(x=region.center.x, y=region.center.y),
            width=region.width,
            height=region.height,
            angle=Angle(region.angle.to('deg').value, 'deg')
        )
    
    # def load_getsf_source_catlog(self,url):
    #     cut_main_cat = pd.read_csv(url
    #                       ,skiprows=114, delim_whitespace=True
    #                       ,names=[
    #                             "NO", "S1", "S2", "SX", "XCO_P", "YCO_P", "WCS_ACOOR", "WCS_DCOOR",
    #                             "FG", "SIGN", "GOOD", "FM01", "SIGNM01", "GOODM01",
    #                             "FXP_BST01", "FXP_ERR01", "FXT_BST01", "FXT_ERR01", "FXT_ALT01",
    #                             "SCALE01", "AFWHM01", "BFWHM01", "ASIZE01", "BSIZE01",
    #                             "THETA01", "FOFA01", "FOOA01", "FOOB01"
    #                         ])
    #     self.getsf_cat = cut_main_cat
    #     getsf_ra_cen = np.array(cut_main_cat['WCS_ACOOR'])
    #     getsf_dec_cen = np.array(cut_main_cat['WCS_DCOOR'])
    #     getsf_majFWHM = np.array(cut_main_cat['AFWHM01']) / self.PIXEL_SCALE.value
    #     getsf_minFWHM = np.array(cut_main_cat['BFWHM01']) / self.PIXEL_SCALE.value
    #     self.getsf_ra_cen = getsf_ra_cen
    #     self.getsf_dec_cen = getsf_dec_cen
    #     self.getsf_majFWHM = getsf_majFWHM
    #     self.getsf_minFWHM = getsf_minFWHM
    #     return None

    def load_getsf_source_catlog(self, url, cat_key='cut_main'):
        """
        加载一个 GETSF 目录文件，并将其存储在以 cat_key 为键的字典中。
        
        参数:
        url (str): 要读取的 CSV 文件的 URL 或路径。
        cat_key (str): 用于在字典中标识这个目录的键 (例如: '1_cat', 'main_cat')。
        """
        
        # 1. 读取数据
        # 注意：我将 'cut_main_cat' 重命名为 'catalog_df'，因为它不再总是 'main'
        catalog_df = pd.read_csv(url
                          ,skiprows=114, delim_whitespace=True
                          ,names=[
                                "NO", "S1", "S2", "SX", "XCO_P", "YCO_P", "WCS_ACOOR", "WCS_DCOOR",
                                "FG", "SIGN", "GOOD", "FM01", "SIGNM01", "GOODM01",
                                "FXP_BST01", "FXP_ERR01", "FXT_BST01", "FXT_ERR01", "FXT_ALT01",
                                "SCALE01", "AFWHM01", "BFWHM01", "ASIZE01", "BSIZE01",
                                "THETA01", "FOFA01", "FOOA01", "FOOB01"
                            ])
        
        # 2. 将 DataFrame 存入字典
        self.getsf_cat[cat_key] = catalog_df
        
        # 3. 提取列数据
        getsf_ra_cen = np.array(catalog_df['WCS_ACOOR'])
        getsf_dec_cen = np.array(catalog_df['WCS_DCOOR'])
        getsf_majFWHM = np.array(catalog_df['AFWHM01']) / self.PIXEL_SCALE.value
        getsf_minFWHM = np.array(catalog_df['BFWHM01']) / self.PIXEL_SCALE.value
        
        # 4. 将提取的数组存入对应的字典中，使用相同的 key
        self.getsf_ra_cen[cat_key] = getsf_ra_cen
        self.getsf_dec_cen[cat_key] = getsf_dec_cen
        self.getsf_majFWHM[cat_key] = getsf_majFWHM
        self.getsf_minFWHM[cat_key] = getsf_minFWHM
        
        # print(f"目录 '{cat_key}' 已加载并处理。") # 添加一个提示
        
        return None # 保持返回 None
    

    ### 也可以选择直接加载raw source 列表 (from Chen Ai)
    def load_raw_source_list(self, csv_url, cat_key='center_region'):
        csv_data = pd.read_csv(csv_url)
        method = np.array(csv_data['Method'])
        ins_getsf = method == 'getsf'
        ins_leaves = method == 'astrodendro'
        self.leaf_out_all_ra_array = np.array(csv_data['RA(deg)'][ins_leaves])
        self.leaf_out_all_dec_array = np.array(csv_data['Dec(deg)'][ins_leaves])
        self.visiual_selection_leaves = np.array(csv_data['Bool'][ins_leaves])

        # 先默认getsf只有一个区域
        self.getsf_ra_cen = {cat_key: np.array(csv_data['RA(deg)'][ins_getsf])}
        self.getsf_dec_cen = {cat_key: np.array(csv_data['Dec(deg)'][ins_getsf])}
        self.visiual_selection_getsf = {cat_key: np.array(csv_data['Bool'][ins_getsf])}
        self.getsf_AFWHM01 = {cat_key: np.array(csv_data['Major(arcsec)'][ins_getsf])}
        self.getsf_BFWHM01 = {cat_key: np.array(csv_data['Minor(arcsec)'][ins_getsf])}
        self.getsf_THETA01 = {cat_key: np.array(csv_data['PA(deg)'][ins_getsf])}
        self.center_key = cat_key
        return None
    

    def show_all_raw_sources(self,cmap=hue_sat_value2_cmap,fontsize=20,vmax=1):
        source_name = self.head['OBJECT']
        fig = plt.figure(figsize=(30,8))       # X、Y轴标签字体大小
        ax =  fig.add_subplot(1,3,1,projection=self.wcs.celestial)

        plt.rcParams['xtick.labelsize'] = fontsize
        plt.rcParams['ytick.labelsize'] = fontsize
        # X、Y轴刻度标签字体大小
        plt.rcParams['axes.labelsize'] = fontsize
        # mean,median,std = sigma_clipped_stats(T_img,sigma=3.0)
        #norm1 = ImageNormalize(stretch=LogStretch(),vmin=std,vmax=vmax)
        T_img_normal = self.Brightness_Temperature(self.img,self.Freq)
        imshow1 = ax.imshow(T_img_normal,vmin=0,vmax=vmax,origin='lower',cmap=cmap)

        # getsf_x_cen,getsf_y_cen = self.wcs.celestial.all_world2pix(self.getsf_ra_cen,self.getsf_dec_cen,0)
        # ax.plot(getsf_x_cen,getsf_y_cen,marker='+',color='blue',markersize=5,linestyle='None',markerfacecolor='none',markeredgewidth=0.5,label='getsf sources')
        # leaf_out_all_x,leaf_out_all_y = self.wcs.celestial.all_world2pix(self.leaf_out_all_ra,self.leaf_out_all_dec,0)
        # ax.plot(leaf_out_all_x,leaf_out_all_y,marker='o',color='blue',markersize=7,linestyle='None',markerfacecolor='none',markeredgewidth=1,label='dendrogram leaves')
        # box_main_pixel_normal = self.box_main.to_pixel(self.wcs.celestial)
        # new_box_main_pixel_normal = self.copy_rectangle_pixel_region(box_main_pixel_normal)
        # new_box_main_pixel_normal.plot(ax=ax,edgecolor='red',facecolor='none',lw=1)

        # ax.legend(loc='upper left',prop = {'size':15})

        self.plot_regions_and_catalogs(ax)

        effect = withStroke(linewidth=2, foreground='grey')
        wcsaxes.add_beam(ax=ax,header=self.head,pad=2,path_effects=[effect])
        ax.set_xlabel('R.A.')
        ax.set_ylabel('Dec.')
        ax.tick_params(axis='both', length=8, width=2,direction='in',color='black',labelcolor='black')
        ax.minorticks_on()
        ax.set_aspect('equal')
        # ax.set_title(source_name + ' Normal',pad=40,fontsize=30)
        ax.set_title(self.filename + '\nAll sources',pad=40,fontsize=30)
        cb1 = plt.colorbar(imshow1, pad=0, aspect=20,fraction=0.1)
        # cb1.set_ticks(LogLocator(base=10.0))  # 使 colorbar 刻度以 log 方式显示
        # cb1.ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=[0.2,0.3,0.4, 0.5,0.6, 0.7,0.8,0.9], numticks=10))
        cb1.ax.tick_params(labelsize=fontsize, length=8, width=2)  #设置色标刻度字体大小。
        cb1.ax.tick_params(direction='in')
        cb1.ax.xaxis.set_ticks_position('bottom')  # 让刻度移动到下方
        cb1.ax.tick_params(axis="both", which="major", width=2, length=8, direction='in')
        cb1.ax.tick_params(axis="both", which="minor", width=1.5, length=4, direction='in')
        font = {'color'  : 'black','weight' : 'normal','size'   : fontsize}
        cb1.set_label('Brightness Temperature (K)',fontdict=font) #设置colorbar的标签字体及其大小
        # plt.subplots_adjust(wspace=0.0)
        plt.show()
        return fig

    def plot_regions_and_catalogs(self, ax):
        """
        在给定的 Matplotlib Axes (ax) 上绘制所有区域、
        它们对应的 GETSF 目录源，以及在外的 leaves。
        
        参数:
        ax (matplotlib.axes.Axes): 要在上面绘图的 Axes 对象。
        """
        
        # --- 1. 绘制所有区域之外的 Leaves (Dendrogram Leaves) ---
        # 这部分似乎是全局的，所以只画一次
        try:
            # 假设 self.leaf_out_all_ra/dec 是世界坐标系 (RA, Dec)
            leaf_out_all_x, leaf_out_all_y = self.wcs.celestial.all_world2pix(
                self.leaf_out_all_ra, self.leaf_out_all_dec, 0
            )
            ax.plot(
                leaf_out_all_x, leaf_out_all_y,
                marker='o', color='blue', markersize=7,
                linestyle='None', markerfacecolor='none',
                markeredgewidth=1, label='Dendrogram Leaves' # 图例1
            )
        except AttributeError:
            print("警告: 未找到 self.leaf_out_all_ra/dec 属性，跳过绘制 leaves。")
        except Exception as e:
            print(f"警告: 绘制 leaves 失败: {e}")

        
        # --- 2. 循环绘制每个区域 (Region) 及其对应的源 (Sources) ---
        # 定义一组颜色，用于区分不同的 区域/目录 对
        colors = ['red', 'green', 'cyan', 'magenta', 'orange', 'purple']
        
        # 用于确保 'GETSF Sources' 图例只显示一次
        has_plotted_source_label = False

        # 遍历 self.box_main 字典 (键和值)
        for i, (key, region_sky) in enumerate(self.box_main.items()):
            
            # 选择一个颜色
            color = colors[i % len(colors)]
            
            # --- A. 绘制区域的边界框 ---
            try:
                # `region_sky` 是存在 self.box_main 里的 SkyRegion 对象
                box_main_pixel = region_sky.to_pixel(self.wcs.celestial)
                
                # [关键] 调用你提供的静态方法
                new_box_main_pixel = self.copy_rectangle_pixel_region(box_main_pixel)
                
                # 绘制
                new_box_main_pixel.plot(ax=ax, edgecolor=color, facecolor='none', lw=1.5)
                
                # [新增] 在区域中心附近添加区域名称 (key)
                ax.text(
                    new_box_main_pixel.center.x,
                    new_box_main_pixel.center.y + new_box_main_pixel.height / 2, # 放在框的上方
                    str(key),
                    color=color,
                    fontsize=14,
                    ha='center',
                    va='bottom'
                )
                
            except Exception as e:
                print(f"错误: 绘制区域 '{key}' 失败: {e}")
                continue # 跳过这个区域

            # --- B. 绘制与该区域对应的 GETSF 目录源 ---
            try:
                # [关键] 假设 self.getsf_ra_cen 中有 *完全相同* 的 'key'
                if key not in self.getsf_ra_cen:
                    print(f"警告: 在 self.getsf_ra_cen 中未找到键 '{key}'。")
                    print(f"    (可用键: {list(self.getsf_ra_cen.keys())})")
                    continue
                
                # 获取这个区域对应的 RA 和 Dec 数组
                ra_this_cat = self.getsf_ra_cen[key]
                dec_this_cat = self.getsf_dec_cen[key]
                
                if len(ra_this_cat) == 0:
                    continue # 如果这个目录是空的，跳过

                # 转换到像素坐标
                getsf_x_cen, getsf_y_cen = self.wcs.celestial.all_world2pix(
                    ra_this_cat, dec_this_cat, 0
                )
                
                # 处理图例：只为第一个添加图例，后续的不添加
                label_for_this_plot = None
                if not has_plotted_source_label:
                    label_for_this_plot = 'GETSF Sources' # 图例2
                    has_plotted_source_label = True
                
                # 绘制
                ax.plot(
                    getsf_x_cen, getsf_y_cen,
                    marker='+', color=color, markersize=5,
                    linestyle='None', markerfacecolor='none',
                    markeredgewidth=0.5, label=label_for_this_plot
                )

            except Exception as e:
                print(f"错误: 绘制 '{key}' 对应的 GETSF sources 失败: {e}")


        # --- 3. 显示图例 ---
        ax.legend(loc='upper left', prop={'size': 15})
        
        return None
    
    def get_getsf_compact_sources(self, cmap='viridis', **kwargs):
        """
        整合了 GETSF 源筛选 (尺寸 + Unsharp Masking SNR)、FITS 文件加载和可视化绘图的功能。

        筛选逻辑 (必须同时满足):
        1. 尺寸筛选: (Geometric Mean of FWHM) <= compact_threshold * BeamSize
        2. 形态筛选: Unsharp Masking 后的 Residual Peak SNR >= residual_snr_threshold

        参数 (**kwargs):
            plot (bool): 是否绘图，默认为 False。
            getsf_work_dir_name (str): getsf 工作目录名，用于寻找 FITS 文件。
            image_type (str): 图片后缀类型。
            compact_threshold (float): 尺寸阈值 (倍数于 beam size)，默认 2.0。
            residual_snr_threshold (float): 残差信噪比阈值，默认 5.0。
            cutout_shape (tuple): 用于计算 residual 的切片大小，默认 (85, 85)。
            kernel_factor (float): Unsharp Masking 高斯核的 sigma 倍数 (相对于 beam sigma)，默认 2.0。
        """
        
        # --- 0. 参数解析 ---
        # show_plots = kwargs.get('plot', False)
        show_hist = kwargs.get('show_hist', False)
        show_selection = kwargs.get('show_selection', True)
        show_sources = kwargs.get('show_sources', True)
        getsf_work_dir_name = kwargs.get('getsf_work_dir_name', 'G025_Band6_TM1+TM2')
        image_type = kwargs.get('image_type', '')
        
        # 核心筛选参数
        compact_threshold = kwargs.get('compact_threshold', 2.0)
        residual_snr_threshold = kwargs.get('residual_snr_threshold', 5.0)
        if self.distance <= 3090:
            size = int(60 * 3090 / self.distance)
        else:
            size = 60
        cutout_shape = kwargs.get('cutout_shape', (size, size))
        kernel_factor = kwargs.get('kernel_factor', 2.0) # 控制平滑核的大小，一般比 Beam 大一点
        sigma_type = kwargs.get('sigma_type', 'MAD') # 控制残差噪声计算方式，'clip','mask','MAD' 目前是这三种

        region_keys = list(self.getsf_cat.keys())
        self.region_keys = region_keys

        # --- 1. 加载 FITS 数据 (现在必须执行，因为计算依赖图像) ---  # 画图的时候要用到
        wcs_cut_all = {}
        img_cut_all = {}
        self.region_keys = region_keys

        # print("正在加载 FITS 文件以进行源筛选...")
        for i, key in enumerate(region_keys):
            # 动态生成目录名
            if i == 0:
                dir_name = 'cut_main' + image_type
                dir_name_ori = 'cut_main'
            else:
                dir_name = f'cut_{i + 1}' + image_type
                dir_name_ori = f'cut_{i + 1}'
            
            # 构建文件路径
            file_path = (f'../getsf/work/' + getsf_work_dir_name + f'/{dir_name}/run/image/'
                            f'{self.filename[0:-5]}.{dir_name_ori}.m.fits')
            
            try:
                with fits.open(file_path) as hdu:
                    header = hdu[0].header
                    wcs_cut_all[key] = WCS(header)
                    # 处理数据维度
                    if header['NAXIS'] == 4:
                        img_cut_all[key] = hdu[0].data[0][0]
                    elif header['NAXIS'] == 3:
                        img_cut_all[key] = hdu[0].data[0]
                    else:
                        img_cut_all[key] = hdu[0].data
            except FileNotFoundError:
                print(f"警告: FITS 文件未找到: {file_path}。该区域的源将无法进行 Residual 筛选。")
                continue

        
        # --- 2. 核心逻辑：双重筛选 ---
        
        self.getsf_compact_sources_id = {}
        size_lists_for_plotting = {} # 仅用于绘图统计
        
        # 预计算 Beam 参数 (pixel 单位)
        size_beam = np.sqrt(self.BEAM_MAJOR.value * self.BEAM_MINOR.value)
        bmaj_pix = self.BEAM_MAJOR.value / self.PIXEL_SCALE.value
        bmin_pix = self.BEAM_MINOR.value / self.PIXEL_SCALE.value
        bpa_deg = self.BEAM_PA.value
        
        # 将 Beam FWHM 转换为 Sigma
        sigma_maj_pix = bmaj_pix / (2 * np.sqrt(2 * np.log(2)))
        sigma_min_pix = bmin_pix / (2 * np.sqrt(2 * np.log(2)))
        
        # Unsharp Masking 的核通常要比 Beam 稍大，这里使用 kernel_factor 控制
        # 注意 astropy Gaussian2DKernel 的 theta 是弧度
        theta_rad = np.deg2rad(90.0 + bpa_deg) 
        smooth_kernel = Gaussian2DKernel(
            x_stddev= kernel_factor * sigma_maj_pix, 
            y_stddev= kernel_factor * sigma_min_pix, 
            theta=theta_rad
        )

        img_data = self.img
        wcs_data = self.wcs
        T_img_full = self.Brightness_Temperature(img_data, self.Freq)

        fig_selection_list = []

        for key in region_keys:

            getsf_cat = self.getsf_cat[key]

            # --- A. 尺寸筛选标准 ---
            majFWHM_list = np.array(getsf_cat['AFWHM01'])
            minFWHM_list = np.array(getsf_cat['BFWHM01'])
            NO_id = np.array(getsf_cat['NO'])
            
            size_list = np.sqrt(majFWHM_list * minFWHM_list)
            size_lists_for_plotting[key] = size_list # 存一下用于画直方图
            
            # 判据 1: 尺寸足够小
            scale_ratio_gaussian = size_list / size_beam
            mask_size = scale_ratio_gaussian <= compact_threshold

            # --- B. Residual SNR 筛选标准 ---
            # 为了保持一致性，先转成亮温 (K) 再计算，和你提供的 snippet 一致
            
            residual_snr_list = []
            
            # 遍历该目录下的每个源
            for idx, row in getsf_cat.iterrows():
                # 获取源的中心坐标 (RA, Dec)
                ra_cen, dec_cen = row['WCS_ACOOR'], row['WCS_DCOOR']
                AFWHM_11 = row['AFWHM01']
                BFWHM_11 = row['BFWHM01']
                THETA_11 = row['THETA01']
                
                # 转为该切片图像的像素坐标
                x_pix, y_pix = wcs_data.celestial.all_world2pix(ra_cen, dec_cen, 0)
                position = (x_pix, y_pix)
                
                # 切片 (Cutout)
                cutout = Cutout2D(T_img_full, position, cutout_shape, mode='partial', wcs=wcs_data.celestial)
                
                # 检查切片是否包含 NaN (如果源在边缘可能会有 NaN)
                if np.isnan(cutout.data).any():
                    # 如果有 NaN，视情况处理。这里简单给个 -1，表示不通过
                    residual_snr_list.append(-1.0)
                    continue
                
                # 卷积 (Unsharp Masking 的平滑部分)
                # convolved_image = convolve_fft(cutout.data, smooth_kernel, preserve_nan=False)
                convolved_image = convolve_fft(cutout.data, smooth_kernel, boundary='fill', fill_value=np.nan, preserve_nan=True)

                # 计算残差图
                residual_img = cutout.data - convolved_image
                
                # 统计残差图的 Sigma (Local RMS)
                # 使用 sigma clipping 去除极值影响，得到背景噪声水平
                if sigma_type == 'clip':
                    _, _, sigma_residual = sigma_clipped_stats(residual_img, sigma=3.0, maxiters=None)
                elif sigma_type == 'mask':
                    AFWHM_11_pix = AFWHM_11 / self.PIXEL_SCALE.value * 3 / (2*np.sqrt(2*np.log(2)))
                    BFWHM_11_pix = BFWHM_11 / self.PIXEL_SCALE.value * 3 / (2*np.sqrt(2*np.log(2)))
                    a_radius = AFWHM_11_pix
                    b_radius = BFWHM_11_pix
                    theta_rad = np.deg2rad(90 + THETA_11) # 保持和你原本绘图逻辑一致的角度
                    xcen_pix_11,ycen_pix_11 = cutout.wcs.all_world2pix(ra_cen, dec_cen, 0)
                    # --- 2. 创建 Aperture 对象 ---
                    position = (xcen_pix_11, ycen_pix_11)
                    aperture = EllipticalAperture(position, a=a_radius, b=b_radius, theta=theta_rad)
                    mask = aperture.to_mask(method='center') 
                    # 将 mask 映射回全图尺寸 (这一步很关键，因为 to_mask 默认只切出包围盒)
                    mask_image = mask.to_image(residual_img.shape) 
                    # 取反：我们需要椭圆“外”的区域，所以 mask_image 为 0 (False) 的地方才是背景
                    # 注意：mask_image 是 1.0 或 0.0，转为 bool 类型更安全
                    background_data = residual_img[mask_image == 0] 
                    sigma_residual = np.std(background_data)
                elif sigma_type == 'MAD':
                    sigma_residual = mad_std(residual_img, ignore_nan=True)
                else:
                    sigma_type == 'MAD'
                    print('We have only three sigma_type: clip, mask, MAD! Use MAD by default.')
                    sigma_residual = mad_std(residual_img, ignore_nan=True)
                
                # 获取中心像素的残差值 (Peak Flux of the Compact Source)
                center_y, center_x = residual_img.shape
                peak_residual = residual_img[center_y//2, center_x//2]
                
                # 计算 SNR
                if sigma_residual > 0:
                    snr_this = peak_residual / sigma_residual
                else:
                    snr_this = 0
                
                residual_snr_list.append(snr_this)
            
            residual_snr_array = np.array(residual_snr_list)
            
            # 判据 2: 残差信噪比足够高
            mask_snr = residual_snr_array >= residual_snr_threshold
            
            # --- C. 综合判据 ---
            final_mask = mask_size & mask_snr
            
            # 筛选出最终的 NO
            selected_NOs = NO_id[final_mask]
            self.getsf_compact_sources_id[key] = selected_NOs
            
            # print(f"区域 {key}: 总源数 {len(NO_id)}, 尺寸筛选后 {np.sum(mask_size)}, "
            #       f"SNR筛选后 {np.sum(mask_snr)}, 最终选中 {len(selected_NOs)}")

            if show_selection:

                all_all = np.arange(1,len(NO_id)+1,1)
                not_selected_all = all_all[~np.isin(all_all,selected_NOs)]

                fig_selection,axall = plt.subplots(1,2,figsize=(24,8))

                ax = axall[0]
                ax.plot(selected_NOs,residual_snr_array[final_mask],marker='o',linestyle='None'
                        ,color='forestgreen',markersize=5,label='Selected Sources')
                ax.plot(not_selected_all,residual_snr_array[not_selected_all-1],marker='o',linestyle='None',color='red',
                        markerfacecolor='none',markersize=5,label='Not Selected Sources')
                ax.axhline(y=residual_snr_threshold, color='gray', linestyle='--',linewidth=1)
                ax.legend(fontsize=20)
                ax.set_yscale('log')
                ax.set_xlabel('Source ID',fontsize=20)
                ax.set_ylabel('Residual Peak / Residual Sigma',fontsize=20)
                ax.minorticks_on()
                ax.tick_params(axis='both', which='major', length=8, width=1.5,direction='in',top=True,right=True)
                ax.tick_params(axis='both', which='minor', length=4, width=1.5,direction='in',top=True,right=True)

                ax = axall[1]

                ax.plot(scale_ratio_gaussian[final_mask],residual_snr_array[final_mask],marker='o',linestyle='None'
                        ,color='forestgreen',markersize=5,label='Selected Sources')
                ax.plot(scale_ratio_gaussian[not_selected_all-1],residual_snr_array[not_selected_all-1],marker='o'
                        ,linestyle='None',color='red',markerfacecolor='none',markersize=5,label='Not Selected Sources')
                ax.axhline(y=residual_snr_threshold, color='gray', linestyle='--',linewidth=1)
                ax.axvline(x=compact_threshold, color='gray', linestyle='-',linewidth=1)
                ax.legend(fontsize=20)
                ax.set_yscale('log')
                # ax.set_xscale('log')
                ax.set_xlabel('Gaussian Scale / Beam Scale',fontsize=20)
                ax.set_ylabel('Residual Peak / Residual Sigma',fontsize=20)
                ax.minorticks_on()
                ax.tick_params(axis='both', which='major', length=8, width=1.5,direction='in',top=True,right=True)
                ax.tick_params(axis='both', which='minor', length=4, width=1.5,direction='in',top=True,right=True)

                fig_selection_list.append(fig_selection)



        # --- 3. 绘图部分 (逻辑不变，只绘制最终筛选结果) ---
        if show_hist:
            # 3.1 绘制尺寸分布直方图 (辅助看分布)
            # print("正在生成源尺寸分布直方图...")
            for key in region_keys:
                if key not in size_lists_for_plotting: continue
                plt.figure(figsize=(10, 6))
                plt.hist(size_lists_for_plotting[key], bins=30, histtype='step', color='black', label=f'Region Key: {key}')
                plt.axvline(size_beam * compact_threshold, color='orange', linestyle='--', label=f'Size Thresh ({compact_threshold}x)')
                plt.title(f'Source Size Distribution (Key: {key})')
                plt.xlabel('Source Size (Geometric Mean of FWHM)')
                plt.legend()
                plt.show()

        fig_plot_list = []

        if show_sources:
            # 3.2 绘制图像和标注
            # print("正在生成各区域的源标注图像...")    


            for region_key in region_keys:
                if region_key not in img_cut_all: continue

                cutout1_cat = self.getsf_cat[region_key]
                NO_selected_main = self.getsf_compact_sources_id[region_key]
                
                wcs_cut_main = wcs_cut_all[region_key]
                img_cut_main = img_cut_all[region_key]
                img_cut_main[np.isnan(img_cut_main)] = 0 # 绘图时简单的 NaN 处理

                PIXEL_SCALE = self.PIXEL_SCALE
                fontsize = 20

                fig_plot, ax = plt.subplots(figsize=(20, 12), subplot_kw={'projection': wcs_cut_main})
                plt.rcParams.update({'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize, 'axes.labelsize': fontsize})
                
                T_img = self.Brightness_Temperature(img_cut_main, self.Freq)

                norm1 = ImageNormalize(stretch=LogStretch(), vmin=max(np.percentile(T_img, 10), 1e-3), vmax=T_img.max())
                imshow1 = ax.imshow(T_img, norm=norm1, origin='lower', cmap=cmap)

                text = []
                num_source = 0
                for i, row in cutout1_cat.iterrows():
                    # 注意：这里为了绘图准确，建议使用 WCS 转 Pix，而不是直接用 catalogue 的 XCO_P
                    # 因为 catalogue 的 XCO_P 坐标系定义可能与 header 略有差异 (1-based vs 0-based)
                    # 但为了兼容你原代码习惯，这里暂时保留原样或微调
                    # 推荐方式:
                    ra, dec = row['WCS_ACOOR'], row['WCS_DCOOR']
                    x_cen, y_cen = wcs_cut_main.celestial.all_world2pix(ra, dec, 0)
                    
                    majFWHM = row['AFWHM01'] / PIXEL_SCALE.value
                    minFWHM = row['BFWHM01'] / PIXEL_SCALE.value
                    theta = row['THETA01']
                    current_NO = row['NO']

                    if current_NO in NO_selected_main:
                        # 选中的源：红色粗圈 + 编号
                        ellipse_this = Ellipse(xy=(x_cen, y_cen), width=majFWHM, height=minFWHM,
                                               angle=90 + theta, edgecolor='red', facecolor='none', linewidth=1)
                        ax.add_patch(ellipse_this)
                        ax.plot(x_cen, y_cen, marker='+', color='red', markersize=8, linestyle='None', markeredgewidth=1)
                        text.append(ax.text(x_cen + 5, y_cen + 5, s=f'{num_source + 1}', color='red', fontsize=12, weight='bold'))
                        num_source += 1
                    else:
                        # 被剔除的源：青色细圈 (可选: 注释掉以保持图像干净)
                        ellipse_this = Ellipse(xy=(x_cen, y_cen), width=majFWHM, height=minFWHM,
                                               angle=90 + theta, edgecolor='black', facecolor='none', linewidth=0.5, alpha=0.5)
                        ax.add_patch(ellipse_this)
                        # ax.plot(x_cen, y_cen, marker='+', color='cyan', markersize=5, linestyle='None', markeredgewidth=0.5, alpha=0.5)
                        # print('not_selected?', current_NO, x_cen, y_cen, residual_snr_array[int(current_NO) - 1])

                if text:
                    adjust_text(text, ax=ax, arrowprops=dict(arrowstyle='->', color='red', lw=0.5))
                
                ax.set_xlabel('R.A.')
                ax.set_ylabel('Dec.')
                ax.set_title(f"Selected Compact Sources (Key: {region_key})\nSize <= {compact_threshold}xBeam & ResSNR >= {residual_snr_threshold}", fontsize=fontsize, pad=20)
                
                # Colorbar
                cb1 = plt.colorbar(imshow1, pad=0.01, aspect=25, fraction=0.08)
                cb1.ax.tick_params(labelsize=fontsize)
                cb1.set_label('Brightness Temperature (K)', fontsize=fontsize)

                plt.show()
                fig_plot_list.append(fig_plot)

        if len(fig_selection_list) == 1:
            fig_selection_list = fig_selection_list[0]
        
        if len(fig_plot_list) == 1:
            fig_plot_list = fig_plot_list[0]

        if show_selection and show_sources:
            return fig_selection_list, fig_plot_list
        if show_selection and (not show_sources):
            return fig_selection_list
        if (not show_selection) and show_sources:
            return fig_plot_list
        else:
            return None

    def get_getsf_compact_sources_2(self, cmap='viridis', **kwargs):
        # 2的意思是从外围导出的center sources的筛选
        """
        整合了 GETSF 源筛选 (尺寸 + Unsharp Masking SNR)、FITS 文件加载和可视化绘图的功能。

        筛选逻辑 (必须同时满足):
        1. 尺寸筛选: (Geometric Mean of FWHM) <= compact_threshold * BeamSize
        2. 形态筛选: Unsharp Masking 后的 Residual Peak SNR >= residual_snr_threshold

        参数 (**kwargs):
            plot (bool): 是否绘图，默认为 False。
            getsf_work_dir_name (str): getsf 工作目录名，用于寻找 FITS 文件。
            image_type (str): 图片后缀类型。
            compact_threshold (float): 尺寸阈值 (倍数于 beam size)，默认 2.0。
            residual_snr_threshold (float): 残差信噪比阈值，默认 5.0。
            cutout_shape (tuple): 用于计算 residual 的切片大小，默认 (85, 85)。
            kernel_factor (float): Unsharp Masking 高斯核的 sigma 倍数 (相对于 beam sigma)，默认 2.0。
        """
        
        # --- 0. 参数解析 ---
        # show_plots = kwargs.get('plot', False)
        show_selection = kwargs.get('show_selection', True)
        show_sources = kwargs.get('show_sources', True)
        
        # 核心筛选参数
        compact_threshold = kwargs.get('compact_threshold', 2.0)
        residual_snr_threshold = kwargs.get('residual_snr_threshold', 5.0)
        if self.distance <= 3090:
            size = int(60 * 3090 / self.distance)
        else:
            size = 60
        cutout_shape = kwargs.get('cutout_shape', (size, size))
        kernel_factor = kwargs.get('kernel_factor', 2.0) # 控制平滑核的大小，一般比 Beam 大一点
        sigma_type = kwargs.get('sigma_type', 'MAD') # 控制残差噪声计算方式，'clip','mask','MAD' 目前是这三种
        
        # --- 1. 加载 FITS 数据 (现在必须执行，因为计算依赖图像) ---  # 画图的时候要用到 
        file_path = kwargs.get('file_path','') # 输入center region 的fits files
        
        try:
            with fits.open(file_path) as hdu:
                header = hdu[0].header
                wcs_cut_all = WCS(header)
                # 处理数据维度
                if header['NAXIS'] == 4:
                    img_cut_all = hdu[0].data[0][0]
                elif header['NAXIS'] == 3:
                    img_cut_all = hdu[0].data[0]
                else:
                    img_cut_all = hdu[0].data
        except FileNotFoundError:
            print(f"警告: FITS 文件未找到: {file_path}。该区域的源将无法进行 Residual 筛选。")

        # --- 2. 核心逻辑：双重筛选 ---
        
        self.getsf_compact_sources_id = {}
        
        # 预计算 Beam 参数 (pixel 单位)
        size_beam = np.sqrt(self.BEAM_MAJOR.value * self.BEAM_MINOR.value)
        bmaj_pix = self.BEAM_MAJOR.value / self.PIXEL_SCALE.value
        bmin_pix = self.BEAM_MINOR.value / self.PIXEL_SCALE.value
        bpa_deg = self.BEAM_PA.value
        
        # 将 Beam FWHM 转换为 Sigma
        sigma_maj_pix = bmaj_pix / (2 * np.sqrt(2 * np.log(2)))
        sigma_min_pix = bmin_pix / (2 * np.sqrt(2 * np.log(2)))
        
        # Unsharp Masking 的核通常要比 Beam 稍大，这里使用 kernel_factor 控制
        # 注意 astropy Gaussian2DKernel 的 theta 是弧度
        theta_rad = np.deg2rad(90.0 + bpa_deg) 
        smooth_kernel = Gaussian2DKernel(
            x_stddev= kernel_factor * sigma_maj_pix, 
            y_stddev= kernel_factor * sigma_min_pix, 
            theta=theta_rad
        )

        img_data = self.img
        wcs_data = self.wcs
        T_img_full = self.Brightness_Temperature(img_data, self.Freq)

        # for key in region_keys:

        ra_cen_all = self.getsf_ra_cen[self.center_key]
        dec_cen_all = self.getsf_dec_cen[self.center_key]
        AFWHM01_all = self.getsf_AFWHM01[self.center_key]
        BFWHM01_all = self.getsf_BFWHM01[self.center_key]
        visual_selection_all = self.visiual_selection_getsf[self.center_key]
        NO_id = np.arange(1,len(ra_cen_all)+1,1)
            
        size_list = np.sqrt(AFWHM01_all * BFWHM01_all)
            
        # 判据 1: 尺寸足够小
        scale_ratio_gaussian = size_list / size_beam
        mask_size = scale_ratio_gaussian <= compact_threshold

        # --- B. Residual SNR 筛选标准 ---
        # 为了保持一致性，先转成亮温 (K) 再计算，和你提供的 snippet 一致
            
        residual_snr_list = []
            
        # 遍历该目录下的每个源
        for idx, ra_cen  in enumerate(ra_cen_all):
            # 获取源的中心坐标 (RA, Dec)
            dec_cen = dec_cen_all[idx]
            
            # 转为该切片图像的像素坐标
            x_pix, y_pix = wcs_data.celestial.all_world2pix(ra_cen, dec_cen, 0)
            position = (x_pix, y_pix)
            
            # 切片 (Cutout)
            cutout = Cutout2D(T_img_full, position, cutout_shape, mode='partial', wcs=wcs_data.celestial)
            
            # 检查切片是否包含 NaN (如果源在边缘可能会有 NaN)
            if np.isnan(cutout.data).any():
                # 如果有 NaN，视情况处理。这里简单给个 -1，表示不通过
                residual_snr_list.append(-1.0)
                continue
            
            # 卷积 (Unsharp Masking 的平滑部分)
            # convolved_image = convolve_fft(cutout.data, smooth_kernel, preserve_nan=False)
            convolved_image = convolve_fft(cutout.data, smooth_kernel, boundary='fill', fill_value=np.nan, preserve_nan=True)

            # 计算残差图
            residual_img = cutout.data - convolved_image
            
            # 统计残差图的 Sigma (Local RMS)
            # 使用 sigma clipping 去除极值影响，得到背景噪声水平
            if sigma_type == 'clip':
                _, _, sigma_residual = sigma_clipped_stats(residual_img, sigma=3.0, maxiters=None)
            elif sigma_type == 'MAD':
                sigma_residual = mad_std(residual_img, ignore_nan=True)
            else:
                sigma_type == 'MAD'
                print('We have only three sigma_type: clip, mask, MAD! Use MAD by default.')
                sigma_residual = mad_std(residual_img, ignore_nan=True)
            
            # 获取中心像素的残差值 (Peak Flux of the Compact Source)
            center_y, center_x = residual_img.shape
            peak_residual = residual_img[center_y//2, center_x//2]
            
            # 计算 SNR
            if sigma_residual > 0:
                snr_this = peak_residual / sigma_residual
            else:
                snr_this = 0
            
            residual_snr_list.append(snr_this)
            
        residual_snr_array = np.array(residual_snr_list)
        
        # 判据 2: 残差信噪比足够高
        mask_snr = residual_snr_array >= residual_snr_threshold
        
        # --- C. 综合判据 ---
        final_mask = mask_size & mask_snr
        
        # 筛选出最终的 NO
        selected_NOs = NO_id[final_mask]
        self.getsf_compact_sources_id[self.center_key] = selected_NOs
        
        # print(f"区域 {key}: 总源数 {len(NO_id)}, 尺寸筛选后 {np.sum(mask_size)}, "
        #       f"SNR筛选后 {np.sum(mask_snr)}, 最终选中 {len(selected_NOs)}")

        if show_selection:

            all_all = NO_id
            not_selected_all = all_all[~np.isin(all_all,selected_NOs)]

            fig_selection,axall = plt.subplots(1,2,figsize=(24,8))
            visual_not_selected = NO_id[(visual_selection_all==1) & (final_mask==False)]
            selected_not_visual = NO_id[(visual_selection_all==0) & (final_mask==True)]

            ax = axall[0]
            text = []
            ax.plot(selected_NOs,residual_snr_array[final_mask],marker='o',linestyle='None'
                    ,color='forestgreen',markersize=5,label='Selected Sources')
            ax.plot(not_selected_all,residual_snr_array[not_selected_all-1],marker='o',linestyle='None',color='red',
                    markerfacecolor='none',markersize=5,label='Not Selected Sources')
            ax.plot(NO_id[visual_selection_all==1],residual_snr_array[visual_selection_all==1],marker='s',linestyle='None',color='blue',
                    markerfacecolor='none',markersize=10,label='Visual Selected Sources')
            
            for i, no in enumerate(visual_not_selected, start=1):
                x = NO_id[no - 1]
                y = residual_snr_array[no - 1]
                text.append(ax.text(
                    x+0.05, y+0.1,
                    str(i),
                    color='green',
                    fontsize=12
                ))

            for i, no in enumerate(selected_not_visual, start=1):
                x = NO_id[no - 1]
                y = residual_snr_array[no - 1]
                text.append(ax.text(
                    x+0.05, y+0.1,
                    str(i),
                    color='skyblue',
                    fontsize=15
                ))

            # adjust_text(text, ax=ax, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))

            ax.axhline(y=residual_snr_threshold, color='gray', linestyle='--',linewidth=1)
            ax.legend(fontsize=20)
            ax.set_yscale('log')
            ax.set_xlabel('Source ID',fontsize=20)
            ax.set_ylabel('Residual Peak / Residual Sigma',fontsize=20)
            ax.minorticks_on()
            ax.tick_params(axis='both', which='major', length=8, width=1.5,direction='in',top=True,right=True)
            ax.tick_params(axis='both', which='minor', length=4, width=1.5,direction='in',top=True,right=True)

            ax = axall[1]
            text = []
            ax.plot(scale_ratio_gaussian[final_mask],residual_snr_array[final_mask],marker='o',linestyle='None'
                    ,color='forestgreen',markersize=5,label='Selected Sources')
            ax.plot(scale_ratio_gaussian[not_selected_all-1],residual_snr_array[not_selected_all-1],marker='o'
                    ,linestyle='None',color='red',markerfacecolor='none',markersize=5,label='Not Selected Sources')
            ax.plot(scale_ratio_gaussian[visual_selection_all==1],residual_snr_array[visual_selection_all==1],marker='s'
                    ,linestyle='None',color='blue',markerfacecolor='none',markersize=10,label='Visual Selected Sources')
            
            for i, no in enumerate(visual_not_selected, start=1):
                x = scale_ratio_gaussian[no - 1]
                y = residual_snr_array[no - 1]
                text.append(ax.text(
                    x+0.05, y+0.1,
                    str(i),
                    color='green',
                    fontsize=12
                ))

            for i, no in enumerate(selected_not_visual, start=1):
                x = scale_ratio_gaussian[no - 1]
                y = residual_snr_array[no - 1]
                text.append(ax.text(
                    x+0.05, y+0.1,
                    str(i),
                    color='skyblue',
                    fontsize=15
                ))

            # adjust_text(text, ax=ax, arrowprops=dict(arrowstyle='->', color='black', lw=0.5))

            ax.axhline(y=residual_snr_threshold, color='gray', linestyle='--',linewidth=1)
            ax.axvline(x=compact_threshold, color='gray', linestyle='-',linewidth=1)
            ax.legend(fontsize=20)
            ax.set_yscale('log')
            # ax.set_xscale('log')
            ax.set_xlabel('Gaussian Scale / Beam Scale',fontsize=20)
            ax.set_ylabel('Residual Peak / Residual Sigma',fontsize=20)
            ax.minorticks_on()
            ax.tick_params(axis='both', which='major', length=8, width=1.5,direction='in',top=True,right=True)
            ax.tick_params(axis='both', which='minor', length=4, width=1.5,direction='in',top=True,right=True)

        if show_sources:
            # 3.2 绘制图像和标注
            # print("正在生成各区域的源标注图像...")
            THETA01_all = self.getsf_THETA01[self.center_key]
            
            wcs_cut_main = wcs_cut_all.celestial
            img_cut_main = img_cut_all
            img_cut_main[np.isnan(img_cut_main)] = 0 # 绘图时简单的 NaN 处理

            PIXEL_SCALE = self.PIXEL_SCALE
            fontsize = 20

            fig_plot, ax = plt.subplots(figsize=(20, 12), subplot_kw={'projection': wcs_cut_main})
            plt.rcParams.update({'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize, 'axes.labelsize': fontsize})
            
            T_img = self.Brightness_Temperature(img_cut_main, self.Freq)

            norm1 = ImageNormalize(stretch=LogStretch(), vmin=max(np.percentile(T_img, 10), 1e-3), vmax=T_img.max())
            imshow1 = ax.imshow(T_img, norm=norm1, origin='lower', cmap=cmap)

            text = []
            num_sourceVS = 0
            num_sourceVnS = 0
            num_sourceSnV = 0
            NO_selected_main = self.getsf_compact_sources_id[self.center_key]
            for i, ra in enumerate(ra_cen_all):
                # 注意：这里为了绘图准确，建议使用 WCS 转 Pix，而不是直接用 catalogue 的 XCO_P
                # 因为 catalogue 的 XCO_P 坐标系定义可能与 header 略有差异 (1-based vs 0-based)
                # 但为了兼容你原代码习惯，这里暂时保留原样或微调
                # 推荐方式:
                dec = dec_cen_all[i]   
                x_cen, y_cen = wcs_cut_main.celestial.all_world2pix(ra, dec, 0)
                
                majFWHM = AFWHM01_all[i] / PIXEL_SCALE.value
                minFWHM = BFWHM01_all[i] / PIXEL_SCALE.value
                theta = THETA01_all[i]
                current_NO = NO_id[i]

                if current_NO in NO_selected_main and current_NO in NO_id[visual_selection_all==1]:
                    # 选中的源：红色粗圈 + 编号
                    ellipse_this = Ellipse(xy=(x_cen, y_cen), width=majFWHM, height=minFWHM,
                                            angle=90 + theta, edgecolor='red', facecolor='none', linewidth=1)
                    ax.add_patch(ellipse_this)
                    ax.plot(x_cen, y_cen, marker='+', color='red', markersize=4, linestyle='None', markeredgewidth=1)
                    text.append(ax.text(x_cen + 5, y_cen + 5, s=f'{num_sourceVS + 1}', color='red', fontsize=12, weight='bold'))
                    num_sourceVS += 1
                elif current_NO not in NO_id[visual_selection_all==1] and current_NO in NO_selected_main:
                    # 选中的源 但是 视觉不佳：蓝色方圈 + 编号
                    ellipse_this = Ellipse(xy=(x_cen, y_cen), width=majFWHM, height=minFWHM,
                                            angle=90 + theta, edgecolor='blue', facecolor='none', linewidth=1)
                    ax.add_patch(ellipse_this)
                    ax.plot(x_cen, y_cen, marker='s', color='blue', markersize=6, linestyle='None', markeredgewidth=1,markerfacecolor='none')
                    text.append(ax.text(x_cen + 5, y_cen + 5, s=f'{num_sourceSnV + 1}', color='blue', fontsize=12, weight='bold'))
                    num_sourceSnV += 1
                elif current_NO not in NO_selected_main and current_NO in NO_id[visual_selection_all==1]:
                    # 视觉选中的源但未通过筛选：绿色方圈 + 编号
                    ellipse_this = Ellipse(xy=(x_cen, y_cen), width=majFWHM, height=minFWHM,
                                            angle=90 + theta, edgecolor='blue', facecolor='none', linewidth=1, alpha=0.5)
                    ax.add_patch(ellipse_this)
                    ax.plot(x_cen, y_cen, marker='^', color='green', markersize=8, linestyle='None', markeredgewidth=1,markerfacecolor='none')
                    text.append(ax.text(x_cen + 5, y_cen + 5, s=f'{num_sourceVnS + 1}', color='green', fontsize=12, weight='bold', alpha=0.5))
                    num_sourceVnS += 1
                else:
                    # 被剔除的源：青色细圈 (可选: 注释掉以保持图像干净)
                    ellipse_this = Ellipse(xy=(x_cen, y_cen), width=majFWHM, height=minFWHM,
                                            angle=90 + theta, edgecolor='black', facecolor='none', linewidth=0.5, alpha=0.5)
                    ax.add_patch(ellipse_this)
                    # ax.plot(x_cen, y_cen, marker='+', color='cyan', markersize=5, linestyle='None', markeredgewidth=0.5, alpha=0.5)
                    # print('not_selected?', current_NO, x_cen, y_cen, residual_snr_array[int(current_NO) - 1])

            if text:
                adjust_text(text, ax=ax, arrowprops=dict(arrowstyle='->', color='grey', lw=0.5))
            
            legend_handles = [
                Line2D(
                    [0], [0],
                    marker='s',
                    color='blue',
                    linestyle='None',
                    markersize=6,
                    markerfacecolor='none',
                    markeredgewidth=1,
                    label='Selected but not visually selected'
                ),
                Line2D(
                    [0], [0],
                    marker='^',
                    color='green',
                    linestyle='None',
                    markersize=8,
                    markerfacecolor='none',
                    markeredgewidth=1,
                    alpha=0.5,
                    label='Visually selected but not passed'
                )
            ]
            ax.legend(
                handles=legend_handles,
                fontsize=12
            )
            ax.set_xlabel('R.A.')
            ax.set_ylabel('Dec.')
            ax.set_title(f"Selected Compact Sources (Key: {self.center_key})\nSize <= {compact_threshold}xBeam & ResSNR >= {residual_snr_threshold}", fontsize=fontsize, pad=20)
            
            # Colorbar
            cb1 = plt.colorbar(imshow1, pad=0.01, aspect=25, fraction=0.08)
            cb1.ax.tick_params(labelsize=fontsize)
            cb1.set_label('Brightness Temperature (K)', fontsize=fontsize)

            plt.show()
        if show_selection and show_sources:
            return fig_selection, fig_plot
        if show_selection and (not show_sources):
            return fig_selection
        if (not show_selection) and show_sources:
            return fig_plot
        else:
            return None
    
    # 从外围region挑选leaves，也通过一样的标砖筛选
    def get_dendrogram_compact_sources(self, cmap='viridis', **kwargs):
        # --- 0. 参数解析 ---
        # show_plots = kwargs.get('plot', False)
        show_selection = kwargs.get('show_selection', True)
        show_sources = kwargs.get('show_sources', True)
        
        # 核心筛选参数
        residual_snr_threshold = kwargs.get('residual_snr_threshold', 5.0)
        kernel_factor = kwargs.get('kernel_factor', 2.0) # 控制平滑核的大小，一般比 Beam 大一点
        sigma_type = kwargs.get('sigma_type', 'MAD') # 控制残差噪声计算方式，'clip','MAD' 目前是这三种

        # self.leaf_out_all_ra, self.leaf_out_all_dec
        leaf_not_allchan_ra = np.concatenate([self.leaf_out_matched_ra_array,self.leaf_out_unmatched_ra])
        leaf_not_allchan_dec = np.concatenate([self.leaf_out_matched_dec_array,self.leaf_out_unmatched_dec])

        # output_dir = os.path.join('cutout_ps_dir', '18517_Band6_TM1+TM2','others')
        # os.makedirs(output_dir, exist_ok=True)
        # size = int(60 * 3090 / self.distance)
        # cutout_size = (size, size)  # 切片的大小 (height, width)

        if self.distance <= 3090:
            size = int(60 * 3090 / self.distance)
        else:
            size = 60
        cutout_size = kwargs.get('cutout_shape', (size, size))

        all = np.arange(1,len(leaf_not_allchan_ra)+1,1)

        bmaj_pix = self.BEAM_MAJOR.value / self.PIXEL_SCALE.value
        bmin_pix = self.BEAM_MINOR.value / self.PIXEL_SCALE.value
        bpa_deg = self.psfPA

        sigma_maj_pix = bmaj_pix / (2*np.sqrt(2*np.log(2)))
        sigma_min_pix = bmin_pix / (2*np.sqrt(2*np.log(2)))
        theta_rad = np.deg2rad(90.0 + bpa_deg)

        beam = Gaussian2DKernel(
                    x_stddev= kernel_factor * sigma_maj_pix, 
                    y_stddev= kernel_factor * sigma_min_pix, 
                    theta=theta_rad
                )

        residual_peak2sigma_list = np.zeros_like(leaf_not_allchan_ra,dtype=float)

        for i,j in enumerate(leaf_not_allchan_ra):
            # if i in current_point_source_list:
            xcen , ycen = self.wcs.celestial.all_world2pix(j,leaf_not_allchan_dec[i],0)
            position = (xcen, ycen)  # 中心位置 (x, y)
            # print(position)
            cutout = Cutout2D(self.img, position, cutout_size, mode='partial',wcs=self.wcs.celestial)
            judge = not np.isnan(cutout.data).any()
            if  judge == True:# & (i in manual_selection): 
                T_img = self.Brightness_Temperature(cutout.data,self.Freq)
                # convolved_image = convolve_fft(T_img, beam, preserve_nan=False)
                convolved_image = convolve_fft(T_img, beam, boundary='fill', fill_value=np.nan, preserve_nan=True)
                residual_img = T_img - convolved_image
                if sigma_type == 'clip':
                    _, _, sigma_residual = sigma_clipped_stats(residual_img, sigma=3.0, maxiters=None)
                elif sigma_type == 'MAD':
                    sigma_residual = mad_std(residual_img, ignore_nan=True)
                else:
                    sigma_type == 'MAD'
                    print('We have only three sigma_type: clip, mask, MAD! Use MAD by default.')
                    sigma_residual = mad_std(residual_img, ignore_nan=True)
                # sigma_residual = sigma_clipped_stats(residual_img,sigma=3.0,maxiters=None)[-1]
                residual_peak2sigma_list[i] = residual_img[cutout_size[0]//2,cutout_size[1]//2] / sigma_residual

        # 判据 2: 残差信噪比足够高
        mask_snr = residual_peak2sigma_list >= residual_snr_threshold
        final_mask = mask_snr
        # 筛选出最终的 NO
        selected_NOs = all[final_mask]
        self.selected_leaves = selected_NOs
        self.leaf_not_allchan_ra = leaf_not_allchan_ra
        self.leaf_not_allchan_dec = leaf_not_allchan_dec

        # print(f"区域 {key}: 总源数 {len(NO_id)}, 尺寸筛选后 {np.sum(mask_size)}, "
        #       f"SNR筛选后 {np.sum(mask_snr)}, 最终选中 {len(selected_NOs)}")

        if show_selection:
            not_selected = all[~np.isin(all,selected_NOs)] - 1
            # not_selected = all-1

            fig_selection,ax = plt.subplots(figsize=(12,8))
            ax.plot(selected_NOs,residual_peak2sigma_list[selected_NOs-1],marker='o',linestyle='None'
                        ,color='forestgreen',markersize=5,label='Selected Sources')
            ax.plot(not_selected+1,residual_peak2sigma_list[not_selected],marker='o'
                        ,linestyle='None',color='red',markerfacecolor='none',markersize=5,label='Not Selected Sources')
            ax.axhline(y=residual_snr_threshold, color='gray', linestyle='--',linewidth=1)
            ax.legend(fontsize=20)
            ax.set_yscale('log')
            ax.set_xlabel('Source ID',fontsize=20)
            ax.set_ylabel('Residual Peak / Residual Sigma',fontsize=20)
            ax.minorticks_on()
            ax.tick_params(axis='both', which='major', length=8, width=1.5,direction='in',top=True,right=True)
            ax.tick_params(axis='both', which='minor', length=4, width=1.5,direction='in',top=True,right=True)
            plt.show()
        
        if show_sources:
            from adjustText import adjust_text
            cmap = cmap #'nipy_spectral' 
            fontsize = 20
            show_ellipse = True
            vmax = 1 # in K

            fig_plot = plt.figure(figsize=(30,8))       # X、Y轴标签字体大小
            ax =  fig_plot.add_subplot(1,3,1,projection=self.wcs.celestial)
            # ax1 = fig.add_subplot(1,3,2,projection=aaa_18517_allchan.wcs.celestial)
            # ax2 = fig.add_subplot(1,3,3,projection=aaa_18517_allchan.wcs.celestial)

            plt.rcParams['xtick.labelsize'] = fontsize
            plt.rcParams['ytick.labelsize'] = fontsize
            # X、Y轴刻度标签字体大小
            plt.rcParams['axes.labelsize'] = fontsize
            # mean,median,std = sigma_clipped_stats(T_img,sigma=3.0)
            #norm1 = ImageNormalize(stretch=LogStretch(),vmin=std,vmax=vmax)
            T_img_18517_normal = self.Brightness_Temperature(self.img,self.Freq)
            imshow1 = ax.imshow(T_img_18517_normal,vmin=0,vmax=vmax,origin='lower',cmap=cmap)

            # I18517_normal_x_cen,I18517_normal_y_cen = aaa_18517.wcs.celestial.all_world2pix(I18517_normal_ra_cen,I18517_normal_dec_cen,0)
            # ax.plot(I18517_normal_x_cen,I18517_normal_y_cen,marker='+',color='blue',markersize=5,linestyle='None',markerfacecolor='none',markeredgewidth=0.5,label='getsf sources')
            # ax.plot(leaf_x_out_18517,leaf_y_out_18517,marker='o',color='blue',markersize=7,linestyle='None',markerfacecolor='none',markeredgewidth=1,label='dendrogram leaves')
            # ax.plot(sperate_source_x_18517,sperate_source_y_18517,marker='o',color='blue',markersize=7,linestyle='None',markerfacecolor='none',markeredgewidth=1,label='dendrogram leaves')
            text=[]
            leaf_out_nonallchan_x,leaf_out_nonallchan_y = self.wcs.celestial.all_world2pix(leaf_not_allchan_ra[final_mask],leaf_not_allchan_dec[final_mask],0)
            ax.plot(leaf_out_nonallchan_x,leaf_out_nonallchan_y,marker='o',color='red',markersize=7,linestyle='None',markerfacecolor='none',markeredgewidth=1,label='dendrogram selected leaves')
            for i ,num_source in enumerate(range(len(leaf_out_nonallchan_x))):
                if i % 2 == 0:
                    text.append(ax.text(leaf_out_nonallchan_x[num_source]+15,leaf_out_nonallchan_y[num_source]+10,s=f'{num_source+1}',color='black',fontsize=8))
                else:
                    text.append(ax.text(leaf_out_nonallchan_x[num_source]-15,leaf_out_nonallchan_y[num_source]-10,s=f'{num_source+1}',color='black',fontsize=8))

            adjust_text(text, ax=ax, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

            leaf_out_nonallchan_x_rest,leaf_out_nonallchan_y_rest = self.wcs.celestial.all_world2pix(leaf_not_allchan_ra[~final_mask],leaf_not_allchan_dec[~final_mask],0)
            ax.plot(leaf_out_nonallchan_x_rest,leaf_out_nonallchan_y_rest,marker='s',color='grey',markersize=7,linestyle='None',markerfacecolor='none',markeredgewidth=1,label='dendrogram not selected leaves')
            ax.legend(loc='upper left',prop = {'size':15})
            effect = withStroke(linewidth=2, foreground='grey')
            wcsaxes.add_beam(ax=ax,header=self.head,pad=2,path_effects=[effect])
            ax.set_xlabel('R.A.')
            ax.set_ylabel('Dec.')
            ax.tick_params(axis='both', length=8, width=2,direction='in',color='black',labelcolor='black')
            ax.minorticks_on()
            ax.set_aspect('equal')

            cb1 = plt.colorbar(imshow1, pad=0, aspect=20,fraction=0.1)
            # cb1.set_ticks(LogLocator(base=10.0))  # 使 colorbar 刻度以 log 方式显示
            # cb1.ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=[0.2,0.3,0.4, 0.5,0.6, 0.7,0.8,0.9], numticks=10))
            cb1.ax.tick_params(labelsize=fontsize, length=8, width=2)  #设置色标刻度字体大小。
            cb1.ax.tick_params(direction='in')
            cb1.ax.xaxis.set_ticks_position('bottom')  # 让刻度移动到下方
            cb1.ax.tick_params(axis="both", which="major", width=2, length=8, direction='in')
            cb1.ax.tick_params(axis="both", which="minor", width=1.5, length=4, direction='in')
            font = {'color'  : 'black','weight' : 'normal','size'   : fontsize}
            cb1.set_label('Brightness Temperature (K)',fontdict=font) #设置colorbar的标签字体及其大小
            # plt.subplots_adjust(wspace=0.0)
            plt.show()

        if show_selection and show_sources:
            return fig_selection, fig_plot
        if show_selection and (not show_sources):
            return fig_selection
        if (not show_selection) and show_sources:
            return fig_plot
        else:
            return None
        
    def get_dendrogram_compact_sources_2(self, cmap='viridis', **kwargs):
        # --- 0. 参数解析 ---
        # show_plots = kwargs.get('plot', False)
        show_selection = kwargs.get('show_selection', True)
        show_sources = kwargs.get('show_sources', True)
        
        # 核心筛选参数
        residual_snr_threshold = kwargs.get('residual_snr_threshold', 5.0)
        kernel_factor = kwargs.get('kernel_factor', 2.0) # 控制平滑核的大小，一般比 Beam 大一点
        sigma_type = kwargs.get('sigma_type', 'MAD') # 控制残差噪声计算方式，'clip','MAD' 目前是这三种

        # self.leaf_out_all_ra, self.leaf_out_all_dec
        leaf_not_allchan_ra = self.leaf_out_all_ra_array
        leaf_not_allchan_dec = self.leaf_out_all_dec_array
        visual_selected_leaves = self.visiual_selection_leaves



        # output_dir = os.path.join('cutout_ps_dir', '18517_Band6_TM1+TM2','others')
        # os.makedirs(output_dir, exist_ok=True)
        # size = int(60 * 3090 / self.distance)
        # cutout_size = (size, size)  # 切片的大小 (height, width)

        if self.distance <= 3090:
            size = int(60 * 3090 / self.distance)
        else:
            size = 60
        cutout_size = kwargs.get('cutout_shape', (size, size))

        all = np.arange(1,len(leaf_not_allchan_ra)+1,1)

        bmaj_pix = self.BEAM_MAJOR.value / self.PIXEL_SCALE.value
        bmin_pix = self.BEAM_MINOR.value / self.PIXEL_SCALE.value
        bpa_deg = self.psfPA

        sigma_maj_pix = bmaj_pix / (2*np.sqrt(2*np.log(2)))
        sigma_min_pix = bmin_pix / (2*np.sqrt(2*np.log(2)))
        theta_rad = np.deg2rad(90.0 + bpa_deg)

        beam = Gaussian2DKernel(
                    x_stddev= kernel_factor * sigma_maj_pix, 
                    y_stddev= kernel_factor * sigma_min_pix, 
                    theta=theta_rad
                )

        residual_peak2sigma_list = np.zeros_like(leaf_not_allchan_ra,dtype=float)

        for i,j in enumerate(leaf_not_allchan_ra):
            # if i in current_point_source_list:
            xcen , ycen = self.wcs.celestial.all_world2pix(j,leaf_not_allchan_dec[i],0)
            position = (xcen, ycen)  # 中心位置 (x, y)
            # print(position)
            cutout = Cutout2D(self.img, position, cutout_size, mode='partial',wcs=self.wcs.celestial)
            judge = not np.isnan(cutout.data).any()
            if  judge == True:# & (i in manual_selection): 
                T_img = self.Brightness_Temperature(cutout.data,self.Freq)
                # convolved_image = convolve_fft(T_img, beam, preserve_nan=False)
                convolved_image = convolve_fft(T_img, beam, boundary='fill', fill_value=np.nan, preserve_nan=True)
                residual_img = T_img - convolved_image
                if sigma_type == 'clip':
                    _, _, sigma_residual = sigma_clipped_stats(residual_img, sigma=3.0, maxiters=None)
                elif sigma_type == 'MAD':
                    sigma_residual = mad_std(residual_img, ignore_nan=True)
                else:
                    sigma_type == 'MAD'
                    print('We have only three sigma_type: clip, mask, MAD! Use MAD by default.')
                    sigma_residual = mad_std(residual_img, ignore_nan=True)
                # sigma_residual = sigma_clipped_stats(residual_img,sigma=3.0,maxiters=None)[-1]
                residual_peak2sigma_list[i] = residual_img[cutout_size[0]//2,cutout_size[1]//2] / sigma_residual

        # 判据 2: 残差信噪比足够高
        mask_snr = residual_peak2sigma_list >= residual_snr_threshold
        final_mask = mask_snr

        all_NO = all
        visual_selected = visual_selected_leaves == 1

        VS_MS  = all_NO[ visual_selected &  final_mask]   # visual & selected
        nVS_MS = all_NO[~visual_selected &  final_mask]   # selected but not visual
        VS_nMS = all_NO[ visual_selected & ~final_mask]   # visual but not selected
        nVS_nMS= all_NO[~visual_selected & ~final_mask]   # neither

        # 筛选出最终的 NO
        selected_NOs = all[final_mask]
        self.selected_leaves = selected_NOs
        self.leaf_not_allchan_ra = leaf_not_allchan_ra
        self.leaf_not_allchan_dec = leaf_not_allchan_dec

        # print(f"区域 {key}: 总源数 {len(NO_id)}, 尺寸筛选后 {np.sum(mask_size)}, "
        #       f"SNR筛选后 {np.sum(mask_snr)}, 最终选中 {len(selected_NOs)}")

        if show_selection:
            fig_selection, ax = plt.subplots(figsize=(12, 8))

            # --- Selected (mask) ---
            ax.plot(
                all_NO[final_mask],
                residual_peak2sigma_list[final_mask],
                marker='o', linestyle='None',
                color='forestgreen', markersize=5,
                label='Selected (Mask)'
            )

            # --- Not selected ---
            ax.plot(
                all_NO[~final_mask],
                residual_peak2sigma_list[~final_mask],
                marker='o', linestyle='None',
                color='red', markerfacecolor='none',
                markersize=5,
                label='Not Selected'
            )

            # --- Visual selected ---
            ax.plot(
                all_NO[visual_selected],
                residual_peak2sigma_list[visual_selected],
                marker='s', linestyle='None',
                color='blue', markerfacecolor='none',
                markersize=10,
                label='Visual Selected'
            )

            # --- 编号：VS but not MS ---
            for i, no in enumerate(VS_nMS, start=1):
                ax.text(
                    no + 0.05,
                    residual_peak2sigma_list[no - 1] + 0.1,
                    str(i),
                    color='green',
                    fontsize=12
                )

            # --- 编号：MS but not VS ---
            for i, no in enumerate(nVS_MS, start=1):
                ax.text(
                    no + 0.05,
                    residual_peak2sigma_list[no - 1] + 0.1,
                    str(i),
                    color='skyblue',
                    fontsize=12
                )

            ax.axhline(y=residual_snr_threshold, color='gray', linestyle='--', linewidth=1)

            ax.set_yscale('log')
            ax.set_xlabel('Leaf ID', fontsize=20)
            ax.set_ylabel('Residual Peak / Residual Sigma', fontsize=20)
            ax.legend(fontsize=15)

            ax.minorticks_on()
            ax.tick_params(axis='both', which='major', length=8, width=1.5,
                        direction='in', top=True, right=True)
            ax.tick_params(axis='both', which='minor', length=4, width=1.5,
                        direction='in', top=True, right=True)

            plt.show()
        
        if show_sources:
            from adjustText import adjust_text
            cmap = cmap #'nipy_spectral' 
            fontsize = 20
            show_ellipse = True
            vmax = 1 # in K

            fig_plot = plt.figure(figsize=(30,8))       # X、Y轴标签字体大小
            ax =  fig_plot.add_subplot(1,3,1,projection=self.wcs.celestial)
            # ax1 = fig.add_subplot(1,3,2,projection=aaa_18517_allchan.wcs.celestial)
            # ax2 = fig.add_subplot(1,3,3,projection=aaa_18517_allchan.wcs.celestial)

            plt.rcParams['xtick.labelsize'] = fontsize
            plt.rcParams['ytick.labelsize'] = fontsize
            # X、Y轴刻度标签字体大小
            plt.rcParams['axes.labelsize'] = fontsize
            # mean,median,std = sigma_clipped_stats(T_img,sigma=3.0)
            #norm1 = ImageNormalize(stretch=LogStretch(),vmin=std,vmax=vmax)
            T_img_18517_normal = self.Brightness_Temperature(self.img,self.Freq)
            imshow1 = ax.imshow(T_img_18517_normal,vmin=0,vmax=vmax,origin='lower',cmap=cmap)

            # I18517_normal_x_cen,I18517_normal_y_cen = aaa_18517.wcs.celestial.all_world2pix(I18517_normal_ra_cen,I18517_normal_dec_cen,0)
            # ax.plot(I18517_normal_x_cen,I18517_normal_y_cen,marker='+',color='blue',markersize=5,linestyle='None',markerfacecolor='none',markeredgewidth=0.5,label='getsf sources')
            # ax.plot(leaf_x_out_18517,leaf_y_out_18517,marker='o',color='blue',markersize=7,linestyle='None',markerfacecolor='none',markeredgewidth=1,label='dendrogram leaves')
            # ax.plot(sperate_source_x_18517,sperate_source_y_18517,marker='o',color='blue',markersize=7,linestyle='None',markerfacecolor='none',markeredgewidth=1,label='dendrogram leaves')
            text=[]
            # --- VS & MS ---
            x, y = self.wcs.celestial.all_world2pix(
                leaf_not_allchan_ra[VS_MS - 1],
                leaf_not_allchan_dec[VS_MS - 1], 0
            )
            ax.plot(
                x, y,
                marker='+', color='red',
                markersize=6, linestyle='None',
                label='Selected & Visual'
            )

            # --- MS but not VS ---
            x, y = self.wcs.celestial.all_world2pix(
                leaf_not_allchan_ra[nVS_MS - 1],
                leaf_not_allchan_dec[nVS_MS - 1], 0
            )
            ax.plot(
                x, y,
                marker='s', color='blue',
                markerfacecolor='none',
                markersize=7, linestyle='None',
                label='Selected but not visual'
            )

            # --- VS but not MS ---
            x, y = self.wcs.celestial.all_world2pix(
                leaf_not_allchan_ra[VS_nMS - 1],
                leaf_not_allchan_dec[VS_nMS - 1], 0
            )
            ax.plot(
                x, y,
                marker='^', color='green',
                markerfacecolor='none',
                markersize=8, linestyle='None',
                alpha=0.5,
                label='Visual but not selected'
            )

            # not VS and not MS :
            x, y = self.wcs.celestial.all_world2pix(
                leaf_not_allchan_ra[nVS_nMS - 1],
                leaf_not_allchan_dec[nVS_nMS - 1], 0
            )
            ax.plot(
                x, y,
                marker='o', color='grey',
                markerfacecolor='none',
                markersize=10, linestyle='None',
                alpha=0.5,
                label='Not Visual & Not Selected'
            )

                        
            # for i ,num_source in enumerate(range(len(leaf_out_nonallchan_x))):
            #     if i % 2 == 0:
            #         text.append(ax.text(leaf_out_nonallchan_x[num_source]+15,leaf_out_nonallchan_y[num_source]+10,s=f'{num_source+1}',color='black',fontsize=8))
            #     else:
            #         text.append(ax.text(leaf_out_nonallchan_x[num_source]-15,leaf_out_nonallchan_y[num_source]-10,s=f'{num_source+1}',color='black',fontsize=8))

            adjust_text(text, ax=ax, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

            # leaf_out_nonallchan_x_rest,leaf_out_nonallchan_y_rest = self.wcs.celestial.all_world2pix(leaf_not_allchan_ra[~final_mask],leaf_not_allchan_dec[~final_mask],0)
            # ax.plot(leaf_out_nonallchan_x_rest,leaf_out_nonallchan_y_rest,marker='s',color='grey',markersize=7,linestyle='None',markerfacecolor='none',markeredgewidth=1,label='dendrogram not selected leaves')
            # ax.legend(loc='upper left',prop = {'size':15})
            legend_handles = [
                Line2D([0], [0], marker='+', color='red', linestyle='None',
                    label='Selected & Visual'),
                Line2D([0], [0], marker='s', color='blue', linestyle='None',
                    markerfacecolor='none',
                    label='Selected but not visual'),
                Line2D([0], [0], marker='^', color='green', linestyle='None',
                    markerfacecolor='none', alpha=0.5,
                    label='Visual but not selected')
                ,Line2D([0], [0], marker='o', color='grey', linestyle='None',
                    markerfacecolor='none', alpha=0.5,
                    label='Not Visual & Not Selected')
            ]

            ax.legend(handles=legend_handles, fontsize=14)
            effect = withStroke(linewidth=2, foreground='grey')
            wcsaxes.add_beam(ax=ax,header=self.head,pad=2,path_effects=[effect])
            ax.set_xlabel('R.A.')
            ax.set_ylabel('Dec.')
            ax.tick_params(axis='both', length=8, width=2,direction='in',color='black',labelcolor='black')
            ax.minorticks_on()
            ax.set_aspect('equal')

            cb1 = plt.colorbar(imshow1, pad=0, aspect=20,fraction=0.1)
            # cb1.set_ticks(LogLocator(base=10.0))  # 使 colorbar 刻度以 log 方式显示
            # cb1.ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=[0.2,0.3,0.4, 0.5,0.6, 0.7,0.8,0.9], numticks=10))
            cb1.ax.tick_params(labelsize=fontsize, length=8, width=2)  #设置色标刻度字体大小。
            cb1.ax.tick_params(direction='in')
            cb1.ax.xaxis.set_ticks_position('bottom')  # 让刻度移动到下方
            cb1.ax.tick_params(axis="both", which="major", width=2, length=8, direction='in')
            cb1.ax.tick_params(axis="both", which="minor", width=1.5, length=4, direction='in')
            font = {'color'  : 'black','weight' : 'normal','size'   : fontsize}
            cb1.set_label('Brightness Temperature (K)',fontdict=font) #设置colorbar的标签字体及其大小
            # plt.subplots_adjust(wspace=0.0)
            plt.show()

        if show_selection and show_sources:
            return fig_selection, fig_plot
        if show_selection and (not show_sources):
            return fig_selection
        if (not show_selection) and show_sources:
            return fig_plot
        else:
            return None
    
    # 把 filtered compact source 存入 conn的对应表格
    def save_compact_sources(self,conn,table_name):

        ALLOWED_TABLES = {
            'robust+0.5_filtered_sources',
            'robust-0.5_filtered_sources',
            'allchan_filtered_sources'
        }

        if table_name not in ALLOWED_TABLES:
            raise ValueError(f'Illegal table name: {table_name}')

        region_keys = list(self.getsf_cat.keys())
        for key in region_keys:
            ra_list = self.getsf_ra_cen[key]
            dec_list = self.getsf_dec_cen[key]
            selected_NOs = self.getsf_compact_sources_id[key]
            ra_selected = ra_list[selected_NOs - 1]
            dec_selected = dec_list[selected_NOs - 1]
            for ra, dec in zip(ra_selected, dec_selected):
                conn.execute(
                    f'INSERT INTO "{table_name}" (ra, dec, method) VALUES (?, ?, ?)',
                    (ra, dec, 'getsf')
                )

        ra_leaves_list = self.leaf_not_allchan_ra[self.selected_leaves-1]
        dec_leaves_list = self.leaf_not_allchan_dec[self.selected_leaves-1]
        for ra, dec in zip(ra_leaves_list, dec_leaves_list):
            conn.execute(
                f'INSERT INTO "{table_name}" (ra, dec, method) VALUES (?, ?, ?)',
                (ra, dec, 'leaves')
            )

        conn.commit()
        return None
    
    def save_compact_sources_2(self,conn,table_name):

        ALLOWED_TABLES = {
            'robust+0.5_filtered_sources',
            'robust-0.5_filtered_sources',
            'allchan_filtered_sources'
        }

        if table_name not in ALLOWED_TABLES:
            raise ValueError(f'Illegal table name: {table_name}')

        ra_list = self.getsf_ra_cen[self.center_key]
        dec_list = self.getsf_dec_cen[self.center_key]
        selected_NOs = self.getsf_compact_sources_id[self.center_key]
        ra_selected = ra_list[selected_NOs - 1]
        dec_selected = dec_list[selected_NOs - 1]
        for ra, dec in zip(ra_selected, dec_selected):
            conn.execute(
                f'INSERT INTO "{table_name}" (ra, dec, method) VALUES (?, ?, ?)',
                (ra, dec, 'getsf')
            )

        ra_leaves_list = self.leaf_not_allchan_ra[self.selected_leaves-1]
        dec_leaves_list = self.leaf_not_allchan_dec[self.selected_leaves-1]
        for ra, dec in zip(ra_leaves_list, dec_leaves_list):
            conn.execute(
                f'INSERT INTO "{table_name}" (ra, dec, method) VALUES (?, ?, ?)',
                (ra, dec, 'leaves')
            )

        conn.commit()
        return None
    
    def delete_compact_sources(self, conn, table_name):
        """
        清空指定表格中的所有数据 (相当于 Reset)
        """
        ALLOWED_TABLES = {
            'robust+0.5_filtered_sources',
            'robust-0.5_filtered_sources',
            'allchan_filtered_sources'
        }

        # 1. 安全检查：确保只操作允许的表
        if table_name not in ALLOWED_TABLES:
            raise ValueError(f'Illegal table name: {table_name}')

        # 2. 执行删除操作
        # 注意：这里使用了 f-string 和双引号 "{table_name}"
        # 因为你的表名里包含 '+' 和 '-' 等特殊字符，不加引号 SQL 会报错
        conn.execute(f'DELETE FROM "{table_name}"')
        
        # 3. 提交更改
        conn.commit()
        
        return None

    

def crossmatch_catalogs(sources1: SkyCoord, sources2: SkyCoord, sources3: SkyCoord,
                        radius: u.Quantity, names: tuple = ('cat1', 'cat2', 'cat3')) -> dict:
    """
    对三个天体坐标目录进行交叉匹配，并根据匹配关系对源进行分类。

    此函数使用“中心辐射”法，将目录2和目录3都与目录1进行匹配，
    然后通过集合运算来确定每个源的最终归属。

    Args:
        sources1 (SkyCoord): 第一个天体坐标目录。
        sources2 (SkyCoord): 第二个天体坐标目录。
        sources3 (SkyCoord): 第三个天体坐标目录。
        radius (u.Quantity): 匹配半径，一个 astropy.units 对象 (例如 0.1 * u.arcsec)。
        names (tuple, optional): 三个目录的名称，用于生成字典的键。
                                 默认为 ('cat1', 'cat2', 'cat3')。

    Returns:
        dict: 一个字典，包含所有分类源的坐标信息。
    """
    print("开始交叉匹配...")
    # --- 步骤 1: 将目录2和目录3都匹配到“主目录”1 ---
    idx1_for_2, d2d_21, _ = sources2.match_to_catalog_sky(sources1)
    idx1_for_3, d2d_31, _ = sources3.match_to_catalog_sky(sources1)

    # --- 步骤 2: 根据匹配半径筛选出有效的匹配 ---
    mask_21_matched = d2d_21 < radius
    mask_31_matched = d2d_31 < radius

    # --- 步骤 3: 构建基于“主目录1”索引的集合 ---
    set1_all_indices = set(range(len(sources1)))
    set1_indices_in_2 = set(idx1_for_2[mask_21_matched])
    set1_indices_in_3 = set(idx1_for_3[mask_31_matched])

    print(f"目录1中有 {len(set1_indices_in_2)} 个源与目录2匹配。")
    print(f"目录1中有 {len(set1_indices_in_3)} 个源与目录3匹配。")

    # --- 步骤 4: 使用集合运算来分类源 ---
    idx_in_all = set1_indices_in_2 & set1_indices_in_3
    print(f"找到 {len(idx_in_all)} 个在所有三个目录中都存在的源。")
    idx_in_1_and_2_only = set1_indices_in_2 - set1_indices_in_3
    idx_in_1_and_3_only = set1_indices_in_3 - set1_indices_in_2
    idx_unique_to_1 = set1_all_indices - set1_indices_in_2 - set1_indices_in_3

    # --- 步骤 5: 处理只存在于2和3，但不存在于1中的情况 ---
    
    # 找到所有在目录2中但未匹配到目录1的源的索引
    idx2_unmatched_to_1 = np.where(~mask_21_matched)[0]
    sources2_unmatched = sources2[idx2_unmatched_to_1]

    # 找到所有在目录3中但未匹配到目录1的源的索引
    idx3_unmatched_to_1 = np.where(~mask_31_matched)[0]
    sources3_unmatched = sources3[idx3_unmatched_to_1]
    
    idx_in_2_and_3_only_s2, idx_in_2_and_3_only_s3 = np.array([], dtype=int), np.array([], dtype=int)
    
    if len(sources2_unmatched) > 0 and len(sources3_unmatched) > 0:
        idx3_for_2_unmatched, d2d_unmatched, _ = sources2_unmatched.match_to_catalog_sky(sources3_unmatched)
        mask_unmatched_matched = d2d_unmatched < radius
        
        # ================================================================
        # ===================== 这里是修正的部分 ==========================
        # ================================================================
        
        # 从“未匹配到1的源”的索引数组中，筛选出那些在2和3之间匹配上的
        # 直接使用布尔掩码来索引NumPy数组，这是最简洁和正确的方式
        indices_in_s2_unmatched = np.where(mask_unmatched_matched)[0]
        indices_in_s3_unmatched = idx3_for_2_unmatched[mask_unmatched_matched]

        # 将这些局部索引转换回原始目录中的全局索引
        idx_in_2_and_3_only_s2 = idx2_unmatched_to_1[indices_in_s2_unmatched]
        idx_in_2_and_3_only_s3 = idx3_unmatched_to_1[indices_in_s3_unmatched]
    
    # 5b. 找到真正独特的源
    # 使用集合运算来高效地找到差集
    set_2_and_3_only = set(idx_in_2_and_3_only_s2)
    set_3_and_2_only = set(idx_in_2_and_3_only_s3)

    idx_unique_to_2 = set(idx2_unmatched_to_1) - set_2_and_3_only
    idx_unique_to_3 = set(idx3_unmatched_to_1) - set_3_and_2_only

    # --- 步骤 6: 创建最终的输出数据结构 ---
    def get_coords_from_indices(source_obj, indices):
        # 确保传入的是列表或numpy数组，而不是集合
        indices = list(indices)
        if not indices:
            return {'ra': np.array([]), 'dec': np.array([])}
        subset = source_obj[indices]
        return {'ra': subset.ra.degree, 'dec': subset.dec.degree}

    results = {
        'in_all_three': get_coords_from_indices(sources1, idx_in_all),
        f'in_{names[0]}_and_{names[1]}_only': get_coords_from_indices(sources1, idx_in_1_and_2_only),
        f'in_{names[0]}_and_{names[2]}_only': get_coords_from_indices(sources1, idx_in_1_and_3_only),
        f'in_{names[1]}_and_{names[2]}_only': get_coords_from_indices(sources2, idx_in_2_and_3_only_s2),
        f'unique_to_{names[0]}': get_coords_from_indices(sources1, idx_unique_to_1),
        f'unique_to_{names[1]}': get_coords_from_indices(sources2, idx_unique_to_2),
        f'unique_to_{names[2]}': get_coords_from_indices(sources3, idx_unique_to_3),
    }

    print("\n匹配完成！结果摘要：")
    for key, value in results.items():
        print(f"- {key}: 找到 {len(value['ra'])} 个源。")
        
    return results


def crossmatch_catalogs_v2(sources1: SkyCoord, sources2: SkyCoord, sources3: SkyCoord = None,
                           radius: u.Quantity = 1.0*u.arcsec, 
                           names: tuple = ('cat1', 'cat2', 'cat3')) -> dict:
    """
    智能交叉匹配：支持 2 个或 3 个天体目录。
    
    如果 sources3 为 None，则只匹配 cat1 和 cat2。
    """
    print("开始交叉匹配...")
    
    # 辅助函数：从索引提取坐标
    def get_coords_from_indices(source_obj, indices):
        indices = list(indices)
        if not indices:
            return {'ra': np.array([]), 'dec': np.array([])}
        subset = source_obj[indices]
        return {'ra': subset.ra.degree, 'dec': subset.dec.degree}

    # === 情况 A: 只有两个目录 ===
    if sources3 is None:
        print(f"检测到双目录模式：匹配 {names[0]} 和 {names[1]}...")
        
        # 1. match 2 to 1
        idx1_for_2, d2d_21, _ = sources2.match_to_catalog_sky(sources1)
        mask_21_matched = d2d_21 < radius
        
        # 2. 集合运算
        set1_all = set(range(len(sources1)))
        set1_matched = set(idx1_for_2[mask_21_matched])
        
        idx_in_both = set1_matched
        idx_unique_to_1 = set1_all - set1_matched
        
        # 3. 找 unique to 2 (即 2 中没匹配上 1 的)
        idx2_unmatched = np.where(~mask_21_matched)[0]
        
        results = {
            f'in_{names[0]}_and_{names[1]}': get_coords_from_indices(sources1, idx_in_both),
            f'unique_to_{names[0]}': get_coords_from_indices(sources1, idx_unique_to_1),
            f'unique_to_{names[1]}': get_coords_from_indices(sources2, idx2_unmatched),
        }
        
        print(f"匹配完成！共找到 {len(idx_in_both)} 个共有源。")
        return results

    # === 情况 B: 三个目录 (原有逻辑) ===
    print(f"检测到三目录模式：匹配 {names[0]}, {names[1]}, {names[2]}...")
    
    # --- 步骤 1: 将目录2和目录3都匹配到“主目录”1 ---
    idx1_for_2, d2d_21, _ = sources2.match_to_catalog_sky(sources1)
    idx1_for_3, d2d_31, _ = sources3.match_to_catalog_sky(sources1)

    # --- 步骤 2: 筛选有效匹配 ---
    mask_21_matched = d2d_21 < radius
    mask_31_matched = d2d_31 < radius

    # --- 步骤 3: 构建集合 ---
    set1_all_indices = set(range(len(sources1)))
    set1_indices_in_2 = set(idx1_for_2[mask_21_matched])
    set1_indices_in_3 = set(idx1_for_3[mask_31_matched])

    # --- 步骤 4: 分类基于主目录的源 ---
    idx_in_all = set1_indices_in_2 & set1_indices_in_3
    idx_in_1_and_2_only = set1_indices_in_2 - set1_indices_in_3
    idx_in_1_and_3_only = set1_indices_in_3 - set1_indices_in_2
    idx_unique_to_1 = set1_all_indices - set1_indices_in_2 - set1_indices_in_3

    # --- 步骤 5: 处理只存在于2和3的情况 ---
    idx2_unmatched_to_1 = np.where(~mask_21_matched)[0]
    sources2_unmatched = sources2[idx2_unmatched_to_1]

    idx3_unmatched_to_1 = np.where(~mask_31_matched)[0]
    sources3_unmatched = sources3[idx3_unmatched_to_1]
    
    idx_in_2_and_3_only_s2 = []
    
    if len(sources2_unmatched) > 0 and len(sources3_unmatched) > 0:
        idx3_for_2_unmatched, d2d_unmatched, _ = sources2_unmatched.match_to_catalog_sky(sources3_unmatched)
        mask_unmatched_matched = d2d_unmatched < radius
        
        # 获取匹配上的局部索引 -> 全局索引
        indices_in_s2_unmatched = np.where(mask_unmatched_matched)[0]
        idx_in_2_and_3_only_s2 = idx2_unmatched_to_1[indices_in_s2_unmatched]
    
    set_2_and_3_only_s2 = set(idx_in_2_and_3_only_s2)
    
    # 这里为了简便，unique to 2 和 3 的计算稍微简化一点，只用 s2 的索引去推
    # 注意：为了完全严谨，unique to 3 需要反向查，但在你的应用场景下通常只需保证 s2 没被消耗掉即可
    idx_unique_to_2 = set(idx2_unmatched_to_1) - set_2_and_3_only_s2
    
    # Unique to 3 的计算稍微复杂一点点：
    # 它是 (3中没匹配1的) - (3中没匹配1 且 匹配了2的)
    # 我们需要找出那些匹配了2的 3的索引
    idx_in_2_and_3_only_s3 = []
    if len(sources2_unmatched) > 0 and len(sources3_unmatched) > 0:
         indices_in_s3_unmatched = idx3_for_2_unmatched[mask_unmatched_matched] # 局部索引
         idx_in_2_and_3_only_s3 = idx3_unmatched_to_1[indices_in_s3_unmatched] # 全局索引
    
    idx_unique_to_3 = set(idx3_unmatched_to_1) - set(idx_in_2_and_3_only_s3)

    results = {
        'in_all_three': get_coords_from_indices(sources1, idx_in_all),
        f'in_{names[0]}_and_{names[1]}_only': get_coords_from_indices(sources1, idx_in_1_and_2_only),
        f'in_{names[0]}_and_{names[2]}_only': get_coords_from_indices(sources1, idx_in_1_and_3_only),
        f'in_{names[1]}_and_{names[2]}_only': get_coords_from_indices(sources2, idx_in_2_and_3_only_s2),
        f'unique_to_{names[0]}': get_coords_from_indices(sources1, idx_unique_to_1),
        f'unique_to_{names[1]}': get_coords_from_indices(sources2, idx_unique_to_2),
        f'unique_to_{names[2]}': get_coords_from_indices(sources3, idx_unique_to_3),
    }

    print("\n匹配完成！")
    return results


# 交叉匹配并且绘制在图上的具体位置
def venn_classified_and_visualize(instance,conn,cmap='ds9a',show=True,vmax=None):
    cursor1 = conn.execute('SELECT ra, dec, method FROM "robust+0.5_filtered_sources"')
    cursor2 = conn.execute('SELECT ra, dec, method FROM "robust-0.5_filtered_sources"')
    cursor3 = conn.execute('SELECT ra, dec, method FROM "allchan_filtered_sources"')
    rows1 = cursor1.fetchall()
    rows2 = cursor2.fetchall()
    rows3 = cursor3.fetchall()


    # 拆成两个数组
    ra_array1 = np.array([row[0] for row in rows1])
    dec_array1 = np.array([row[1] for row in rows1])
    # ra_array1 = ra_array1 + 360
    ra_array2 = np.array([row[0] for row in rows2])
    dec_array2 = np.array([row[1] for row in rows2])
    # ra_array2 = ra_array2 + 360
    ra_array3 = np.array([row[0] for row in rows3])
    dec_array3 = np.array([row[1] for row in rows3])
    # ra_array3 = ra_array3 + 360

    method1 = np.array([row[2] for row in rows1])
    method2 = np.array([row[2] for row in rows2])
    method3 = np.array([row[2] for row in rows3])

    sources_pos1 = SkyCoord(ra=ra_array1*u.deg,dec=dec_array1*u.deg)
    sources_pos2 = SkyCoord(ra=ra_array2*u.deg,dec=dec_array2*u.deg)
    sources_pos3 = SkyCoord(ra=ra_array3*u.deg,dec=dec_array3*u.deg)

    venn_dict = crossmatch_catalogs(sources_pos1,sources_pos2,sources_pos3,radius=0.1*u.arcsec,names=['robust+0.5','robust-0.5','allchan'])

    if show:
        # def plot_least_boxes_region(instance,xcoords,ycoords,T_img,std,cmap='Blues', fontsize = 25,show_ellipse=True,vmax=None):  #这里instance传入aaa_18517这种实例
        T_img = instance.Brightness_Temperature(instance.img,instance.Freq)
        # std = instance.Brightness_Temperature(std,instance.Freq) # T_std_original in K
        fontsize = 25
        if vmax is None:
            vmax = np.nanmax(T_img) # in K
        norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)

        fig, ax = plt.subplots(figsize=(20,12),subplot_kw={'projection': instance.wcs.celestial})         # X、Y轴标签字体大小
        plt.rcParams['xtick.labelsize'] = fontsize
        plt.rcParams['ytick.labelsize'] = fontsize
        # X、Y轴刻度标签字体大小
        plt.rcParams['axes.labelsize'] = fontsize
        # mean,median,std = sigma_clipped_stats(T_img,sigma=3.0)
        #norm1 = ImageNormalize(stretch=LogStretch(),vmin=std,vmax=vmax)
        imshow1 = ax.imshow(T_img,origin='lower',cmap=cmap,norm=norm) #,norm=norm1

        # in all three
        ra_array_7, dec_array_7 = venn_dict['in_all_three']['ra'], venn_dict['in_all_three']['dec']
        source_ra_pix, source_dec_pix = instance.wcs.celestial.all_world2pix(ra_array_7,dec_array_7,0)
        ax.plot(source_ra_pix,source_dec_pix,marker='+',color='white',markersize=5,linestyle='None'    #'#FF8C00'
                        ,markerfacecolor='none',markeredgewidth=1,label='In All Three')

        # in 1 and 2
        ra_array_4, dec_array_4 = venn_dict['in_robust+0.5_and_robust-0.5_only']['ra'], venn_dict['in_robust+0.5_and_robust-0.5_only']['dec']
        source_ra_pix, source_dec_pix = instance.wcs.celestial.all_world2pix(ra_array_4,dec_array_4,0)
        ax.plot(source_ra_pix,source_dec_pix,marker='o',color='#1E90FF',markersize=7,linestyle='None'    #'#1E90FF'
                        ,markerfacecolor='none',markeredgewidth=1,label='In 1 and 2')

        # in 1 and 3
        ra_array_5, dec_array_5 = venn_dict['in_robust+0.5_and_allchan_only']['ra'], venn_dict['in_robust+0.5_and_allchan_only']['dec']
        source_ra_pix, source_dec_pix = instance.wcs.celestial.all_world2pix(ra_array_5,dec_array_5,0)
        ax.plot(source_ra_pix,source_dec_pix,marker='o',color='#FF1493',markersize=7,linestyle='None'    #'#32CD32'
                        ,markerfacecolor='none',markeredgewidth=1,label='In 1 and 3')

        # in 2 and 3
        ra_array_6, dec_array_6 = venn_dict['in_robust-0.5_and_allchan_only']['ra'], venn_dict['in_robust-0.5_and_allchan_only']['dec']
        source_ra_pix, source_dec_pix = instance.wcs.celestial.all_world2pix(ra_array_6,dec_array_6,0)
        ax.plot(source_ra_pix,source_dec_pix,marker='o',color='#ADFF2F',markersize=7,linestyle='None'    #'#FF1493'
                        ,markerfacecolor='none',markeredgewidth=1,label='In 2 and 3')

        # only in 1
        ra_array_1, dec_array_1 = venn_dict['unique_to_robust+0.5']['ra'], venn_dict['unique_to_robust+0.5']['dec']
        source_ra_pix, source_dec_pix = instance.wcs.celestial.all_world2pix(ra_array_1,dec_array_1,0)
        ax.plot(source_ra_pix,source_dec_pix,marker='s',color='#1E90FF',markersize=7,linestyle='None'    #'#FFA500'
                        ,markerfacecolor='none',markeredgewidth=1,label='Only in 1')

        # only in 2
        ra_array_2, dec_array_2 = venn_dict['unique_to_robust-0.5']['ra'], venn_dict['unique_to_robust-0.5']['dec']
        source_ra_pix, source_dec_pix = instance.wcs.celestial.all_world2pix(ra_array_2,dec_array_2,0)
        ax.plot(source_ra_pix,source_dec_pix,marker='s',color='#FF1493',markersize=7,linestyle='None'    #'#00CED1'
                        ,markerfacecolor='none',markeredgewidth=1,label='Only in 2')

        # only in 3
        ra_array_3, dec_array_3 = venn_dict['unique_to_allchan']['ra'], venn_dict['unique_to_allchan']['dec']
        source_ra_pix, source_dec_pix = instance.wcs.celestial.all_world2pix(ra_array_3,dec_array_3,0)
        ax.plot(source_ra_pix,source_dec_pix,marker='s',color='#ADFF2F',markersize=7,linestyle='None'    #'#ADFF2F'#FF1493
                        ,markerfacecolor='none',markeredgewidth=1,label='Only in 3')

        ax.legend(loc='upper left', prop={'size': 15})

        # for i ,num_source in enumerate(range(len(source_ra_pix))):
        #     ax.text(source_ra_pix[num_source]+10,source_dec_pix[num_source]+10,s=f'{num_source+1}',color='white',fontsize=8)
        effect = withStroke(linewidth=2, foreground='grey')
        wcsaxes.add_beam(ax=ax,header=instance.head,pad=2,path_effects=[effect])
        ax.set_xlabel('R.A.')
        ax.set_ylabel('Dec.')
        ax.tick_params(axis='both', length=8, width=2,direction='in',color='white')
        ax.coords[0].display_minor_ticks(True)
        ax.coords[1].display_minor_ticks(True)
        ax.tick_params(axis='both',which='minor',length=4)
        ax.set_aspect('equal')
        effect = withStroke(linewidth=2, foreground='red')
        wcsaxes.add_beam(ax=ax,header=instance.head,pad=2,path_effects=[effect])
        fontprops = fm.FontProperties(size=20)
        sbar = wcsaxes.add_scalebar(ax=ax,length=10000 / instance.distance * 1000 * u.mas,label='10000 au',color='white',fontproperties=fontprops)
        cb1 = plt.colorbar(imshow1, pad=0, aspect=20,fraction=0.1)
        # cb1.set_ticks(LogLocator(base=10.0))  # 使 colorbar 刻度以 log 方式显示
        # cb1.ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=[0.2,0.3,0.4, 0.5,0.6, 0.7,0.8,0.9], numticks=10))
        cb1.ax.tick_params(labelsize=fontsize, length=8, width=2)  #设置色标刻度字体大小。
        cb1.ax.tick_params(direction='in')
        cb1.ax.xaxis.set_ticks_position('bottom')  # 让刻度移动到下方
        cb1.ax.tick_params(axis="both", which="major", width=2, length=8, direction='in')
        cb1.ax.tick_params(axis="both", which="minor", width=1.5, length=4, direction='in')
        font = {'family' : 'serif','color'  : 'darkred','weight' : 'normal','size'   : fontsize}
        cb1.set_label('Brightness Temperature (K)',fontdict=font) #设置colorbar的标签字体及其大小
        plt.show()

    return venn_dict, fig

def save_venn_sources(conn, venn_dict):
    """
    Save venn-classified sources into table `venn_sources`.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    venn_dict : dict
        Dictionary containing venn-classified sources with 'ra' and 'dec'.
    """

    # --- 定义 venn 分类编码 ---
    VENN_CODE_MAP = {
        'unique_to_robust+0.5': 1,
        'unique_to_robust-0.5': 2,
        'unique_to_allchan': 3,
        'in_robust+0.5_and_robust-0.5_only': 4,
        'in_robust+0.5_and_allchan_only': 5,
        'in_robust-0.5_and_allchan_only': 6,
        'in_all_three': 7,
    }

    # --- 建表（如果不存在） ---
    conn.execute(
        '''
        CREATE TABLE IF NOT EXISTS venn_sources (
            ra REAL,
            dec REAL,
            venn_code INTEGER
        )
        '''
    )

    # --- 写入数据 ---
    for key, code in VENN_CODE_MAP.items():

        if key not in venn_dict:
            continue

        ra_arr = venn_dict[key]['ra']
        dec_arr = venn_dict[key]['dec']

        if len(ra_arr) != len(dec_arr):
            raise ValueError(
                f'RA/Dec length mismatch in {key}: '
                f'{len(ra_arr)} vs {len(dec_arr)}'
            )

        rows = [
            (float(ra), float(dec), code)
            for ra, dec in zip(ra_arr, dec_arr)
        ]

        if rows:
            conn.executemany(
                'INSERT INTO venn_sources (ra, dec, venn_code) VALUES (?, ?, ?)',
                rows
            )

    conn.commit()


def mark_suspicious_sources(conn, source_array, tab_name='venn_sources'):
    """
    在指定表中新增一列 flag_suspicious，根据提供的 1-based 索引数组标记 True/False。

    Parameters
    ----------
    conn : sqlite3.Connection
        数据库连接对象。
    source_array : list or np.array
        包含存疑源索引的数组 (1-based index，即 1 代表第一行数据)。
        例如: [1, 5, 12] 代表第1、第5、第12个源存疑。
    tab_name : str, optional
        表名，默认为 'venn_sources'。
    """
    cursor = conn.cursor()

    # --- 1. 尝试添加列 (如果不存在) ---
    try:
        # SQLite 中 BOOLEAN 其实存的是 0 或 1
        cursor.execute(f'ALTER TABLE "{tab_name}" ADD COLUMN flag_suspicious BOOLEAN')
    except sqlite3.OperationalError as e:
        # 如果报错信息包含 "duplicate column name"，说明列已存在，我们直接忽略，继续后面的更新操作
        if 'duplicate column name' in str(e).lower():
            pass
        else:
            raise e

    # --- 2. 重置列数据 (覆盖旧数据) ---
    # 先将所有行的 flag_suspicious 设为 False (0)
    cursor.execute(f'UPDATE "{tab_name}" SET flag_suspicious = 0')

    # --- 3. 标记存疑源 ---
    if source_array is not None and len(source_array) > 0:
        # 确保输入是列表或数组，并转换为 Python int 格式 (executemany 需要)
        # 假设 source_array 里的索引对应 SQLite 的 rowid (1-based)
        if isinstance(source_array, np.ndarray):
            ids_to_mark = [(int(idx),) for idx in source_array]
        else:
            ids_to_mark = [(int(idx),) for idx in source_array]

        # 批量更新：WHERE rowid = ?
        # SQLite 的 rowid 默认是从 1 开始自增的，正好对应你的 1-based 索引
        cursor.executemany(
            f'UPDATE "{tab_name}" SET flag_suspicious = 1 WHERE rowid = ?', 
            ids_to_mark
        )

    conn.commit()
    print(f"Table '{tab_name}' updated: {len(ids_to_mark)} sources marked as suspicious.")



def delete_venn_sources(conn):
    """
    Delete all records from table `venn_sources` safely.
    
    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    """
    table_name = 'venn_sources'
    
    # 1. 检查表是否存在
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", 
        (table_name,)
    )
    if cursor.fetchone() is None:
        print(f"Table '{table_name}' does not exist. No deletion needed.")
        return

    # 2. 执行清空操作
    try:
        # TRUNCATE TABLE 在 SQLite 中不支持，标准做法是 DELETE FROM
        cursor = conn.execute(f'DELETE FROM {table_name}')
        deleted_count = cursor.rowcount
        conn.commit()
        
        print(f"Table '{table_name}' cleared. Deleted {deleted_count} rows.")

    except sqlite3.Error as e:
        conn.rollback() # 发生错误时回滚，保证数据库安全
        print(f"An error occurred while clearing '{table_name}': {e}")
        raise


# CASA imfit functions
import casatasks.importfits as importfits
import casatasks.imfit as imfit
import casatasks.exportfits as exportfits

def plot_residual_new(real_img,model_img,fitlog,header,wcs,pix2arcsec,RMS,cmap=hue_sat_value2_cmap,fontsize = 25,box_region=None,id_slice=None):
    font = {'color'  : 'black','weight' : 'normal','size'   : fontsize}#'family' : 'serif',
    fig = plt.figure(figsize=(40,12))
    plt.rcParams['xtick.labelsize'] = fontsize
    plt.rcParams['ytick.labelsize'] = fontsize
    # X、Y轴刻度标签字体大小
    plt.rcParams['axes.labelsize'] = fontsize
    # gs = gridspec.GridSpec(4,1) # 创立2 * 6 网格
    # gs.update(wspace=-0.334,hspace=0)
    # 对第一行进行绘制
    # gs_set_array = np.array([gs[0,0:1],gs[0,1:2],gs[0,2:3]])

    # 计算指数范围
    vmin = RMS * 3
    vmax = real_img.max()
    min_exp = int(np.floor(np.log10(vmin)))
    max_exp = int(np.ceil(np.log10(vmax)))

    tickvals = []
    for exp in range(min_exp, max_exp + 1):
        for coef in [1, 4]:
            val = coef * 10**exp
            if vmin <= val <= vmax:
                tickvals.append(val)

    ax0 = fig.add_subplot(131,projection=wcs)
    ax1 = fig.add_subplot(132,projection=wcs)
    ax2 = fig.add_subplot(133,projection=wcs)
    ax = np.array([ax0,ax1,ax2])
    # norm1 = LogNorm(vmin=vmin,vmax=real_img.max())
    norm1 = PowerNorm(gamma=0.5,vmin=0,vmax=vmax)
    # print(real_img.max())
    imshow1 = ax[0].imshow(real_img,norm=norm1,origin='lower',cmap=hue_sat_value2_cmap#'viridis'
                            ,alpha=1,interpolation='bicubic')
    effect = withStroke(linewidth=2, foreground='red')
    wcsaxes.add_beam(ax=ax[0],header=header,pad=2,path_effects=[effect])
    ax[0].text(0.02, 0.9,'Data', transform=ax[0].transAxes, 
    verticalalignment='top', horizontalalignment='left', fontsize=30,color='black')

    w2h = ax[0].get_window_extent().width / ax[0].get_window_extent().height
    # ax1_divider = make_axes_locatable(ax[0])
    # # Add an Axes above the main Axes.
    # cax1 = ax1_divider.append_axes("top", size="{}%".format(1/(20 * w2h)*100), pad=0)
    cb1 = plt.colorbar(imshow1,ax=ax[0], aspect=20,fraction=0.1,orientation='horizontal')
    cb1.set_ticks(LogLocator(base=10.0))  # 使 colorbar 刻度以 log 方式显示
    cb1.ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=[0.2,0.3,0.4, 0.5,0.6, 0.7,0.8,0.9], numticks=10))
    cb1.ax.tick_params(labelsize=fontsize, length=8, width=2)  #设置色标刻度字体大小。
    cb1.ax.tick_params(direction='in')
    cb1.ax.xaxis.set_ticks_position('bottom')  # 让刻度移动到下方
    cb1.ax.tick_params(axis="both", which="major", width=2, length=8, direction='in')
    cb1.ax.tick_params(axis="both", which="minor", width=1.5, length=4, direction='in')
    font = {'family' : 'serif','color'  : 'darkred','weight' : 'normal','size'   : fontsize}
    cb1.set_label('Intensity (Jy/beam)',fontdict=font) #设置colorbar的标签字体及其大小
        
    ax[0].set_ylabel('Dec. Offset')
    ax[0].set_xlabel('R.A. Offset')
    ax[0].coords[0].display_minor_ticks(True)
    ax[0].coords[1].display_minor_ticks(True)
    ax[0].coords[0].display_minor_ticks(True)
    ax[0].coords[1].set_ticks_position('lr')
    ax[0].coords[0].set_ticks_position('b')
    # ax[0].coords[0].coord_wrap = 180 * u.degree
    # ax[0].coords[0].set_major_formatter('s.ss')
    # ax[0].coords[1].set_major_formatter('s.ss')
    ax[0].tick_params(axis='both', length=8, width=2, direction='in')

    # print(model_img)
    imshow2 = ax[1].imshow(model_img,norm=norm1,origin='lower',cmap=hue_sat_value2_cmap
                            ,alpha=1,interpolation='bicubic') #gaussian_array_B+gaussian_array
    wcsaxes.add_beam(ax=ax[1],header=header,pad=2,path_effects=[effect])
    ax[1].text(0.02, 0.9,'Model', transform=ax[1].transAxes, 
    verticalalignment='top', horizontalalignment='left', fontsize=30,color='black')

    w2h = ax[1].get_window_extent().width / ax[1].get_window_extent().height
    cb2 = plt.colorbar(imshow1,ax=ax[1], aspect=20,fraction=0.1,orientation='horizontal')
    cb2.set_ticks(LogLocator(base=10.0))  # 使 colorbar 刻度以 log 方式显示
    cb2.ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=[0.2,0.3,0.4, 0.5,0.6, 0.7,0.8,0.9], numticks=10))
    cb2.ax.tick_params(labelsize=fontsize, length=8, width=2)  #设置色标刻度字体大小。
    cb2.ax.tick_params(direction='in')
    cb2.ax.xaxis.set_ticks_position('bottom')  # 让刻度移动到下方
    cb2.ax.tick_params(axis="both", which="major", width=2, length=8, direction='in')
    cb2.ax.tick_params(axis="both", which="minor", width=1.5, length=4, direction='in')
    font = {'family' : 'serif','color'  : 'darkred','weight' : 'normal','size'   : fontsize}
    cb2.set_label('Intensity (Jy/beam)',fontdict=font) #设置colorbar的标签字体及其大小    

    ax[1].coords[0].set_ticks_position('b')
    ax[1].coords[0].default_label = ''
    ax[1].set_xlabel('R.A. Offset')
    ax[1].coords[0].display_minor_ticks(True)
    ax[1].coords[1].default_label = ''
    #ax[1].set_ylabel('Dec. Offset')
    ax[1].coords[1].set_ticks_visible(True)
    ax[1].coords[1].set_ticklabel_visible(False)
    ax[1].coords[1].display_minor_ticks(True)
    ax[1].coords[0].display_minor_ticks(True)
    #ax[1].coords[0].set_ticks([0,1] * u.arcsec)
    # ax[1].coords[0].coord_wrap = 180 * u.degree
    # ax[1].coords[0].set_major_formatter('s.ss')
    # ax[1].coords[1].set_major_formatter('s.ss')
    ax[1].tick_params(axis='both', length=8, width=2, direction='in')

    n_comp = len(fitlog)   # 成分数量
    for i in range(n_comp):
        # 世界坐标中心
        ra_center = fitlog['LongICRS'][i]
        dec_center = fitlog['LatICRS'][i]
        ra_pix, dec_pix = wcs.all_world2pix(ra_center, dec_center, 0)

        # convolved
        conmaj = fitlog['ConMaj'][i] / pix2arcsec
        conmin = fitlog['ConMin'][i] / pix2arcsec
        conPA  = fitlog['ConPA'][i]

        e_con = Ellipse(
            xy=(ra_pix, dec_pix),
            width=conmaj, height=conmin, angle=conPA + 90,
            edgecolor='grey', linestyle='--', facecolor='none', lw=4,
            label='convolved FWHM' if i == 0 else None
        )
        ax[1].add_patch(e_con)

        # deconvolved
        deconmaj = fitlog['DeconMaj'][i] / pix2arcsec
        deconmin = fitlog['DeconMin'][i] / pix2arcsec
        deconPA  = fitlog['DeconPA'][i]

        e_decon = Ellipse(
            xy=(ra_pix, dec_pix),
            width=deconmaj, height=deconmin, angle=deconPA + 90,
            edgecolor='black', linestyle='-', facecolor='none', lw=3,
            label='deconvolved FWHM' if i == 0 else None
        )
        ax[1].add_patch(e_decon)

    # 图例（只显示一次标签）
    ax[1].legend(fontsize=20, bbox_to_anchor=(0.98, 0.95), loc='upper right')
    residual_map = real_img  - model_img
    vmin = residual_map.min()
    vmax = residual_map.max()
    vset = max(np.abs(vmin),np.abs(vmax))
    imshow3 = ax[2].imshow(residual_map,origin='lower',vmin=-vset,vmax=vset,cmap='bwr'#'bwr'
                            ,interpolation='bicubic')#,cmap='Greys_r'
    # 计算图像的均值和标准偏差
    mean = np.mean(residual_map)
    stddev = RMS
    # 设置 σ 级别
    sigma_levels = [0 + n * stddev for n in np.array([-3,-2,-1,1,2,3])]  # 1σ, 2σ, 3σ
    # 添加等高线表示 1σ, 2σ, 3σ
    color_set = np.array(['y','b','r','r','b','y'])
    labels = np.array(['$3\sigma$','$2\sigma$','$1\sigma$','$1\sigma$','$2\sigma$','$3\sigma$'])
    for index,level in enumerate(sigma_levels):
        contour = ax[2].contour(residual_map, levels=[level], colors=color_set[index], linewidths=1, linestyles='dashed',label=labels[index])
        custom_labels = [labels[index]]  # 可以根据需要自定义标签

        # 添加等高线标签
        ax[2].clabel(contour, inline=True, fontsize=10, fmt=dict(zip(contour.levels, custom_labels)))

        #clabels = plt.clabel(contour, inline=True, fontsize=15,use_clabeltext=True)  # 添加标签
    wcsaxes.add_beam(ax=ax[2],header=header,pad=2,path_effects=[effect])
    ax[2].text(0.02, 0.9,'Residual', transform=ax[2].transAxes, 
    verticalalignment='top', horizontalalignment='left', fontsize=30,color='black')

    w2h = ax[2].get_window_extent().width / ax[2].get_window_extent().height
    ax3_divider = make_axes_locatable(ax[2])
    # Add an Axes above the main Axes.
    cax3 = ax3_divider.append_axes("top", size="{}%".format(1/(20 * w2h)*100), pad=0)
    cb3 = fig.colorbar(imshow3,cax=cax3,orientation='horizontal')
    cb3.ax.coords[1].default_label = ''
    cb3.ax.coords[1].set_ticks_visible(False)
    cb3.ax.coords[1].set_ticklabel_visible(False)
    cb3.ax.coords[0].set_ticklabel_position('b')
    cb3.ax.coords[0].set_ticks_position('b')
    cb3.ax.coords.grid(False)
    #cb1.ax.coords[0].set_ticks([10,100] * u.dimensionless_unscaled)
    cb3.ax.coords[0].display_minor_ticks(True)
    cb3.ax.coords[0].set_ticklabel(size=20,exclude_overlapping=True)
    if np.floor(np.max(residual_map)) > 5:
        ticks_res = np.linspace(np.ceil(-np.max(residual_map))+2,np.floor(np.max(residual_map))-2,7)
    else:
        ticks_res = np.linspace(np.ceil(-np.max(residual_map)),np.floor(np.max(residual_map)),7)
    # cb3.ax.coords[0].set_ticks(ticks_res * u.dimensionless_unscaled,size=8)
    #cb1.ax.coords[0].set_ticks([5,50,150] * u.dimensionless_unscaled,size=4)

    cb3.ax.coords[0].set_axislabel_position('t')
    cb3.ax.coords[0].set_axislabel('Intensity (Jy/beam)',fontdict=font) #设置colorbar的标签字体及其大小
    cb3.ax.tick_params(axis="both", which="major", width=2, length=8, direction='in')

    ax[2].set_xlabel('R.A. Offset')
    ax[2].coords[0].display_minor_ticks(True)
    ax[2].coords[1].set_ticks_position('l')
    ax[2].coords[0].set_ticks_position('b')
    ax[2].coords[1].default_label = ''
    ax[2].coords[1].display_minor_ticks(True)
    ax[2].coords[1].set_ticks_visible(True)
    ax[2].coords[1].set_ticklabel_visible(False)
    # ax[2].coords[0].coord_wrap = 180 * u.degree
    # ax[2].coords[0].set_major_formatter('s.ss')
    # ax[2].coords[1].set_major_formatter('s.ss')
    ax[2].tick_params(axis='both', length=8, width=2,direction='in')
    # print(real_img,'\n',model_img)

    if box_region is not None:
        items = box_region.split(',')
        items_stripped = [item.strip() for item in items]
        numbers = [float(item) for item in items_stripped]
        x1, y1, x2, y2 = numbers
        box = Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='black', facecolor='none', lw=1.5)
        ax[0].add_patch(box)
        box = Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='black', facecolor='none', lw=1.5)
        ax[1].add_patch(box)
        box = Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='black', facecolor='none', lw=1.5)
        ax[2].add_patch(box)

    if id_slice is not None:
        idx,idy = id_slice
        ax[0].plot(idy, idx, marker='x', color='black', markersize=8, linestyle='None', markeredgewidth=2)
        ax[1].plot(idy, idx, marker='x', color='black', markersize=8, linestyle='None', markeredgewidth=2)
        ax[2].plot(idy, idx, marker='x', color='black', markersize=8, linestyle='None', markeredgewidth=2)
        ax[0].axhline(y=idx, color='black', linestyle=':', linewidth=1)
        ax[0].axvline(x=idy, color='black', linestyle=':', linewidth=1)
        ax[1].axhline(y=idx, color='black', linestyle=':', linewidth=1)
        ax[1].axvline(x=idy, color='black', linestyle=':', linewidth=1)
        ax[2].axhline(y=idx, color='black', linestyle=':', linewidth=1)
        ax[2].axvline(x=idy, color='black', linestyle=':', linewidth=1)



    plt.show()
    return fig



def fwhm_to_sigma(fwhm):
    """FWHM转标准差"""
    return fwhm / (2 * np.sqrt(2 * np.log(2)))

def plot_gauss_slices_astropy_old(real_img,model_img, wcs, pix2arcsec ,log, axis='x', idx=0):
    """
    用 Astropy Gaussian2D 绘制一维切片（x或y方向），包括每个高斯分量的贡献和总模型
    axis: 'x'表示取某一行，'y'表示取某一列
    idx: 切片的索引（行号或列号）
    """
    ny, nx = real_img.shape
    x = np.arange(nx)
    y = np.arange(ny)
    if axis == 'x':
        slice_real = real_img[idx, :]
        slice_model = model_img[idx, :]
        plt.plot(x, slice_real, 'k-', label='Real Data')
        plt.plot(x, slice_model, 'b-', label='Model Img')
    else:
        slice_real = real_img[:, idx]
        slice_model = model_img[:, idx]
        plt.plot(y, slice_real, 'k-', label='Real Data')
        plt.plot(y, slice_model, 'b-', label='Model Img')

    n_comp = len(log['Peak'])
    total_model = np.zeros_like(model_img)
    model_comp = []
    total_model_slice = np.zeros_like(slice_real)
    for i in range(n_comp):
        amp = log['Peak'][i]
        ra_center = log['LongICRS'][i]
        dec_center = log['LatICRS'][i]
        ra_pix, dec_pix = wcs.all_world2pix(ra_center, dec_center, 0)
        sigma_x = fwhm_to_sigma(log['ConMaj'][i]) / pix2arcsec
        sigma_y = fwhm_to_sigma(log['ConMin'][i]) / pix2arcsec
        theta = np.deg2rad(log['ConPA'][i])  # astropy用弧度

        # 创建 Astropy Gaussian2D
        gauss_model = Gaussian2D(amplitude=amp, x_mean=ra_pix, y_mean=dec_pix,
                                 x_stddev=sigma_x, y_stddev=sigma_y, theta=theta + np.pi/2)

        # meshgrid
        xs, ys = np.meshgrid(x, y)
        gauss_img = gauss_model(xs, ys) # 直接评估

        # 取切片
        if axis == 'x':
            gauss_slice = gauss_img[idx, :]
            plt.plot(x, gauss_slice, '--', label=f'Comp {i+1}')
        else:
            gauss_slice = gauss_img[:, idx]
            plt.plot(y, gauss_slice, '--', label=f'Comp {i+1}')
        total_model_slice += gauss_slice
        total_model += gauss_img
        model_comp.append(gauss_img)

    # 总模型
    if axis == 'x':
        plt.plot(x, total_model_slice, 'r:', lw=2, label='Model Total')
        plt.xlabel('x')
    else:
        plt.plot(y, total_model_slice, 'r:', lw=2, label='Model Total')
        plt.xlabel('y')

    if real_img.max() >= 50 * np.median(real_img):
        plt.yscale('log')
        plt.ylim(bottom=np.median(real_img) / 10)
    plt.legend()
    plt.title(f'Slice at {axis}={idx} (Astropy Gaussian2D)')
    plt.show()
    return total_model, model_comp

def plot_gauss_slices_astropy(real_img, model_img, wcs, pix2arcsec, log, axis='x', idx=0, idy=0, logscale=False, box_region=None):
    """
    axis: 'x'表示取某一行，'y'表示取某一列，'both'表示同时画x和y两个切片
    idx: x方向（行）切片索引
    idy: y方向（列）切片索引
    box_region: 字符串，格式如 '3,4,34,24' 或 '3  , 4,  34, 24'，代表(x1, y1, x2, y2)
    """
    ny, nx = real_img.shape
    x = np.arange(nx)
    y = np.arange(ny)
    
    # 处理 box_region
    def get_box_coords(box_region):
        if box_region is None:
            return None
        try:
            items = box_region.split(',')
            items_stripped = [item.strip() for item in items]
            numbers = [float(item) for item in items_stripped]
            if len(numbers) == 4:
                return numbers
            else:
                return None
        except Exception:
            return None

    box_coords = get_box_coords(box_region)
    # box_coords = [x1, y1, x2, y2] if valid else None

    def plot_slice(ax, axis, idx, real_img, model_img, log, wcs, pix2arcsec, box_coords):
        if axis == 'x':
            pos = 'y'
            slice_real = real_img[idx, :]
            slice_model = model_img[idx, :]
            ax.plot(x, slice_real, 'k-', label='Real Data')
            ax.plot(x, slice_model, 'b-', label='Model Img')
        else:
            pos = 'x'
            slice_real = real_img[:, idx]
            slice_model = model_img[:, idx]
            ax.plot(y, slice_real, 'k-', label='Real Data')
            ax.plot(y, slice_model, 'b-', label='Model Img')

        n_comp = len(log['Peak'])
        total_model = np.zeros_like(model_img)
        model_comp = []
        total_model_slice = np.zeros_like(slice_real)
        for i in range(n_comp):
            amp = log['Peak'][i]
            ra_center = log['LongICRS'][i]
            dec_center = log['LatICRS'][i]
            ra_pix, dec_pix = wcs.all_world2pix(ra_center, dec_center, 0)
            sigma_x = fwhm_to_sigma(log['ConMaj'][i]) / pix2arcsec
            sigma_y = fwhm_to_sigma(log['ConMin'][i]) / pix2arcsec
            theta = np.deg2rad(log['ConPA'][i])  # astropy用弧度

            gauss_model = Gaussian2D(amplitude=amp, x_mean=ra_pix, y_mean=dec_pix,
                                     x_stddev=sigma_x, y_stddev=sigma_y, theta=theta + np.pi/2)
            xs, ys = np.meshgrid(x, y)
            gauss_img = gauss_model(xs, ys)
            if axis == 'x':
                gauss_slice = gauss_img[idx, :]
                ax.plot(x, gauss_slice, '--', label=f'Comp {i+1}')
            else:
                gauss_slice = gauss_img[:, idx]
                ax.plot(y, gauss_slice, '--', label=f'Comp {i+1}')
            total_model_slice += gauss_slice
            total_model += gauss_img
            model_comp.append(gauss_img)

        # 总模型
        if axis == 'x':
            ax.plot(x, total_model_slice, 'r:', lw=2, label='Model Total')
            ax.set_xlabel('x')
            # 画虚线（竖线）在 x1,x2 处
            if box_coords is not None:
                x1, y1, x2, y2 = box_coords
                ax.axvline(x1, color='k', linestyle='--')
                ax.axvline(x2, color='k', linestyle='--')
        else:
            ax.plot(y, total_model_slice, 'r:', lw=2, label='Model Total')
            ax.set_xlabel('y')
            # 画虚线（竖线）在 y1,y2 处
            if box_coords is not None:
                x1, y1, x2, y2 = box_coords
                ax.axvline(y1, color='k', linestyle='--')
                ax.axvline(y2, color='k', linestyle='--')
        if logscale:
            ax.set_yscale('log')
            ax.set_ylim(bottom=np.median(real_img) / 10)
        ax.legend()
        ax.set_title(f'Slice at {pos}={idx} (Astropy Gaussian2D)')
        return total_model, model_comp

    if axis in ['x', 'y']:
        fig, ax = plt.subplots(figsize=(7, 5))
        total_model, model_comp = plot_slice(ax, axis, idx if axis=='x' else idy, real_img, model_img, log, wcs, pix2arcsec, box_coords)
        plt.show()
        return total_model, model_comp, fig
    elif axis == 'both':
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        total_model_x, model_comp_x = plot_slice(axes[0], 'x', idx, real_img, model_img, log, wcs, pix2arcsec, box_coords)
        total_model_y, model_comp_y = plot_slice(axes[1], 'y', idy, real_img, model_img, log, wcs, pix2arcsec, box_coords)
        plt.tight_layout()
        plt.show()
        # return {"x": (total_model_x, model_comp_x), "y": (total_model_y, model_comp_y), "fig": fig}
        return total_model_x, model_comp_x, total_model_y, model_comp_y, fig
    
def sum_circle_with_plot(fits_file, fitlog_file, fitlog_summary_file, rms_noise, sigma_level=3, cmap=hue_sat_value2_cmap):
    """
    Perform sum_circle photometry and plot the aperture on the image.

    Parameters:
    - fits_file: str, path to the FITS file containing the image data.
    - fitlog_file: str, path to the CASA fit log file.
    - rms_noise: float, RMS noise level in Jy/beam.
    - sigma_level: int, sigma level for the aperture radius (default: 3).
    - cmap: matplotlib colormap, colormap for the image (default: hue_sat_value2_cmap).

    Returns:
    - flux: float, total flux within the aperture.
    - flux_err: float, error in the total flux.
    """
    # Load the FITS file and fit log
    instance = Formation_Cluster(fits_file)
    df = pd.read_csv(fitlog_summary_file, index_col=False, header=0, delim_whitespace=True, skiprows=1)
    fitlog = df.shift(axis=1)

    offset_val = 0.0
    offset_err = 0.0

    # Extract zero-level offset from the fit log
    with open(fitlog_file, "r") as f:
        for line in f:
            if "Zero level offset fit:" in line:
                match = re.search(r"([-+]?\d+\.\d+e?[-+]?\d*)\s*\+/-\s*([-+]?\d+\.\d+e?[-+]?\d*)", line)
                if match:
                    offset_val = float(match.group(1))
                    offset_err = float(match.group(2))
                    print("Zero offset =", offset_val, "Jy/beam")
                    print("Error =", offset_err, "Jy/beam")

    # Extract source center and aperture radius
    ra_center = fitlog['LongICRS'][0]
    dec_center = fitlog['LatICRS'][0]
    x_cen, y_cen = instance.wcs.celestial.wcs_world2pix(ra_center, dec_center, 0)
    mm_fwhm = np.sqrt(fitlog['ConMaj'][0] * fitlog['ConMin'][0]) / instance.PIXEL_SCALE.value
    r = mm_fwhm / (2 * np.sqrt(2 * np.log(2))) * sigma_level  # Aperture radius in pixels

    # Perform sum_circle photometry
    flux, flux_err, _ = sep.sum_circle(
        instance.img - offset_val,
        np.array([x_cen]),
        np.array([y_cen]),
        r=np.array([r]),
        err=rms_noise
    )
    flux = flux[0] / (instance.sr_beam / instance.sr_pixel)
    flux_err = flux_err[0] / (instance.sr_beam / instance.sr_pixel)

    print(f"Total flux: {flux:.6f} Jy")
    print(f"Flux error: {flux_err:.6f} Jy")

    # Plot the image with the aperture
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection=instance.wcs.celestial)

    # Display the image
    norm = simple_norm(instance.img, 'linear', percent=99.5)
    im = ax.imshow(instance.img, origin='lower', cmap=cmap, norm=norm)

    # Add the aperture circle
    circ = Circle((x_cen, y_cen), r, edgecolor='red', facecolor='none', lw=2)
    ax.add_patch(circ)

    # Add a marker at the center
    ax.plot(x_cen, y_cen, marker='+', color='yellow', markersize=10, mew=1.5)

    # Add a colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Flux [Jy/beam]")

    ax.set_xlabel("RA (J2000)")
    ax.set_ylabel("Dec (J2000)")
    plt.title(f"{sigma_level}$\\sigma$ aperture for sum_circle photometry",fontsize=25)
    plt.show()

    return flux, flux_err
# 单独拟合，放弃批量处理
def casa_imfit_manually(fits_url,base_instance,manual_estimate=None,show_fitting_result=False
                        ,zero_level=False,box_set=None,show_one_dim_result=False,axis_set='both',idx=0,idy=0,logscale=False,full_return=False
                        ,sum_circle=False,RMS=1.53e-5,sigma_level=3,fcen_ra=None,fcen_dec=None,Ipeak=None,point_source=False,
                        savepath=None,fig_basename=''):
    """
    fits_url: fits文件的路径
    base_instance: AAA class的实例，用于提供像素尺度beam等信息
    manual_estimate: 手动提供的估计文件名，如果为None则自动生成,在和fits同级别的文件夹上，自动生成的则放在新建的文件夹内
    show_fitting_result: 是否显示拟合结果图
    box_set: 如果提供了box_set，则在imfit中使用box参数,box='81, 81, 112, 109'类似的
    """
    basename = os.path.basename(fits_url).replace(".fits", "")
    dirname = os.path.dirname(fits_url)
    os.makedirs(os.path.join(dirname, basename), exist_ok=True)
    working_dir = os.path.join(dirname, basename)
    os.makedirs(os.path.join(dirname, basename, "casa_img"), exist_ok=True)
    os.makedirs(os.path.join(dirname, basename, "modelimg"), exist_ok=True)
    hdu_this = fits.open(fits_url)
    cutout_size = hdu_this[0].data.shape
    wcs_cutout = WCS(hdu_this[0].header)
    if manual_estimate is None:
        estimate_path = os.path.join(working_dir, "estimate_auto.dat")
        if (fcen_ra is not None) and (fcen_dec is not None) and (Ipeak is not None):
        # if fcen_x and fcen_y:
            fcen_x, fcen_y = wcs_cutout.celestial.all_world2pix(fcen_ra, fcen_dec, 0)
            if point_source:
                with open(estimate_path, "w") as f:
                    f.write("{}, {}, {}, {}arcsec, {}arcsec, {}deg, xyabp".format(Ipeak,fcen_x,fcen_y,
                                    base_instance.BEAM_MAJOR.value,base_instance.BEAM_MINOR.value,base_instance.BEAM_PA.value))
            else:
                with open(estimate_path, "w") as f:
                    f.write("{}, {}, {}, {}arcsec, {}arcsec, {}deg, xy".format(Ipeak,fcen_x,fcen_y,
                                    base_instance.BEAM_MAJOR.value,base_instance.BEAM_MINOR.value,base_instance.BEAM_PA.value))
        else:
            flux_center_guess = hdu_this[0].data[cutout_size[0]//2,cutout_size[1]//2]
            with open(estimate_path, "w") as f:
                f.write("{}, {}, {}, {}arcsec, {}arcsec, {}deg".format(flux_center_guess,cutout_size[0]/2,cutout_size[1]/2,
                                base_instance.BEAM_MAJOR.value,base_instance.BEAM_MINOR.value,base_instance.BEAM_PA.value))
    else:
        estimate_path = os.path.join(dirname, manual_estimate)

    importfits(
        fits_url,
        imagename=os.path.join(working_dir, "casa_img"),
        overwrite=True
    )

    if box_set is None:
        imfit(
            imagename=os.path.join(working_dir, "casa_img"),
            estimates=estimate_path,
            logfile=os.path.join(working_dir, "fit_log.dat"),
            append=False,
            summary=os.path.join(working_dir, "fit_summary_log.dat"),
            overwrite=True,
            model=os.path.join(working_dir, "modelimg", "casa_model")
        )
    else:
        imfit(
            imagename=os.path.join(working_dir, "casa_img"),
            box=box_set,
            estimates=estimate_path,
            dooff = zero_level,
            logfile=os.path.join(working_dir, "fit_log.dat"),
            append=False,
            summary=os.path.join(working_dir, "fit_summary_log.dat"),
            overwrite=True,
            model=os.path.join(working_dir, "modelimg", "casa_model")
        )

    exportfits(
        imagename=os.path.join(working_dir, "modelimg", "casa_model"),
        fitsimage=os.path.join(working_dir, "model.fits"),
        overwrite=True
    )

    fitlog_name = os.path.join(working_dir, "fit_summary_log.dat"),
    df = pd.read_csv(fitlog_name[0],index_col=False,header=0,delim_whitespace=True,skiprows=1)
    fitlog = df.shift(axis=1)
    real_img = hdu_this[0].data
    wcs = WCS(hdu_this[0].header).celestial
    hdu_model = fits.open(os.path.join(working_dir, "model.fits"))
    # print(hdu_model[0])
    # print(hdu_model[0].data)
    if hdu_model[0].data.ndim == 2:
        model_img = hdu_model[0].data
    elif hdu_model[0].data.ndim == 4:
        model_img = hdu_model[0].data[0][0]
    elif hdu_model[0].data.ndim == 3:
        model_img = hdu_model[0].data[0]
    else:
        raise ValueError("Model image has unsupported number of dimensions.")

    if show_fitting_result:
        print('showing fitting result')
        # mean,median,RMS = sigma_clipped_stats(base_instance.img,sigma=3.0,maxiters=10)
        fig_residual = plot_residual_new(
            real_img,
            model_img,
            fitlog,
            header=base_instance.head,
            wcs=wcs,
            pix2arcsec=base_instance.PIXEL_SCALE.value,
            RMS=RMS,
            cmap=hue_sat_value2_cmap,
            box_region=box_set,
            id_slice=(idx, idy)
        )

    if show_one_dim_result:
        print('showing one dimensional fitting result')
        total_model_x, model_comp_x, total_model_y, model_comp_y, fig_slice = plot_gauss_slices_astropy(real_img
                                  , model_img
                                  , wcs
                                  , base_instance.PIXEL_SCALE.value 
                                  , fitlog
                                  , axis=axis_set, 
                                  idx=idx,idy=idy,
                                  logscale=logscale,
                                    box_region=box_set
                                  )
    else:
        total_model = None
        model_comp = None

    if savepath is not None:
        figname1 = fig_basename + '_residual.png'
        figname2 = fig_basename + '_slice.png'
        fig_residual.savefig(os.path.join(savepath,figname1),dpi=300,bbox_inches = 'tight')
        fig_slice.savefig(os.path.join(savepath,figname2),dpi=300,bbox_inches = 'tight')

    if sum_circle:
        flux, flux_err = sum_circle_with_plot(
            fits_file=fits_url,
            fitlog_file=os.path.join(working_dir, "fit_log.dat"),
            fitlog_summary_file=os.path.join(working_dir, "fit_summary_log.dat"),
            rms_noise=RMS,
            sigma_level=sigma_level,
            cmap='ds9a'
        )
        print(f"Sum_circle flux: {flux:.6f} Jy")
        print(f"Sum_circle flux error: {flux_err:.6f} Jy")
    
    if full_return and sum_circle:
        return fitlog, total_model, model_comp, real_img, model_img, flux, flux_err
    elif full_return and not sum_circle:
        return fitlog, total_model, model_comp, real_img, model_img
    elif sum_circle and not full_return:
        return fitlog, flux, flux_err
    else:
        return fitlog


def write_imfit_estimates(fits_path, sources, output_filename="estimate_auto.dat"):
    """
    生成 CASA imfit 的初始猜测文件 (.dat)。
    
    参数:
    ----------
    fits_path : str
        FITS 文件的路径 (用于读取 WCS 和 Beam 信息)。
    sources : list of dict
        包含每个源信息的列表。每个字典可以包含以下键:
        - 'peak': (必须) 峰值强度猜测值。
        - 'x', 'y': (可选) 像素坐标。
        - 'ra', 'dec': (可选) 天球坐标 (度)。如果提供了 x,y 则忽略此项。
        - 'major', 'minor', 'pa': (可选) 形状猜测 (字符串, e.g., '10arcsec'). 默认使用 Beam 大小。
        - 'fixed': (可选) 固定的参数字符串 (e.g., 'xyabp', 'f', 'ab'). 默认为空。
    output_filename : str
        输出文件的名称，默认为 estimate_auto.dat。
    
    返回:
    ----------
    saved_path : str
        生成文件的完整路径。
    """
    
    # 1. 确定基础路径
    base_dir = os.path.dirname(fits_path)
    estimate_path = os.path.join(base_dir, output_filename)
    
    # 2. 读取 FITS 头文件以获取 WCS 和 Beam 信息
    if not os.path.exists(fits_path):
        raise FileNotFoundError(f"FITS file not found: {fits_path}")
        
    with fits.open(fits_path) as hdul:
        header = hdul[0].header
        wcs = WCS(header)
        
        # 获取 Beam 信息作为默认形状
        # 注意：不同 FITS 头文件关键词可能略有不同，这里处理标准情况
        try:
            # 转换为 arcsec 用于写入文件 (假设 header 中 BMAJ 是度)
            bmaj = header.get('BMAJ', 0) * 3600 
            bmin = header.get('BMIN', 0) * 3600
            bpa = header.get('BPA', 0)
            
            default_major = f"{bmaj:.6f}arcsec"
            default_minor = f"{bmin:.6f}arcsec"
            default_pa = f"{bpa:.6f}deg"
        except Exception as e:
            print(f"Warning: Could not read Beam info from header ({e}). Using dummy defaults.")
            default_major = "1.0arcsec"
            default_minor = "1.0arcsec"
            default_pa = "0.0deg"

    # 3. 写入文件
    with open(estimate_path, "w") as f:
        # 写入注释头 (可选，但有助于阅读)
        f.write("# peak, x, y, major, minor, pa, fixed\n")
        
        for src in sources:
            # --- 步骤 A: 获取峰值 ---
            ipeak = src.get('peak')
            if ipeak is None:
                print("Warning: Skipping a source without 'peak' value.")
                continue
                
            # --- 步骤 B: 获取坐标 (优先像素，其次 RA/Dec) ---
            if 'x' in src and 'y' in src:
                fcen_x = src['x']
                fcen_y = src['y']
            elif 'ra' in src and 'dec' in src:
                # 将 RA/Dec (deg) 转换为 像素坐标
                # 0 表示 origin=0 (numpy style)，CASA imfit 读取时通常兼容
                fcen_x, fcen_y = wcs.celestial.all_world2pix(src['ra'], src['dec'], 0)
                # 如果是标量，取 float；如果是数组，取第一个元素
                if isinstance(fcen_x, (np.ndarray, list)):
                    fcen_x, fcen_y = float(fcen_x), float(fcen_y)
            else:
                print("Warning: Skipping a source without coordinates (x,y or ra,dec).")
                continue
            
            # --- 步骤 C: 获取形状 (默认为 Beam) ---
            major = src.get('major', default_major)
            minor = src.get('minor', default_minor)
            pa = src.get('pa', default_pa)
            
            # --- 步骤 D: 获取固定参数字符串 ---
            # 例如: 'xy' (固定位置), 'ab' (固定大小), 'p' (固定PA), 'f' (固定通量?)
            # 根据 CASA imfit 文档，通常把要固定的参数缩写放在最后
            fixed_str = src.get('fixed', "")
            
            # --- 步骤 E: 格式化写入 ---
            # 格式: peak, x, y, major, minor, pa, fixed_string
            line = f"{ipeak}, {fcen_x}, {fcen_y}, {major}, {minor}, {pa}, {fixed_str}\n"
            f.write(line)
            
    return estimate_path


def sum_circle_with_plot_photutils(
    fits_file,
    fitlog_file,
    fitlog_summary_file,
    rms_noise,
    sigma_level=3,
    cmap=hue_sat_value2_cmap,
):
    """
    Perform circular aperture photometry using photutils and plot the aperture.

    Parameters
    ----------
    fits_file : str
        Path to the FITS file.
    fitlog_file : str
        CASA imfit log file (for zero-level offset).
    fitlog_summary_file : str
        CASA imfit summary table.
    rms_noise : float
        RMS noise in Jy/beam.
    sigma_level : float, optional
        Radius = sigma_level × Gaussian sigma (default: 3).
    cmap : matplotlib colormap, optional
        Colormap for plotting.

    Returns
    -------
    flux : float
        Integrated flux in Jy.
    flux_err : float
        Flux uncertainty in Jy.
    """

    # --------------------------------------------------
    # Load image & fit results
    # --------------------------------------------------
    instance = Formation_Cluster(fits_file)

    df = pd.read_csv(
        fitlog_summary_file,
        index_col=False,
        header=0,
        delim_whitespace=True,
        skiprows=1,
    )
    fitlog = df.shift(axis=1)

    # --------------------------------------------------
    # Read zero-level offset from CASA imfit log
    # --------------------------------------------------
    offset_val = 0.0
    offset_err = 0.0

    with open(fitlog_file, "r") as f:
        for line in f:
            if "Zero level offset fit:" in line:
                match = re.search(
                    r"([-+]?\d+\.\d+e?[-+]?\d*)\s*\+/-\s*([-+]?\d+\.\d+e?[-+]?\d*)",
                    line,
                )
                if match:
                    offset_val = float(match.group(1))
                    offset_err = float(match.group(2))

    # 强制设置为0
    offset_val = 0.0

    # --------------------------------------------------
    # Source center & aperture radius
    # --------------------------------------------------
    ra_center = fitlog["LongICRS"][0]
    dec_center = fitlog["LatICRS"][0]
    print(ra_center, dec_center)

    x_cen, y_cen = instance.wcs.celestial.wcs_world2pix(
        ra_center, dec_center, 0
    )

    # geometric mean FWHM → Gaussian sigma → r
    fwhm_pix = (
        np.sqrt(fitlog["ConMaj"][0] * fitlog["ConMin"][0])
        / instance.PIXEL_SCALE.value
    )
    sigma_pix = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    r = sigma_level * sigma_pix
    print(r * instance_this.PIXEL_SCALE.value, "arcsec")

    # --------------------------------------------------
    # Photutils aperture photometry
    # --------------------------------------------------
    aperture = CircularAperture([(x_cen, y_cen)], r=r)

    # error map (Jy/beam per pixel)
    error_map = np.ones_like(instance.img) * rms_noise

    phot_table = aperture_photometry(
        instance.img - offset_val,
        aperture,
        error=error_map,
        method="exact",
    )

    flux = phot_table["aperture_sum"][0]
    flux_err = phot_table["aperture_sum_err"][0]

    # Jy/beam → Jy
    beam2pix = instance.sr_beam / instance.sr_pixel
    flux /= beam2pix
    flux_err /= np.sqrt(beam2pix)

    print(f"Total flux: {flux:.6e} Jy")
    print(f"Flux error: {flux_err:.6e} Jy")

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection=instance.wcs.celestial)

    norm = simple_norm(instance.img, "linear", percent=99.5)
    im = ax.imshow(instance.img, origin="lower", cmap=cmap, norm=norm)

    circ = Circle(
        (x_cen, y_cen),
        r,
        edgecolor="red",
        facecolor="none",
        lw=2,
    )
    ax.add_patch(circ)

    ax.plot(
        x_cen,
        y_cen,
        marker="+",
        color="yellow",
        markersize=10,
        mew=1.5,
    )

    plt.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="Flux [Jy/beam]",
    )

    ax.set_xlabel("RA (J2000)")
    ax.set_ylabel("Dec (J2000)")
    ax.set_title(
        f"{sigma_level}$\\sigma$ aperture photometry (photutils)",
        fontsize=20,
    )

    plt.show()

    return flux, flux_err

def sum_circle_with_plot_sedfluxer(
    fits_file,
    instance_original, # 整个大图的 Formation_Cluster 实例
    sigma_level=3,
    cmap=hue_sat_value2_cmap,
    show_plots=True,
):
    """
    Perform circular aperture photometry using SedFluxer and plot the aperture.

    Parameters
    ----------
    fits_file : str
        Path to the FITS file.
    fitlog_file : str
        CASA imfit log file (for zero-level offset, currently unused).
    fitlog_summary_file : str
        CASA imfit summary table.
    rms_noise : float
        RMS noise (kept for interface consistency, SedFluxer may not need it).
    sigma_level : float, optional
        Radius = sigma_level × Gaussian sigma (default: 3).
    cmap : matplotlib colormap, optional
        Colormap for plotting.
    show_plots : bool
        Whether to show aperture plot.

    Returns
    -------
    flux : float
        Integrated flux in Jy.
    flux_err : float
        Flux uncertainty in Jy.
    """

    basename = os.path.basename(fits_file).replace(".fits", "")
    dirname = os.path.dirname(fits_file)
    working_dir = os.path.join(dirname, basename)
    fitlog_summary_file=os.path.join(working_dir, "fit_summary_log.dat")

    # --------------------------------------------------
    # Load image & fit results
    # --------------------------------------------------
    instance = Formation_Cluster(fits_file)

    df = pd.read_csv(
        fitlog_summary_file,
        index_col=False,
        header=0,
        delim_whitespace=True,
        skiprows=1,
    )
    fitlog = df.shift(axis=1)

    # --------------------------------------------------
    # Source center (ICRS)
    # --------------------------------------------------
    ra_center = fitlog["LongICRS"][0]
    dec_center = fitlog["LatICRS"][0]

    central_coords = SkyCoord(
        ra=ra_center * u.deg,
        dec=dec_center * u.deg,
        frame="icrs",
    )

    # --------------------------------------------------
    # Aperture radius (same logic as before)
    # --------------------------------------------------
    fwhm_pix = (
        np.sqrt(fitlog["ConMaj"][0] * fitlog["ConMin"][0])
        / instance.PIXEL_SCALE.value
    )
    sigma_pix = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    # arcsec
    aper_rad = sigma_level * sigma_pix * instance.PIXEL_SCALE.value
    print(f"Aperture radius = {aper_rad:.3f} arcsec")

    # --------------------------------------------------
    # SedFluxer photometry  ⭐替换核心⭐
    # --------------------------------------------------
    fluxer = SedFluxer(instance_original.hdu[0])

    flux_obj = fluxer.get_flux(
        central_coords,
        aper_rad,          # source aperture
        aper_rad,          # inner annulus (same, if你之前也是这样用)
        aper_rad * 2.0,    # outer annulus
    )

    flux = flux_obj.flux_bkgsub
    flux_err = flux_obj.fluc_error

    print(f"Total flux: {flux:.6e} Jy")
    print(f"Flux error: {flux_err:.6e} Jy")

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    if show_plots:
        flux_obj.plot(cmap=cmap)

    return flux, flux_err

def perform_bg_subtracted_fit_simple(
    instance, 
    cutout, 
    save_path, 
    std_val, 
    box_initial,       # 必需：字符串格式 'xmin,ymin,xmax,ymax'
    box_final=None,    # 可选：去背景后拟合的box，如果不填则默认与initial一致
    show_plots=False, 
    cmap=None
):
    """
    极简版：背景扣除并重新拟合。
    不保存图片，不计算额外测光，直接指定 Box。
    """
    
    # 默认中心点估计 (取 cutout 中心，用于 imfit 初始搜寻)
    # 假设 cutout 是正方形或长方形
    cy, cx = np.array(cutout.data.shape) // 2
    
    # 如果没指定 final box，就用 initial box
    if box_final is None:
        box_final = box_initial

    # 1. 初始拟合 (Raw Fit)
    # -------------------------------------------------
    # 注意：这里去掉了 try-except 的 fallback 逻辑，假设 box 是准的
    log_raw = casa_imfit_manually(
        save_path,
        instance,
        manual_estimate=None,
        show_fitting_result=show_plots, # 这里控制 imfit 自带的打印/绘图
        zero_level=True,
        box_set=box_initial,
        show_one_dim_result=True,      # 简化显示
        idx=cy, idy=cx,                 # 默认以图片中心开始搜索
        RMS=std_val,
        # savepath=None,                # 不保存图片
        # fig_basename=None
    )

    # 2. 解析拟合结果 (用于制作 Mask)
    # -------------------------------------------------
    conmaj_sigma = log_raw['ConMaj'][0] / (2 * np.sqrt(2 * np.log(2))) / instance.PIXEL_SCALE.value
    conmin_sigma = log_raw['ConMin'][0] / (2 * np.sqrt(2 * np.log(2))) / instance.PIXEL_SCALE.value
    conPA = log_raw['ConPA'][0]
    ra_center = log_raw["LongICRS"][0]
    dec_center = log_raw["LatICRS"][0]

    # 3. 制作 Mask & 计算背景
    # -------------------------------------------------
    ra_pix, dec_pix = cutout.wcs.celestial.all_world2pix(ra_center, dec_center, 0)
    
    # Mask 半径：2倍 Sigma
    ap = EllipticalAperture((ra_pix, dec_pix), conmaj_sigma * 2, conmin_sigma * 2, np.radians(conPA+90))
    source_mask = ap.to_mask().to_image(cutout.data.shape).astype(bool)

    SigmaClip_set = SigmaClip(sigma=3.0, maxiters=None, stdfunc=mad_std)
    BG2d = Background2D(cutout.data, (5,5), mask=source_mask, sigma_clip=SigmaClip_set)
    bgmap = BG2d.background

    # 4. 绘图 (仅 Show，不 Save)
    # -------------------------------------------------
    if show_plots:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        vmax = np.nanmax(cutout.data)
        norm = LogNorm(vmin=1e-5, vmax=vmax if vmax > 0 else 1e-3)
        current_cmap = plt.cm.jet.copy() if cmap is None else cmap.copy()

        # 原图
        ax1.imshow(cutout.data, origin='lower', cmap=current_cmap, norm=norm)
        ax1.set_title("Original")
        
        # 背景 + Mask
        ax2.imshow(bgmap, origin='lower', cmap=current_cmap, norm=norm)
        ellipse_mask = Ellipse(xy=(ra_pix, dec_pix), width=conmaj_sigma * 4, height=conmin_sigma * 4,
                            angle=90 + conPA, edgecolor='black', facecolor='none', linewidth=2.0)   
        ax2.add_patch(ellipse_mask)
        ax2.set_title("Background model")

        # 扣除后
        ax3.imshow(cutout.data - bgmap, origin='lower', cmap=current_cmap, norm=norm)
        ax3.set_title("Subtracted")
        
        plt.tight_layout()
        plt.show() # 直接显示

    # 5. 保存去背景 FITS & 最终拟合
    # -------------------------------------------------
    replace_file_path = save_path.replace('.fits','_bgmap.fits')
    
    replace_fits_data(
        original_fits_path=save_path,
        new_data=cutout.data - bgmap,
        output_path=replace_file_path
    )
    
    log_final = casa_imfit_manually(
        replace_file_path,
        instance,
        manual_estimate=None,
        box_set=box_final,
        show_fitting_result=show_plots,
        show_one_dim_result=True, 
        idx=cy, idy=cx,
        RMS=std_val,
        # savepath=None,
        # fig_basename=None
    )
    
    return log_final,replace_file_path

# 存储进入表格函数
def save_to_sql(log, index, ID ,conn , table_name='Band6_TM1+TM2', sum_flux=None, sum_flux_err=None, image_type='normal'):
    # ID 是源的标识符，字符串，根据需要调整
    # 取出一行数据
    row = log.iloc[index]

    if sum_flux is not None and sum_flux_err is not None:
        data = {
        'Source ID': ID,  # 或其他ID规则
        'Total Flux': sum_flux,
        'Total Flux Error': sum_flux_err,
        'Peak Intensity': row['Peak'],
        'Peak Intensity Error': row['PeakErr'],
        'deconmajFWHM': 0,
        'deconmajFWHM Error': 0,
        'deconminFWHM': 0,
        'deconminFWHM Error': 0,
        'deconPA': 0,
        'deconPA Error': 0,
        'ra': row['LongICRS'],
        'ra Error': row['LongICRSerr'],
        'dec': row['LatICRS'],
        'dec Error': row['LatICRSerr'],
        'image': image_type,
        'sum_boolean': 1
    }
    # 构造一行的数据字典（字段名可根据你的表结构调整）
    else:
        data = {
            'Source ID': ID,  # 或其他ID规则
            'Total Flux': row['I'],
            'Total Flux Error': row['Ierr'],
            'Peak Intensity': row['Peak'],
            'Peak Intensity Error': row['PeakErr'],
            'deconmajFWHM': row['DeconMaj'],
            'deconmajFWHM Error': row['DeconMajErr'],
            'deconminFWHM': row['DeconMin'],
            'deconminFWHM Error': row['DeconMinErr'],
            'deconPA': row['DeconPA'],
            'deconPA Error': row['DeconPAErr'],
            'ra': row['LongICRS'],
            'ra Error': row['LongICRSerr'],
            'dec': row['LatICRS'],
            'dec Error': row['LatICRSerr'],
            'image': image_type,
            'sum_boolean': 0
        }

    # 创建一行的DataFrame
    df = pd.DataFrame([data])

    # 追加到数据库
    df.to_sql(table_name, conn, if_exists='append', index=False)

def save_to_sql_2(log, index, ID ,conn ,sum_flux, sum_flux_err,surrounding_mad_std, surrounding_complexity,
                   manual_fit_boolean, asymmetry_boolean, SNR, blending_bool, table_name='Band6_TM1+TM2', image_type='normal'):
    # ID 是源的标识符，字符串，根据需要调整
    # 取出一行数据
    row = log.iloc[index]
    data = {
        'Source ID': ID,  # 或其他ID规则
        'Total Flux': row['I'],
        'Total Flux Error': row['Ierr'],
        'Peak Intensity': row['Peak'],
        'Peak Intensity Error': row['PeakErr'],
        'deconmajFWHM': row['DeconMaj'],
        'deconmajFWHM Error': row['DeconMajErr'],
        'deconminFWHM': row['DeconMin'],
        'deconminFWHM Error': row['DeconMinErr'],
        'deconPA': row['DeconPA'],
        'deconPA Error': row['DeconPAErr'],
        'ra': row['LongICRS'],
        'ra Error': row['LongICRSerr'],
        'dec': row['LatICRS'],
        'dec Error': row['LatICRSerr'],
        'image': image_type,
        'Sum Flux': sum_flux,
        'Sum Flux Error': sum_flux_err,
        'Surrounding MAD Std': surrounding_mad_std,
        'Surrounding Complex Bool': surrounding_complexity,
        'Manual Fit Bool': manual_fit_boolean,
        'Asymmetry Bool': asymmetry_boolean,
        'SNR': SNR,
        'Blending Bool': blending_bool
 }
    
    # 创建一行的DataFrame
    df = pd.DataFrame([data])

    # 追加到数据库
    df.to_sql(table_name, conn, if_exists='append', index=False)


def delete_from_sql(conn, table_name='Band6_TM1+TM2', source_id=None):
    """
    删除表table_name中Source ID为source_id的记录。
    conn: sqlite3数据库连接
    table_name: 要操作的表名（支持特殊字符如+，自动加引号）
    source_id: 要删除的Source ID（字符串）
    """
    if source_id is None:
        raise ValueError("source_id must be specified.")

    # 构造SQL语句，表名加引号支持特殊字符
    sql = f'DELETE FROM "{table_name}" WHERE "Source ID" = ?;'
    cursor = conn.cursor()
    cursor.execute(sql, (source_id,))
    conn.commit()
    cursor.close()


# 辅助 batch processing 的函数
def boxes_overlap(b1, b2):
    x1min, x1max, y1min, y1max = b1
    x2min, x2max, y2min, y2max = b2

    return not (
        x1max < x2min or
        x2max < x1min or
        y1max < y2min or
        y2max < y1min
    )

def group_overlapping_boxes(coords, box_size):
    """
    coords: list of (x, y)
    box_size: int, n x n pixel box

    Returns
    -------
    isolated : list of indices
    groups   : list of lists of indices
    """

    h = box_size // 2
    n = len(coords)

    # 1. box 边界
    boxes = []
    for x, y in coords:
        boxes.append((x-h, x+h, y-h, y+h))

    # 2. 建图（邻接表）
    graph = {i: set() for i in range(n)}

    for i, j in combinations(range(n), 2):
        if boxes_overlap(boxes[i], boxes[j]):
            graph[i].add(j)
            graph[j].add(i)  # 在每个节点添加一个“边”

    # 3. 找连通分量
    visited = set()
    groups = []

    for i in range(n):
        if i in visited:
            continue

        stack = [i]
        component = set()

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            stack.extend(graph[node] - visited)

        if len(component) > 1:
            groups.append(sorted(component))

    # 4. isolated
    overlapped = set(sum(groups, []))
    isolated = [i for i in range(n) if i not in overlapped]

    return isolated, groups

def plot_box_groups(image, coords, box_size, isolated, groups):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(image, origin='lower', cmap='gray')

    h = box_size // 2

    # isolated: 绿色
    for i in isolated:
        x, y = coords[i]
        rect = patches.Rectangle(
            (x-h, y-h), box_size, box_size,
            linewidth=1.5, edgecolor='lime', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(x, y, f'{i}', color='lime')

    # overlap groups: 不同颜色
    colors = plt.cm.tab10.colors

    for gidx, group in enumerate(groups):
        color = colors[gidx % len(colors)]
        for i in group:
            x, y = coords[i]
            rect = patches.Rectangle(
                (x-h, y-h), box_size, box_size,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x, y, f'{i}', color=color)

    ax.set_title("Green: isolated | Colored: overlapping groups")
    plt.show()

def plot_group_source_id(cutout_fits_url, ra_array, dec_array, id_array):
    if isinstance(cutout_fits_url, str):
        aaa_cutout = Formation_Cluster(cutout_fits_url)
    else:
        aaa_cutout = cutout_fits_url
    ny, nx = aaa_cutout.img.shape
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'projection': aaa_cutout.wcs.celestial})
    norm = PowerNorm(0.5,0, np.nanmax(aaa_cutout.img))
    ax.imshow(aaa_cutout.img, origin='lower', cmap=hue_sat_value2_cmap, norm=norm)
    for idx, ra in enumerate(ra_array):
        dec = dec_array[idx]
        id_this = id_array[idx]
        ra_pix,dec_pix = aaa_cutout.wcs.celestial.all_world2pix(ra,dec,0)
        ax.plot(ra_pix, dec_pix, marker='+', markersize=10, markeredgecolor='black', markerfacecolor='black')
        ax.text(ra_pix + 2, dec_pix + 2, str(id_this), color='black', fontsize=12)
    ax.set_xlabel('R.A.')
    ax.set_ylabel('Dec.')
    ax.set_aspect('equal')
    plt.show()
    return fig



def create_cutout_from_coords(ra, dec, img_data, wcs,head, freq=None, 
                              cutout_size=(60, 60), 
                              std_val=None, 
                              bt_func=None, 
                              cmap='viridis',
                              show=True, 
                              save_path=None,
                              source_id="Unknown"):
    """
    根据给定的 RA, Dec 从图像中裁剪出小图，并选择性地展示或保存。

    Parameters
    ----------
    ra : float
        赤经 (Right Ascension)，单位通常为度。
    dec : float
        赤纬 (Declination)，单位通常为度。
    img_data : numpy.ndarray
        原始的大图数据 (2D array)。
    wcs : astropy.wcs.WCS
        原始图像的 WCS 对象。
    freq : float, optional
        观测频率，用于亮度温度转换。如果为 None，则不进行转换。
    cutout_size : tuple, optional
        切片大小 (height, width)，默认 (60, 60)。
    std_val : float, optional
        背景噪声标准差 (RMS)，用于设置 contour 的起始 level。
        如果为 None，将尝试自动估算或不画 contour。
    bt_func : function, optional
        将 Flux 转换为 Brightness Temperature 的函数。
        签名应类似 func(data, freq)。如果为 None，直接使用原始 Flux 数据。
    cmap : str or Colormap, optional
        绘图使用的 colormap。
    show : bool, optional
        是否显示绘图，默认为 True。
    save_path : str, optional
        如果提供路径，将裁剪后的 FITS 文件保存到该路径。默认为 None。
    source_id : str or int, optional
        源的 ID，用于标题和文件名（如果是自动生成文件名）。

    Returns
    -------
    cutout : astropy.nddata.Cutout2D
        裁剪后的 Cutout2D 对象。
    """
    
    # 1. 坐标转换 World -> Pixel
    xcen, ycen = wcs.all_world2pix(ra, dec, 0)
    position = (xcen, ycen)
    
    # 2. 执行裁剪
    # mode='partial' 允许裁剪区域超出原图边缘（用 NaN 填充）
    cutout = Cutout2D(img_data, position, cutout_size, mode='partial', wcs=wcs)
    
    # 3. 检查数据有效性 (如果全是 NaN 则跳过后续绘图/保存)
    if np.isnan(cutout.data).all():
        print(f"Warning: Cutout for Source {source_id} is all NaN. Skipping.")
        return cutout

    # 更新 Header (用于保存和绘图投影)
    # 注意：WCS 转 Header 通常包含 WCS 关键词
    cutout_header = head.copy() 
    cutout_header.update(cutout.wcs.to_header())

    # 4. 数据转换 (Flux -> Brightness Temperature)
    if bt_func is not None and freq is not None:
        plot_data = bt_func(cutout.data, freq)
        # 获取中心点的值作为参考最大值 (注意 cutout 坐标系中心约为 size/2)
        # 但为了稳妥，直接取 cutout 数据中的最大值
        data_max = np.nanmax(plot_data)
        # 同样需要转换中心点原始值或者 std
        # 假设 std_val 已经是转换后的或者是原始 flux 需要转换，这里假定 std_val 是最终单位
        # 如果 std_val 是 flux，需要在外面转好传入，或者在这里转
        # 为了简化，假设传入的 std_val 已经是匹配 plot_data 单位的
        plot_std = std_val if std_val is not None else np.nanstd(plot_data)
    else:
        plot_data = cutout.data
        data_max = np.nanmax(plot_data)
        plot_std = std_val if std_val is not None else np.nanstd(plot_data)

    # 5. 绘图 (show=True)
    if show:
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': cutout.wcs})
        
        # 添加文本标签
        ax.text(0.05, 0.95, f'Source ID: {source_id}', transform=ax.transAxes, fontsize=15,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 设置显示范围 (vmin, vmax)
        # 你的原逻辑：vmin = 3 * std
        vmin = 3 * plot_std
        vmax = data_max
        
        # 防止 vmin >= vmax 导致报错
        if vmin >= vmax:
            vmin = np.nanmin(plot_data)
        
        # 绘制图像
        norm1 = ImageNormalize(stretch=LogStretch(), vmin=vmin, vmax=vmax)
        im = ax.imshow(plot_data, norm=norm1, origin='lower', cmap=cmap)
        
        # 绘制等高线 (Contours)
        # 确保 log2 不会遇到负数或零
        if vmax > vmin and vmin > 0:
            highlim = np.log2(vmax / vmin)
            # 生成指数级增长的 levels: 3sigma, 6sigma, 12sigma...
            levels = vmin * 2**(np.arange(0, highlim + 1)) # +1 确保覆盖到最大值附近
            
            # 生成网格
            x = np.arange(plot_data.shape[1])
            y = np.arange(plot_data.shape[0])
            X, Y = np.meshgrid(x, y)
            
            ax.contour(X, Y, plot_data, levels=levels, colors='black', linewidths=1)
        
        # 添加 Beam (如果需要)
        # 这里需要你确保环境里有 add_beam 或者 wcsaxes 的支持
        try:
            effect = withStroke(linewidth=2, foreground='red')
            # 这是一个示例调用，具体取决于你的库版本
            # wcsaxes 通常集成在 ax 中，或者是单独的函数
            # ax.add_beam(pad=2, path_effects=[effect]) 
            pass 
        except Exception as e:
            # print(f"Beam plot failed: {e}")
            pass

        plt.colorbar(im, ax=ax, label='Brightness Temperature (K)' if bt_func else 'Flux')
        plt.show()

    # 6. 保存 FITS 文件 (save_path != None)
    if save_path is not None:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 创建 HDU
        # 注意：这里保存的是原始 cutout.data (Flux) 还是 plot_data (K) 取决于你的需求
        # 通常科学数据保存原始 Flux 比较好
        new_hdu = fits.PrimaryHDU(data=cutout.data, header=cutout_header)
        new_hdu.writeto(save_path, overwrite=True)
        print(f"Saved cutout to: {save_path}")

    return cutout

def replace_fits_data(original_fits_path, new_data, output_path=None):
    """
    使用原始 FITS 文件的 Header 信息，保存一个新的数据数组。
    
    Parameters
    ----------
    original_fits_path : str
        原始 FITS 文件的路径 (作为模板)。
    new_data : numpy.ndarray
        要写入的新数据。
    output_path : str, optional
        保存路径。如果为 None，则返回 HDUList 对象而不保存文件。
        
    Returns
    -------
    hdul : astropy.io.fits.HDUList or None
        如果 output_path 为 None，返回生成的 HDUList 对象；否则返回 None。
    """
    
    # 1. 读取原始 FITS 文件
    with fits.open(original_fits_path) as hdul:
        # 复制一份，防止修改原文件
        # deepcopy 确保 header 和 data 都是独立的
        # 但 fits.HDUList 没有 deepcopy 方法，通常用 copy() 或者重新构建 HDU
        
        # 获取原始 Header
        original_header = hdul[0].header.copy()
        
        # 2. 检查数据维度并更新 Header
        # 这是一个关键步骤：如果新数据的大小变了（比如 Cutout），
        # 必须更新 NAXIS 相关参数，否则这就是一个损坏的 FITS。
        
        # FITS 标准中，轴顺序是 (NAXISn, ..., NAXIS2, NAXIS1) -> (..., Y, X)
        # Numpy shape 是 (..., Y, X)
        # 所以对应关系是倒序的
        
        n_dim = new_data.ndim
        original_header['NAXIS'] = n_dim
        
        for i in range(n_dim):
            # Numpy shape 的最后一个元素对应 NAXIS1 (X轴)
            # Numpy shape 的倒数第二个元素对应 NAXIS2 (Y轴)
            axis_len = new_data.shape[-(i+1)]
            original_header[f'NAXIS{i+1}'] = axis_len
            
        # 3. 构建新的 Primary HDU
        new_hdu = fits.PrimaryHDU(data=new_data, header=original_header)
        new_hdul = fits.HDUList([new_hdu])
        
        # 4. 保存或返回
        if output_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            new_hdul.writeto(output_path, overwrite=True)
            print(f"File saved to: {output_path}")
            return None
        else:
            return new_hdul
        
def second_moment_pa(x, y, I):
    """
    Compute position angle (PA) from intensity-weighted second moments.

    Parameters
    ----------
    x, y : 1D arrays
        Pixel coordinates relative to center.
    I : 1D array
        Intensity values (must be >= 0).

    Returns
    -------
    pa_deg : float
        Position angle in degrees, measured CCW from +x axis.
    """

    I = np.clip(I, 0, None)
    norm = I.sum()
    if norm == 0:
        return np.nan

    xx = np.sum(I * x * x) / norm
    yy = np.sum(I * y * y) / norm
    xy = np.sum(I * x * y) / norm

    # PA from covariance matrix
    pa = 0.5 * np.arctan2(2 * xy, xx - yy)
    return np.degrees(pa)

def pa_stability_analysis_old(
    img,
    x0,
    y0,
    fwhm_pix,
    r_factors=(1.0, 1.5, 2.0, 2.5, 3.0),
    noise=None,
    plot=True
):
    """
    Analyze the stability of PA derived from second moments
    as a function of aperture radius.

    Parameters
    ----------
    img : 2D array
        Image data.
    x0, y0 : float
        Source center (pixel coordinates).
    fwhm_pix : float
        Reference FWHM in pixels.
    r_factors : tuple
        Radii in units of FWHM.
    noise : float or None
        Optional noise threshold; pixels below 3*noise are ignored.
    plot : bool
        Whether to plot PA vs radius.

    Returns
    -------
    result : dict
        Dictionary with radii, PA list, and PA scatter.
    """

    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]

    dx = xx - x0
    dy = yy - y0
    rr = np.hypot(dx, dy)

    pa_list = []
    radii = []

    for f in r_factors:
        r = f * fwhm_pix
        mask = rr <= r

        if noise is not None:
            mask &= img > 3 * noise  # 除了+=, &=也可以是吧，学到了

        if mask.sum() < 10:          # 参加计算的像素不能太少
            pa_list.append(np.nan)     
            radii.append(r)
            continue

        pa = second_moment_pa(
            dx[mask],
            dy[mask],
            img[mask]
        )

        pa_list.append(pa)
        radii.append(r)

    pa_arr = np.array(pa_list)
    valid = np.isfinite(pa_arr)

    # circular std (important!)
    pa_rad = np.radians(pa_arr[valid])
    pa_scatter = np.degrees(
        np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * pa_rad)))))
    )

    if plot:
        plt.figure(figsize=(6, 4))
        plt.plot(np.array(radii)[valid], pa_arr[valid], 'o-')
        plt.xlabel("Radius (pixel)")
        plt.ylabel("PA (deg)")
        plt.title(f"PA stability (scatter = {pa_scatter:.1f}°)")
        plt.grid(alpha=0.3)
        plt.show()

    return {
        "radii_pix": radii,
        "pa_deg": pa_list,
        "pa_scatter_deg": pa_scatter
    }

def pa_stability_analysis(
    img,
    x0,
    y0,
    fwhm_pix,
    r_factors=(1.0, 1.5, 2.0, 2.5, 3.0),
    noise=None,
    plot=True,
    cmap="gray"
):
    """
    Analyze the stability of PA derived from second moments
    as a function of aperture radius, with visual inspection.

    Parameters
    ----------
    img : 2D array
        Image data.
    x0, y0 : float
        Source center (pixel coordinates).
    fwhm_pix : float
        Reference FWHM in pixels.
    r_factors : tuple
        Radii in units of FWHM.
    noise : float or None
        Optional noise threshold; pixels below 3*noise are ignored.
    plot : bool
        Whether to plot diagnostic figures.
    cmap : str
        Colormap for imshow.

    Returns
    -------
    result : dict
        Dictionary with radii, PA list, and PA scatter.
    """

    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]

    dx = xx - x0
    dy = yy - y0
    rr = np.hypot(dx, dy)

    pa_list = []
    radii = []

    for f in r_factors:
        r = f * fwhm_pix
        mask = rr <= r

        if noise is not None:
            mask &= img > 3 * noise

        if mask.sum() < 10:
            pa_list.append(np.nan)
            radii.append(r)
            continue

        pa = second_moment_pa(
            dx[mask],
            dy[mask],
            img[mask]
        )

        pa_list.append(pa)
        radii.append(r)

    pa_arr = np.array(pa_list)
    valid = np.isfinite(pa_arr)

    # --- circular scatter (180 deg symmetry) ---
    pa_rad = np.deg2rad(pa_arr[valid])
    phi = 2 * pa_rad
    R = np.abs(np.mean(np.exp(1j * phi)))
    pa_scatter = 0.5 * np.sqrt(-2 * np.log(R))
    pa_scatter = np.rad2deg(pa_scatter)

    # ================= plotting =================
    if plot:
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(11, 4), constrained_layout=True
        )

        # --- left: image + apertures ---
        im = ax1.imshow(img, origin="lower", cmap=cmap)
        ax1.plot(x0, y0, "+", color="red", ms=10, mew=2)

        colors = plt.cm.viridis(np.linspace(0, 1, len(r_factors)))

        for r, c in zip(radii, colors):
            circ = Circle((x0, y0), r, ec=c, fc="none", lw=2)
            ax1.add_patch(circ)

        ax1.set_title("Aperture radii used for PA")
        ax1.set_xlabel("x [pix]")
        ax1.set_ylabel("y [pix]")
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

        # --- right: PA vs radius ---
        ax2.plot(
            np.array(radii)[valid],
            pa_arr[valid],
            "o-",
            color="k"
        )
        ax2.set_xlabel("Radius [pix]")
        ax2.set_ylabel("PA [deg]")
        ax2.set_title(f"PA stability (scatter = {pa_scatter:.1f}°)")
        ax2.grid(alpha=0.3)

        plt.show()

    return {
        "radii_pix": radii,
        "pa_deg": pa_list,
        "pa_scatter_deg": pa_scatter
    }

def calculate_asymmetry_index(
    fits_file,
    fitlog_file,
    fitlog_summary_file,
    rms_noise,
    sigma_level=3,
    cmap='viridis', # 默认改一下，如果没有hue_sat_value2_cmap可以用这个
):
    """
    Calculate the Asymmetry Index (A) for a source within a circular aperture.
    
    A = sum(|I - I_180|) / (2 * sum(|I|))  (within aperture)

    Parameters
    ----------
    fits_file : str
        Path to the FITS file.
    fitlog_file : str
        CASA imfit log file.
    fitlog_summary_file : str
        CASA imfit summary table.
    rms_noise : float
        RMS noise in Jy/beam (Used for visualization or potential background A correction).
    sigma_level : float, optional
        Radius = sigma_level × Gaussian sigma.
    cmap : matplotlib colormap, optional
        Colormap for plotting.

    Returns
    -------
    A_index : float
        The computed Asymmetry Index.
    """

    # --------------------------------------------------
    # 1. Load image & fit results (保持你原有的逻辑)
    # --------------------------------------------------
    # 假设 Formation_Cluster 是你定义好的类
    instance = Formation_Cluster(fits_file)

    df = pd.read_csv(
        fitlog_summary_file,
        index_col=False,
        header=0,
        delim_whitespace=True,
        skiprows=1,
    )
    fitlog = df.shift(axis=1)

    # --------------------------------------------------
    # 2. Read zero-level offset
    # --------------------------------------------------
    offset_val = 0.0
    with open(fitlog_file, "r") as f:
        for line in f:
            if "Zero level offset fit:" in line:
                match = re.search(
                    r"([-+]?\d+\.\d+e?[-+]?\d*)\s*\+/-\s*([-+]?\d+\.\d+e?[-+]?\d*)",
                    line,
                )
                if match:
                    offset_val = float(match.group(1))
    
    # 强制归零 (根据你的代码逻辑)
    offset_val = 0.0 

    # --------------------------------------------------
    # 3. Source center & aperture radius
    # --------------------------------------------------
    ra_center = fitlog["LongICRS"][0]
    dec_center = fitlog["LatICRS"][0]
    print(f"Center: RA={ra_center}, Dec={dec_center}")

    # 获取在原图中的像素坐标 (浮点数，亚像素精度)
    x_cen, y_cen = instance.wcs.celestial.wcs_world2pix(
        ra_center, dec_center, 0
    )

    # 计算半径
    fwhm_pix = (
        np.sqrt(fitlog["ConMaj"][0] * fitlog["ConMin"][0])
        / instance.PIXEL_SCALE.value
    )
    sigma_pix = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    r = sigma_level * sigma_pix
    print(f"Aperture Radius: {r:.2f} pixels")

    # --------------------------------------------------
    # 4. Asymmetry Calculation (核心部分)
    # --------------------------------------------------
    
    # 准备数据：减去 offset
    img_data = instance.img - offset_val
    
    # 创建坐标网格 (为了高效计算，我们只在一个切片范围内操作)
    # 截取一个稍微大一点的方形区域以节省内存，保证包含圆
    box_size = int(r * 2 + 5)
    x_min = max(0, int(x_cen) - box_size)
    x_max = min(img_data.shape[1], int(x_cen) + box_size)
    y_min = max(0, int(y_cen) - box_size)
    y_max = min(img_data.shape[0], int(y_cen) + box_size)
    
    # 提取切片数据
    cutout = img_data[y_min:y_max, x_min:x_max]
    
    # 在切片坐标系中的中心位置
    xc_cut = x_cen - x_min
    yc_cut = y_cen - y_min
    
    # 生成切片的像素坐标网格
    ny, nx = cutout.shape
    y_indices, x_indices = np.indices((ny, nx))
    
    # 1. 生成 Mask：距离中心小于 r 的区域
    dist_sq = (x_indices - xc_cut)**2 + (y_indices - yc_cut)**2
    mask = dist_sq <= r**2
    
    # 2. 生成旋转 180 度的图像 I_180
    # 旋转 180 度等价于坐标映射：(x', y') = (2*xc - x, 2*yc - y)
    # 使用 map_coordinates 进行插值，确保亚像素中心的旋转精度
    coords_rot_y = 2 * yc_cut - y_indices
    coords_rot_x = 2 * xc_cut - x_indices
    
    # 注意：map_coordinates 需要 (y, x) 顺序的坐标数组
    # order=1 (线性插值) 通常足够，order=3 (三次样条) 更平滑但可能引入负值伪影
    I_180 = map_coordinates(cutout, [coords_rot_y, coords_rot_x], order=1, mode='constant', cval=0.0)
    
    # 3. 计算 Asymmetry
    # 只计算 Mask 内部的像素
    I_orig_masked = cutout[mask]
    I_rot_masked = I_180[mask]
    
    # 残差绝对值
    residual = np.abs(I_orig_masked - I_rot_masked)
    
    # 公式：sum(|I - I_180|) / (2 * sum(|I|))
    # 分母乘 2 是为了归一化 (因为分子包含了正负两边的偏差)
    sum_abs_res = np.sum(residual)
    sum_abs_flux = np.sum(np.abs(I_orig_masked))
    
    A_index = sum_abs_res / (2 * sum_abs_flux)
    
    print(f"Computed Asymmetry Index (A): {A_index:.4f}")

    # --------------------------------------------------
    # 5. Plotting (可视化检查)
    # --------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 设置显示范围以便看清源
    vmin = -3 * rms_noise
    vmax = np.max(cutout)
    norm = simple_norm(cutout, "linear", min_cut=vmin, max_cut=vmax)
    
    # 原图切片
    axes[0].imshow(cutout, origin='lower', cmap=cmap, norm=norm)
    axes[0].set_title("Original Source")
    axes[0].plot(xc_cut, yc_cut, 'r+', markersize=10) # 标记旋转中心
    circ1 = Circle((xc_cut, yc_cut), r, edgecolor='red', facecolor='none')
    axes[0].add_patch(circ1)
    
    # 旋转图切片
    axes[1].imshow(I_180, origin='lower', cmap=cmap, norm=norm)
    axes[1].set_title("Rotated 180°")
    axes[1].plot(xc_cut, yc_cut, 'r+', markersize=10)
    circ2 = Circle((xc_cut, yc_cut), r, edgecolor='red', facecolor='none')
    axes[1].add_patch(circ2)
    
    # 残差图 (|I - I_180|)
    # 这里用 mask 遮挡外部以便观察
    res_map = np.abs(cutout - I_180)
    res_map[~mask] = 0 # 仅显示 mask 内的残差
    
    im3 = axes[2].imshow(res_map, origin='lower', cmap='inferno') # 用热力图看残差
    axes[2].set_title(f"Residual |I - I180|\nA = {A_index:.3f}")
    plt.colorbar(im3, ax=axes[2], label='Abs Residual')
    
    plt.tight_layout()
    plt.show()

    return A_index

def process_isolated_sources(
    isolate_sources,
    ra_array,
    dec_array,
    venn_code_array,
    instances,          # dict: {'normal': obj, 'rmb05': obj, 'allchan': obj}
    std_dict,           # dict: {'normal': std, 'rmb05': std, 'allchan': std}
    cutout_base_dir,
    r_factors=(1.0, 1.5, 2.0, 2.5, 3.0),
    pa_scatter_threshold=5.0,
    cutout_size=(100, 100),   # 切片大小尽量固定，因为后续的一些切片还是固定的(30,30,70,70)等
    cmap=None, # 对应原代码中的 hue_sat_value2_cmap
    show_plots=True,
    logger=None,
):
    # --- Initialization ---
    final_logs = []
    pa_scatter_array = []
    
    # 根据输入长度初始化数组
    sum_flux_array = np.zeros(len(isolate_sources))
    sum_flux_err_array = np.zeros(len(isolate_sources))
    sum_bool_array = np.zeros(len(isolate_sources))
    image_type = [''] * len(isolate_sources)

    # 使用 enumerate 获取列表索引(idx)用于填充数组，获取 i 用于索引源ID
    for idx, i in enumerate(tqdm(isolate_sources)):
        # if i > 5: break # 保留原有的调试断点逻辑，如不需要可删除
        
        ra = ra_array[i]
        dec = dec_array[i]
        venn_code = venn_code_array[i]
        
        # --- Instance Selection Logic ---
        # 对应原代码：Mapping venn_code to specific instances and stds
        if venn_code in [1, 4, 5, 7]:
            instance_key = 'normal' # 对应 aaa_18517
        elif venn_code in [2, 6]:
            instance_key = 'rmb05' # 对应 aaa_18517_rmb05
        else:
            instance_key = 'allchan' # 对应 aaa_18517_allchan
        
        image_type[idx] = instance_key
        
        instance_this = instances[instance_key]
        std_this = std_dict[instance_key]

        # --- Path Construction ---
        # 改动：使用 os.path.join 和 cutout_base_dir
        sub_dir_name = instance_this.filename.replace('.fits', '')
        output_dir = os.path.join(cutout_base_dir, sub_dir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        save_path_this = os.path.join(output_dir, f'cutout_iso_source{i+1}.fits')

        # --- Create Cutout ---
        cutout_test = create_cutout_from_coords(
            ra, dec, instance_this.img, instance_this.wcs.celestial, instance_this.head, 
            freq=instance_this.Freq,
            cutout_size=cutout_size, 
            std_val=std_this, 
            bt_func=instance_this.Brightness_Temperature, 
            cmap=cmap, 
            show=show_plots, 
            save_path=save_path_this, 
            source_id=i+1
        )

        std_surrounding = mad_std(cutout_test.data)
        
        # --- Logic Branch 1: High Background Noise ---
        if std_surrounding > std_this * 1.5:
            # raw fit
            log_raw = casa_imfit_manually(
                save_path_this,
                instance_this,
                manual_estimate=None,
                show_fitting_result=show_plots,
                zero_level=True,
                box_set='45,45,55,55',
                show_one_dim_result=show_plots, idx=50, idy=50,
                RMS=std_this
            )
            
            conmaj_sigma = log_raw['ConMaj'][0] / (2 * np.sqrt(2 * np.log(2))) / instance_this.PIXEL_SCALE.value
            conmin_sigma = log_raw['ConMin'][0] / (2 * np.sqrt(2 * np.log(2))) / instance_this.PIXEL_SCALE.value
            conPA = log_raw['ConPA'][0]
            ra_center = log_raw["LongICRS"][0]
            dec_center = log_raw["LatICRS"][0]
            
            ra_pix, dec_pix = cutout_test.wcs.celestial.all_world2pix(ra_center, dec_center, 0)
            ap = EllipticalAperture((ra_pix, dec_pix), conmaj_sigma * 2, conmin_sigma * 2, np.radians(conPA+90))
            
            source_mask = ap.to_mask().to_image(cutout_test.data.shape)
            source_mask = source_mask.astype(bool)
            
            SigmaClip_set = SigmaClip(sigma=3.0, maxiters=None, stdfunc=mad_std)
            BG2d = Background2D(cutout_test.data, (5,5),
                                mask=source_mask,
                                sigma_clip=SigmaClip_set)
            bgmap = BG2d.background
            
            # --- Plotting Background Subtraction ---
            # 只有当 show_plots 为 True 时才显示，但保存文件逻辑必须执行
            if show_plots:
                fig, axall = plt.subplots(1, 3, figsize=(18, 6))
                ax, ax2, ax3 = axall.flatten()

                vmax = cutout_test.data.max()
                norm = LogNorm(vmin=1e-7, vmax=vmax)
                
                # 使用传入的 cmap 并在函数内复制以避免修改全局对象
                current_cmap = plt.cm.jet.copy() if cmap is None else cmap.copy()
                current_cmap.set_under(current_cmap(0.0))
                current_cmap.set_over(current_cmap(1.0))

                im1 = ax.imshow(cutout_test.data, origin='lower', cmap=current_cmap, norm=norm)
                im2 = ax2.imshow(bgmap, origin='lower', cmap=current_cmap, norm=norm)

                ellipse_mask = Ellipse(xy=(ra_pix, dec_pix), width=conmaj_sigma * 4, height=conmin_sigma * 4,
                                    angle=90 + conPA, edgecolor='black', facecolor='none', linewidth=2.0)   
                ax2.add_patch(ellipse_mask)

                im3 = ax3.imshow(cutout_test.data - bgmap, origin='lower', cmap=current_cmap, norm=norm)
                ax.set_title(instance_this.filename, fontsize=8)

                cbar = fig.colorbar(im3, ax=axall, orientation='horizontal', fraction=0.1, pad=0.15, extend='both')
            
            replace_file_path = save_path_this.replace('.fits','_bgmap.fits')
            replace_fits_data(
                original_fits_path=save_path_this,
                new_data=cutout_test.data - bgmap,
                output_path=replace_file_path
            )
            
            # Refit on background subtracted image
            log = casa_imfit_manually(
                replace_file_path,
                instance_this,
                manual_estimate=None,
                show_fitting_result=show_plots,
                show_one_dim_result=show_plots, idx=50, idy=50,
                RMS=std_this,
            )
            
            final_logs.append(log)

        # --- Logic Branch 2: Normal Noise Levels ---
        else:
            log = casa_imfit_manually(
                save_path_this,
                instance_this,
                manual_estimate=None,
                show_fitting_result=show_plots,
                zero_level=True,
                box_set='30,30,70,70',
                show_one_dim_result=show_plots, idx=50, idy=50,
                RMS=std_this,
            )

            offset_val = 0.0
            offset_err = 0.0
            working_dir = save_path_this.replace('.fits','')
            fitlog_file = os.path.join(working_dir, "fit_log.dat")
            fitlog_summary_file = os.path.join(working_dir, "fit_summary_log.dat")

            # PA Stability Analysis
            instance_cut_this = Formation_Cluster(save_path_this)
            
            # 需要确保 pandas 读取没问题
            try:
                df = pd.read_csv(fitlog_summary_file, index_col=False, header=0, delim_whitespace=True, skiprows=1)
                fitlog_data = df.shift(axis=1) # 保持原代码逻辑
                ra_center_fit = fitlog_data["LongICRS"][0]
                dec_center_fit = fitlog_data["LatICRS"][0]
                
                x_cen, y_cen = instance_cut_this.wcs.celestial.wcs_world2pix(ra_center_fit, dec_center_fit, 0)
                
                fwhm_pix = (np.sqrt(fitlog_data["ConMaj"][0] * fitlog_data["ConMin"][0]) / instance_this.PIXEL_SCALE.value)
                sigma_pix = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
                
                dict_this = pa_stability_analysis(instance_cut_this.img, x_cen, y_cen, sigma_pix, noise=std_this, plot=show_plots,r_factors=r_factors)
                pa_scatter_this = dict_this['pa_scatter_deg']
                pa_scatter_array.append(pa_scatter_this)
                
                if logger: logger.info(f'PA scatter = {dict_this["pa_scatter_deg"]} degrees.')
                else: print('PA scatter = ', dict_this['pa_scatter_deg'], ' degrees.')
                
            except Exception as e:
                print(f"Error reading fit log or calculating PA scatter for source {i}: {e}")
                pa_scatter_this = 0.0 # Default fallback
                pa_scatter_array.append(pa_scatter_this)

            # --- Sub-branch 2a: Non-Gaussian (High Scatter) ---
            if pa_scatter_this > pa_scatter_threshold:
                central_coords = SkyCoord(ra=ra_center_fit*u.deg, dec=dec_center_fit*u.deg, frame='icrs')
                fluxer = SedFluxer(instance_this.hdu[0])
                aper_rad = 3 * sigma_pix * instance_this.PIXEL_SCALE.value # 3 sigma circle
                
                flux_obj = fluxer.get_flux(central_coords, aper_rad, aper_rad, aper_rad*2)
                if show_plots:
                    flux_obj.plot(cmap='jet') # 原代码写死 jet
                    
                sum_flux_array[idx] = flux_obj.flux_bkgsub # 注意使用 idx
                sum_flux_err_array[idx] = flux_obj.fluc_error
                sum_bool_array[idx] = 1
                final_logs.append(log)
            
            # --- Sub-branch 2b: Gaussian ---
            else:
                # Extract zero-level offset
                if os.path.exists(fitlog_file):
                    with open(fitlog_file, "r") as f:
                        for line in f:
                            if "Zero level offset fit:" in line:
                                match = re.search(r"([-+]?\d+\.\d+e?[-+]?\d*)\s*\+/-\s*([-+]?\d+\.\d+e?[-+]?\d*)", line)
                                if match:
                                    offset_val = float(match.group(1))
                                    offset_err = float(match.group(2))
                                    print("Zero offset =", offset_val, "Jy/beam")
                                    print("Error =", offset_err, "Jy/beam")
                
                # If negative offset, refit without zero level
                if offset_val < 0.0:
                    log = casa_imfit_manually(
                        save_path_this,
                        instance_this,
                        manual_estimate=None,
                        show_fitting_result=show_plots,
                        # zero_level=True, # 原代码注释掉了
                        box_set='30,30,70,70',
                        show_one_dim_result=show_plots, idx=50, idy=50,
                        RMS=std_this,
                    )
                
                final_logs.append(log)
        
        plt.close('all')

    return final_logs, sum_flux_array, sum_flux_err_array, sum_bool_array, image_type


def process_isolated_sources_2_old(  # 这一版对于环境不复杂的源都统一进行sum，并且与imfit的结果独立保存，避免了不对称/PA判断引入的复杂性
    isolate_sources,
    ra_array,
    dec_array,
    venn_code_array,
    instances,          # dict: {'normal': obj, 'rmb05': obj, 'allchan': obj}
    std_dict,           # dict: {'normal': std, 'rmb05': std, 'allchan': std}
    cutout_base_dir,
    cutout_size=(100, 100),   # 切片大小尽量固定，因为后续的一些切片还是固定的(30,30,70,70)等
    cmap=None, # 对应原代码中的 hue_sat_value2_cmap
    show_plots=True,
    logger=None,
):
    # --- Initialization ---
    final_logs = []
    pa_scatter_array = []
    
    # 根据输入长度初始化数组
    sum_flux_array = np.zeros(len(isolate_sources))
    sum_flux_err_array = np.zeros(len(isolate_sources))
    sum_bool_array = np.zeros(len(isolate_sources))
    image_type = [''] * len(isolate_sources)

    # 使用 enumerate 获取列表索引(idx)用于填充数组，获取 i 用于索引源ID
    for idx, i in enumerate(tqdm(isolate_sources)):
        # if i > 5: break # 保留原有的调试断点逻辑，如不需要可删除
        
        ra = ra_array[i]
        dec = dec_array[i]
        venn_code = venn_code_array[i]
        
        # --- Instance Selection Logic ---
        # 对应原代码：Mapping venn_code to specific instances and stds
        if venn_code in [1, 4, 5, 7]:
            instance_key = 'normal' # 对应 aaa_18517
        elif venn_code in [2, 6]:
            instance_key = 'rmb05' # 对应 aaa_18517_rmb05
        else:
            instance_key = 'allchan' # 对应 aaa_18517_allchan
        
        image_type[idx] = instance_key
        
        instance_this = instances[instance_key]
        std_this = std_dict[instance_key]

        # --- Path Construction ---
        # 改动：使用 os.path.join 和 cutout_base_dir
        sub_dir_name = instance_this.filename.replace('.fits', '')
        output_dir = os.path.join(cutout_base_dir, sub_dir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        save_path_this = os.path.join(output_dir, f'cutout_iso_source{i+1}.fits')

        # --- Create Cutout ---
        cutout_test = create_cutout_from_coords(
            ra, dec, instance_this.img, instance_this.wcs.celestial, instance_this.head, 
            freq=instance_this.Freq,
            cutout_size=cutout_size, 
            std_val=std_this, 
            bt_func=instance_this.Brightness_Temperature, 
            cmap=cmap, 
            show=show_plots, 
            save_path=save_path_this, 
            source_id=i+1
        )

        std_surrounding = mad_std(cutout_test.data)
        
        # --- Logic Branch 1: High Background Noise ---
        if std_surrounding > std_this * 1.5:
            # raw fit
            log_raw = casa_imfit_manually(
                save_path_this,
                instance_this,
                manual_estimate=None,
                show_fitting_result=show_plots,
                zero_level=True,
                box_set='45,45,55,55',
                show_one_dim_result=show_plots, idx=50, idy=50,
                RMS=std_this
            )
            
            conmaj_sigma = log_raw['ConMaj'][0] / (2 * np.sqrt(2 * np.log(2))) / instance_this.PIXEL_SCALE.value
            conmin_sigma = log_raw['ConMin'][0] / (2 * np.sqrt(2 * np.log(2))) / instance_this.PIXEL_SCALE.value
            conPA = log_raw['ConPA'][0]
            ra_center = log_raw["LongICRS"][0]
            dec_center = log_raw["LatICRS"][0]
            
            ra_pix, dec_pix = cutout_test.wcs.celestial.all_world2pix(ra_center, dec_center, 0)
            ap = EllipticalAperture((ra_pix, dec_pix), conmaj_sigma * 2, conmin_sigma * 2, np.radians(conPA+90))
            
            source_mask = ap.to_mask().to_image(cutout_test.data.shape)
            source_mask = source_mask.astype(bool)
            
            SigmaClip_set = SigmaClip(sigma=3.0, maxiters=None, stdfunc=mad_std)
            BG2d = Background2D(cutout_test.data, (5,5),
                                mask=source_mask,
                                sigma_clip=SigmaClip_set)
            bgmap = BG2d.background
            
            # --- Plotting Background Subtraction ---
            # 只有当 show_plots 为 True 时才显示，但保存文件逻辑必须执行
            if show_plots:
                fig, axall = plt.subplots(1, 3, figsize=(18, 6))
                ax, ax2, ax3 = axall.flatten()

                vmax = cutout_test.data.max()
                norm = LogNorm(vmin=1e-7, vmax=vmax)
                
                # 使用传入的 cmap 并在函数内复制以避免修改全局对象
                current_cmap = plt.cm.jet.copy() if cmap is None else cmap.copy()
                current_cmap.set_under(current_cmap(0.0))
                current_cmap.set_over(current_cmap(1.0))

                im1 = ax.imshow(cutout_test.data, origin='lower', cmap=current_cmap, norm=norm)
                im2 = ax2.imshow(bgmap, origin='lower', cmap=current_cmap, norm=norm)

                ellipse_mask = Ellipse(xy=(ra_pix, dec_pix), width=conmaj_sigma * 4, height=conmin_sigma * 4,
                                    angle=90 + conPA, edgecolor='black', facecolor='none', linewidth=2.0)   
                ax2.add_patch(ellipse_mask)

                im3 = ax3.imshow(cutout_test.data - bgmap, origin='lower', cmap=current_cmap, norm=norm)
                ax.set_title(instance_this.filename, fontsize=8)

                cbar = fig.colorbar(im3, ax=axall, orientation='horizontal', fraction=0.1, pad=0.15, extend='both')
            
            replace_file_path = save_path_this.replace('.fits','_bgmap.fits')
            replace_fits_data(
                original_fits_path=save_path_this,
                new_data=cutout_test.data - bgmap,
                output_path=replace_file_path
            )
            
            # Refit on background subtracted image
            log = casa_imfit_manually(
                replace_file_path,
                instance_this,
                manual_estimate=None,
                show_fitting_result=show_plots,
                show_one_dim_result=show_plots, idx=50, idy=50,
                RMS=std_this,
            )
            
            final_logs.append(log)

        # --- Logic Branch 2: Normal Noise Levels ---
        else:
            log = casa_imfit_manually(
                save_path_this,
                instance_this,
                manual_estimate=None,
                show_fitting_result=show_plots,
                zero_level=True,
                box_set='30,30,70,70',
                show_one_dim_result=show_plots, idx=50, idy=50,
                RMS=std_this,
            )

            offset_val = 0.0
            offset_err = 0.0
            working_dir = save_path_this.replace('.fits','')
            fitlog_file = os.path.join(working_dir, "fit_log.dat")
            fitlog_summary_file = os.path.join(working_dir, "fit_summary_log.dat")

            df = pd.read_csv(fitlog_summary_file, index_col=False, header=0, delim_whitespace=True, skiprows=1)
            fitlog_data = df.shift(axis=1) # 保持原代码逻辑
            ra_center_fit = fitlog_data["LongICRS"][0]
            dec_center_fit = fitlog_data["LatICRS"][0]
                        
            fwhm_pix = (np.sqrt(fitlog_data["ConMaj"][0] * fitlog_data["ConMin"][0]) / instance_this.PIXEL_SCALE.value)
            sigma_pix = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))

            # --- Sub-branch 2a: Non-Gaussian (High Scatter) ---
            central_coords = SkyCoord(ra=ra_center_fit*u.deg, dec=dec_center_fit*u.deg, frame='icrs')
            fluxer = SedFluxer(instance_this.hdu[0])
            aper_rad = 3 * sigma_pix * instance_this.PIXEL_SCALE.value # 3 sigma circle
            
            flux_obj = fluxer.get_flux(central_coords, aper_rad, aper_rad, aper_rad*2)
            if show_plots:
                flux_obj.plot(cmap='jet') # 原代码写死 jet
                
            sum_flux_array[idx] = flux_obj.flux_bkgsub # 注意使用 idx
            sum_flux_err_array[idx] = flux_obj.fluc_error
            sum_bool_array[idx] = 1

            # Extract zero-level offset
            if os.path.exists(fitlog_file):
                with open(fitlog_file, "r") as f:
                    for line in f:
                        if "Zero level offset fit:" in line:
                            match = re.search(r"([-+]?\d+\.\d+e?[-+]?\d*)\s*\+/-\s*([-+]?\d+\.\d+e?[-+]?\d*)", line)
                            if match:
                                offset_val = float(match.group(1))
                                offset_err = float(match.group(2))
                                print("Zero offset =", offset_val, "Jy/beam")
                                print("Error =", offset_err, "Jy/beam")
            
            # If negative offset, refit without zero level
            if offset_val < 0.0:
                log = casa_imfit_manually(
                    save_path_this,
                    instance_this,
                    manual_estimate=None,
                    show_fitting_result=show_plots,
                    # zero_level=True, # 原代码注释掉了
                    box_set='30,30,70,70',
                    show_one_dim_result=show_plots, idx=50, idy=50,
                    RMS=std_this,
                )
            
            final_logs.append(log)
        
        plt.close('all')

    return final_logs, sum_flux_array, sum_flux_err_array, sum_bool_array, image_type


def process_isolated_sources_2(  # 这一版对于环境不复杂的源都统一进行sum，并且与imfit的结果独立保存，避免了不对称/PA判断引入的复杂性
    isolate_sources,
    ra_array,
    dec_array,
    venn_code_array,
    instances,          # dict: {'normal': obj, 'rmb05': obj, 'allchan': obj}
    std_dict,           # dict: {'normal': std, 'rmb05': std, 'allchan': std}
    cutout_base_dir,
    cutout_size=(100, 100),   # 切片大小尽量固定，因为后续的一些切片还是固定的(30,30,70,70)等
    cmap=None, # 对应原代码中的 hue_sat_value2_cmap
    show_plots=True,
    logger=None,
):
    # --- Initialization ---
    final_logs = []
    pa_scatter_array = []
    
    # 根据输入长度初始化数组
    sum_flux_array = np.zeros(len(isolate_sources))
    sum_flux_err_array = np.zeros(len(isolate_sources))
    # sum_bool_array = np.zeros(len(isolate_sources))
    surrounding_mad_std_array = np.zeros(len(isolate_sources))
    surrounding_complex_bool_array = np.zeros(len(isolate_sources))   # 如果背景不够干净则标记为1，否则为0
    image_type = [''] * len(isolate_sources)

    # 使用 enumerate 获取列表索引(idx)用于填充数组，获取 i 用于索引源ID
    for idx, i in enumerate(tqdm(isolate_sources)):
        # if i > 5: break # 保留原有的调试断点逻辑，如不需要可删除
        
        ra = ra_array[i]
        dec = dec_array[i]
        venn_code = venn_code_array[i]
        
        # --- Instance Selection Logic ---
        # 对应原代码：Mapping venn_code to specific instances and stds
        if venn_code in [1, 4, 5, 7]:
            instance_key = 'normal' # 对应 aaa_18517
        elif venn_code in [2, 6]:
            instance_key = 'rmb05' # 对应 aaa_18517_rmb05
        else:
            instance_key = 'allchan' # 对应 aaa_18517_allchan
        
        image_type[idx] = instance_key
        
        instance_this = instances[instance_key]
        std_this = std_dict[instance_key]

        # --- Path Construction ---
        # 改动：使用 os.path.join 和 cutout_base_dir
        sub_dir_name = instance_this.filename.replace('.fits', '')
        output_dir = os.path.join(cutout_base_dir, sub_dir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        save_path_this = os.path.join(output_dir, f'cutout_iso_source{i+1}.fits')

        # --- Create Cutout ---
        cutout_test = create_cutout_from_coords(
            ra, dec, instance_this.img, instance_this.wcs.celestial, instance_this.head, 
            freq=instance_this.Freq,
            cutout_size=cutout_size, 
            std_val=std_this, 
            bt_func=instance_this.Brightness_Temperature, 
            cmap=cmap, 
            show=show_plots, 
            save_path=save_path_this, 
            source_id=i+1
        )

        std_surrounding = mad_std(cutout_test.data)
        
        # --- Logic Branch 1: High Background Noise ---
        if std_surrounding > std_this * 1.5:
            # raw fit
            log_raw = casa_imfit_manually(
                save_path_this,
                instance_this,
                manual_estimate=None,
                show_fitting_result=show_plots,
                zero_level=True,
                box_set='45,45,55,55',
                show_one_dim_result=show_plots, idx=50, idy=50,
                RMS=std_this
            )
            
            conmaj_sigma = log_raw['ConMaj'][0] / (2 * np.sqrt(2 * np.log(2))) / instance_this.PIXEL_SCALE.value
            conmin_sigma = log_raw['ConMin'][0] / (2 * np.sqrt(2 * np.log(2))) / instance_this.PIXEL_SCALE.value
            conPA = log_raw['ConPA'][0]
            ra_center = log_raw["LongICRS"][0]
            dec_center = log_raw["LatICRS"][0]
            fwhm_pix = (np.sqrt(log_raw["ConMaj"][0] * log_raw["ConMin"][0]) / instance_this.PIXEL_SCALE.value)
            sigma_pix = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))

            # 先不去背景用sedfluxer估计大小
            central_coords = SkyCoord(ra=ra_center*u.deg, dec=dec_center*u.deg, frame='icrs')
            fluxer = SedFluxer(instance_this.hdu[0])
            aper_rad = 3 * sigma_pix * instance_this.PIXEL_SCALE.value # 3 sigma circle
            
            flux_obj = fluxer.get_flux(central_coords, aper_rad, aper_rad, aper_rad*2)
            if show_plots:
                flux_obj.plot(cmap='jet') # 原代码写死 jet
                
            sum_flux_array[idx] = flux_obj.flux_bkgsub # 注意使用 idx
            sum_flux_err_array[idx] = flux_obj.fluc_error

            
            ra_pix, dec_pix = cutout_test.wcs.celestial.all_world2pix(ra_center, dec_center, 0)
            ap = EllipticalAperture((ra_pix, dec_pix), conmaj_sigma * 2, conmin_sigma * 2, np.radians(conPA+90))
            
            source_mask = ap.to_mask().to_image(cutout_test.data.shape)
            source_mask = source_mask.astype(bool)
            
            SigmaClip_set = SigmaClip(sigma=3.0, maxiters=None, stdfunc=mad_std)
            BG2d = Background2D(cutout_test.data, (5,5),
                                mask=source_mask,
                                sigma_clip=SigmaClip_set)
            bgmap = BG2d.background
            
            # --- Plotting Background Subtraction ---
            # 只有当 show_plots 为 True 时才显示，但保存文件逻辑必须执行
            if show_plots:
                fig, axall = plt.subplots(1, 3, figsize=(18, 6))
                ax, ax2, ax3 = axall.flatten()

                vmax = cutout_test.data.max()
                norm = LogNorm(vmin=1e-7, vmax=vmax)
                
                # 使用传入的 cmap 并在函数内复制以避免修改全局对象
                current_cmap = plt.cm.jet.copy() if cmap is None else cmap.copy()
                current_cmap.set_under(current_cmap(0.0))
                current_cmap.set_over(current_cmap(1.0))

                im1 = ax.imshow(cutout_test.data, origin='lower', cmap=current_cmap, norm=norm)
                im2 = ax2.imshow(bgmap, origin='lower', cmap=current_cmap, norm=norm)

                ellipse_mask = Ellipse(xy=(ra_pix, dec_pix), width=conmaj_sigma * 4, height=conmin_sigma * 4,
                                    angle=90 + conPA, edgecolor='black', facecolor='none', linewidth=2.0)   
                ax2.add_patch(ellipse_mask)

                im3 = ax3.imshow(cutout_test.data - bgmap, origin='lower', cmap=current_cmap, norm=norm)
                ax.set_title(instance_this.filename, fontsize=8)

                cbar = fig.colorbar(im3, ax=axall, orientation='horizontal', fraction=0.1, pad=0.15, extend='both')
            
            replace_file_path = save_path_this.replace('.fits','_bgmap.fits')
            replace_fits_data(
                original_fits_path=save_path_this,
                new_data=cutout_test.data - bgmap,
                output_path=replace_file_path
            )
            
            # Refit on background subtracted image
            log = casa_imfit_manually(
                replace_file_path,
                instance_this,
                manual_estimate=None,
                show_fitting_result=show_plots,
                show_one_dim_result=show_plots, idx=50, idy=50,
                RMS=std_this,
            )

            surrounding_mad_std_array[idx] = std_surrounding
            surrounding_complex_bool_array[idx] = 1   # 标记为背景复杂
            final_logs.append(log)

        # --- Logic Branch 2: Normal Noise Levels ---
        else:
            log = casa_imfit_manually(
                save_path_this,
                instance_this,
                manual_estimate=None,
                show_fitting_result=show_plots,
                zero_level=True,
                box_set='30,30,70,70',
                show_one_dim_result=show_plots, idx=50, idy=50,
                RMS=std_this,
            )

            offset_val = 0.0
            offset_err = 0.0
            working_dir = save_path_this.replace('.fits','')
            fitlog_file = os.path.join(working_dir, "fit_log.dat")
            fitlog_summary_file = os.path.join(working_dir, "fit_summary_log.dat")

            df = pd.read_csv(fitlog_summary_file, index_col=False, header=0, delim_whitespace=True, skiprows=1)
            fitlog_data = df.shift(axis=1) # 保持原代码逻辑
            ra_center_fit = fitlog_data["LongICRS"][0]
            dec_center_fit = fitlog_data["LatICRS"][0]
                        
            fwhm_pix = (np.sqrt(fitlog_data["ConMaj"][0] * fitlog_data["ConMin"][0]) / instance_this.PIXEL_SCALE.value)
            sigma_pix = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))

            # --- Sub-branch 2a: Non-Gaussian (High Scatter) ---
            central_coords = SkyCoord(ra=ra_center_fit*u.deg, dec=dec_center_fit*u.deg, frame='icrs')
            fluxer = SedFluxer(instance_this.hdu[0])
            aper_rad = 3 * sigma_pix * instance_this.PIXEL_SCALE.value # 3 sigma circle
            
            flux_obj = fluxer.get_flux(central_coords, aper_rad, aper_rad, aper_rad*2)
            if show_plots:
                flux_obj.plot(cmap='jet') # 原代码写死 jet
                
            sum_flux_array[idx] = flux_obj.flux_bkgsub # 注意使用 idx
            sum_flux_err_array[idx] = flux_obj.fluc_error
            # sum_bool_array[idx] = 1
            surrounding_mad_std_array[idx] = std_surrounding

            # Extract zero-level offset
            if os.path.exists(fitlog_file):
                with open(fitlog_file, "r") as f:
                    for line in f:
                        if "Zero level offset fit:" in line:
                            match = re.search(r"([-+]?\d+\.\d+e?[-+]?\d*)\s*\+/-\s*([-+]?\d+\.\d+e?[-+]?\d*)", line)
                            if match:
                                offset_val = float(match.group(1))
                                offset_err = float(match.group(2))
                                print("Zero offset =", offset_val, "Jy/beam")
                                print("Error =", offset_err, "Jy/beam")
            
            # If negative offset, refit without zero level
            if offset_val < 0.0:
                log = casa_imfit_manually(
                    save_path_this,
                    instance_this,
                    manual_estimate=None,
                    show_fitting_result=show_plots,
                    # zero_level=True, # 原代码注释掉了
                    box_set='30,30,70,70',
                    show_one_dim_result=show_plots, idx=50, idy=50,
                    RMS=std_this,
                )
            
            final_logs.append(log)
        
        plt.close('all')

    return final_logs, sum_flux_array, sum_flux_err_array, surrounding_mad_std_array, surrounding_complex_bool_array, image_type


# though useless, here is an example call
# calculate_asymmetry_index('/home/esker7293/Cluster_formation/Quarks_ipynbs/cutout_ps_dir/I18517+0437.Band6.cycle11.TM1+TM2.contin.allchan.combselfcal.image.tt0/cutout_iso_source3.fits',
#                         '/home/esker7293/Cluster_formation/Quarks_ipynbs/cutout_ps_dir/I18517+0437.Band6.cycle11.TM1+TM2.contin.allchan.combselfcal.image.tt0/cutout_iso_source3/fit_log.dat',
#                         '/home/esker7293/Cluster_formation/Quarks_ipynbs/cutout_ps_dir/I18517+0437.Band6.cycle11.TM1+TM2.contin.allchan.combselfcal.image.tt0/cutout_iso_source3/fit_summary_log.dat',
#                         rms_noise=std_allchan,
#                         sigma_level=3,
#                         cmap=hue_sat_value2_cmap
#                         )

def ensure_table_exists(conn, table_name):
    """
    Ensure SQL table exists. If not, create it with predefined schema.
    """

    query = f"""
    SELECT name FROM sqlite_master
    WHERE type='table' AND name='{table_name}';
    """
    exists = pd.read_sql(query, conn)

    if not exists.empty:
        return  # table already exists

    # Define empty table schema
    columns = {
        'Source ID': 'TEXT',
        'Total Flux': 'REAL',
        'Total Flux Error': 'REAL',
        'Peak Intensity': 'REAL',
        'Peak Intensity Error': 'REAL',
        'deconmajFWHM': 'REAL',
        'deconmajFWHM Error': 'REAL',
        'deconminFWHM': 'REAL',
        'deconminFWHM Error': 'REAL',
        'deconPA': 'REAL',
        'deconPA Error': 'REAL',
        'ra': 'REAL',
        'ra Error': 'REAL',
        'dec': 'REAL',
        'dec Error': 'REAL',
        'image': 'TEXT',
        # 'sum_boolean': 'INTEGER',
        'Sum Flux': 'REAL',
        'Sum Flux Error': 'REAL',
        'Surrounding MAD Std': 'REAL',
        'Surrounding Complex Bool': 'INTEGER',
        'Manual Fit Bool': 'INTEGER',
        'Asymmetry Bool': 'INTEGER',
        'SNR': 'REAL',
        'Blending Bool': 'INTEGER',
    }

    df_empty = pd.DataFrame({k: [] for k in columns})
    df_empty.to_sql(table_name, conn, if_exists='replace', index=False)

    print(f"Table '{table_name}' created.")

def delete_final_sources(conn, table_name = 'final_source_catalogue'):
    """
    Delete all records from table `venn_sources` safely.
    
    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    """
    
    # 1. 检查表是否存在
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", 
        (table_name,)
    )
    if cursor.fetchone() is None:
        print(f"Table '{table_name}' does not exist. No deletion needed.")
        return

    # 2. 执行清空操作
    try:
        # TRUNCATE TABLE 在 SQLite 中不支持，标准做法是 DELETE FROM
        cursor = conn.execute(f'DELETE FROM {table_name}')
        deleted_count = cursor.rowcount
        conn.commit()
        
        print(f"Table '{table_name}' cleared. Deleted {deleted_count} rows.")

    except sqlite3.Error as e:
        conn.rollback() # 发生错误时回滚，保证数据库安全
        print(f"An error occurred while clearing '{table_name}': {e}")
        raise

def drop_final_sources(conn, table_name='final_source_catalogue'):
    """
    Completely remove the table (data + columns + schema).
    """

    try:
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()
        print(f"Table '{table_name}' has been dropped.")

    except sqlite3.Error as e:
        conn.rollback()
        print(f"An error occurred while dropping '{table_name}': {e}")
        raise


