import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import yaml
from uncertainties import unumpy
from scipy.cluster.hierarchy import fclusterdata, linkage
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.constants as cons
from .Formation_Cluster import Formation_Cluster
import os
import sys

pc2au = 3600 * 180 / np.pi 

def fetch_table_as_arrays(conn, table_name):
    """
    读取 SQLite 表中的所有数据，并将每一列转换为 numpy array。
    
    Args:
        conn (sqlite3.Connection): 数据库连接对象。
        table_name (str): 表名。
        
    Returns:
        dict: 一个字典，Key 是列名，Value 是对应的 np.array。
    """
    cursor = conn.cursor()
    
    # 使用 f-string 注入表名，加双引号是为了防止表名中有特殊字符或空格
    # 注意：表名无法使用 ? 占位符，只能这样拼接
    try:
        cursor.execute(f'SELECT * FROM "{table_name}"')
    except sqlite3.OperationalError as e:
        print(f"Error reading table '{table_name}': {e}")
        return {}

    # 1. 获取所有数据
    rows = cursor.fetchall()
    
    # 2. 获取列名 (从 cursor.description 中提取)
    # description 格式为 ((name, type_code, ...), ...)
    if cursor.description:
        col_names = [desc[0] for desc in cursor.description]
    else:
        return {} # 表可能不存在或出错

    # 3. 处理空表情况
    if not rows:
        # 如果表是空的，返回空数组，但也保留列名结构
        return {name: np.array([]) for name in col_names}

    # 4. 核心步骤：转置数据 (Row-based -> Column-based)
    # zip(*rows) 会把 [(r1c1, r1c2), (r2c1, r2c2)] 变成 [(r1c1, r2c1), (r1c2, r2c2)]
    cols_data = list(zip(*rows))

    # 5. 组装成字典并转为 numpy array
    result_dict = {}
    for i, name in enumerate(col_names):
        # 自动推断类型转换 (int, float, string)
        result_dict[name] = np.array(cols_data[i])

    return result_dict

def load_pbcor(config_file):
    with open(config_file, "r") as f:
        cfg = yaml.safe_load(f)

    # =========================
    # Stage 0: Init
    # =========================
    name = cfg["source"]["name"]
    distance = cfg["source"]["distance_pc"]
    normal_fm = cfg["input"]["robust_p05"].replace(".fits",".pbcor.fits")
    fc_pbcor = Formation_Cluster(
        normal_fm,
        distance=distance
    )
    return fc_pbcor

def sigma_wilson(N_sys,value,z=1):
    aaa = 1 / (1 + z**2 / N_sys)
    bbb1 = + z * np.sqrt(value * (1 - value) / N_sys + z**2 / (4 * N_sys**2))
    bbb2 = - z * np.sqrt(value * (1 - value) / N_sys + z**2 / (4 * N_sys**2))
    sigma1 = aaa * (value + z**2 / (2 * N_sys) + bbb1)
    sigma2 = aaa * (value + z**2 / (2 * N_sys) + bbb2)
    return sigma1, sigma2

def calculate_wilson_interval(mf, n_sys, z=1):
    """
    计算威尔逊得分区间 (Wilson Score Interval)。
    
    公式:
    sigma = 1 / (1 + z^2 / n) * ( p + z^2 / (2n) ± z * sqrt( p(1-p)/n + z^2 / (4n^2) ) )

    参数:
    mf (float): 观测到的比例或频率 (Mean Frequency/Fraction), 范围 [0, 1]。
    n_sys (int or float): 样本大小 (Number of systems/observations)。
    z (float): 对应置信水平的 Z 分数。默认为 1.96 (对应约 95% 置信度)。
    
    返回:
    tuple: (lower_bound, upper_bound) 置信区间的下界和上界。
    """
    
    # 预先计算分母，以简化表达式
    denominator = 1 + (z**2) / n_sys
    
    # 计算公式中的中���点调整项 (MF + z^2/2N)
    center_adjusted = mf + (z**2) / (2 * n_sys)
    
    # 计算根号内的部分
    # (MF(1-MF)/N) + (z^2/4N^2)
    discriminant = (mf * (1 - mf) / n_sys) + (z**2) / (4 * n_sys**2)
    
    # 开根号并乘以 z
    root_term = z * np.sqrt(discriminant)
    
    # 计算最终的上下界
    # 这是一个 ± 运算，分别对应上界和下界
    upper_bound = (center_adjusted + root_term) / denominator
    lower_bound = (center_adjusted - root_term) / denominator
    
    return lower_bound, upper_bound

def calculate_poisson_interval(cf, n_sys, z=1):
    """
    计算基于泊松统计 (Poisson Statistics) 的置信区间。
    
    在天文学中，伴星计数通常服从泊松分布。当 CF 较大时 (>0.5)，
    二项分布假设失效，使用泊松误差更为准确。
    
    公式:
    sigma_CF = sqrt(CF / n_sys)
    区间 = CF ± z * sigma_CF

    参数:
    cf (float): 观测到的伴星频率 (Companion Fraction)，可以大于 1。
    n_sys (int or float): 样本大小 (Number of systems)。
    z (float): 对应置信水平的 Z 分数。默认为 1 (代表 1-sigma 不确定度)。
    
    返回:
    tuple: (lower_bound, upper_bound) 置信区间的下界和上界。
    """
    if n_sys <= 0:
        return 0.0, 0.0
        
    # 计算 CF 的标准误差 (Standard Error)
    standard_error = np.sqrt(cf / n_sys)
    
    # 乘以 Z 值得到误差范围
    margin_of_error = z * standard_error
    
    # 计算最终的上下界
    upper_bound = cf + margin_of_error
    
    # 物理限制：伴星频率不可能小于 0，因此强制下界最小为 0
    lower_bound = max(0.0, cf - margin_of_error)
    
    return lower_bound, upper_bound

def P_companion_given_detection_old(d,PComp,Sigma_local,tau=0.75):
    numerator = tau * PComp
    denominator = tau * PComp + (1 - np.exp(-tau * Sigma_local * np.pi * d**2)) * (1 - tau * PComp)
    return numerator / denominator

def P_companion_given_detection(d, P_input, Sigma_local, tau=0.75):
    """
    改进版的贝叶斯先验映射。
    P_input: 这里传入你每次迭代算出来的 CF。
    """
    # 核心改进1：泊松映射 (将期望值 CF 映射为真正的概率 P ∈[0, 1))
    # 物理意义：平均有 CF 个伴星 -> 至少有 1 个伴星的概率
    true_prior = 1.0 - np.exp(-P_input)
    
    # 核心改进2：工程安全锁 (绝对禁止出现 1.0 导致分母项归零爆炸)
    # 给定一个极小的容差，确保 1 - tau*P_prior 永远为正
    # true_prior = min(true_prior, 0.999)
    
    # 标准贝叶斯推断
    numerator = tau * true_prior
    denominator = tau * true_prior + (1 - np.exp(-tau * Sigma_local * np.pi * d**2)) * (1 - tau * true_prior)
    
    return numerator / denominator

class HierarchyProbability:
    def __init__(self, Z, fluxes, distance, CF_ori=0.2, Sigma=770, tau=0.5, CF_intep1d_instance=None, log_bins_mine=None, unfolder_A_matrix=None, P_bound_3D_array=None, density_3d_mine=None):
        self.Z = Z
        self.fluxes = fluxes
        self.n_stars = len(fluxes)
        self.n_nodes = self.n_stars + len(Z) # 总节点数 (叶子 + 聚类)
        self.distance = distance
        self.CF_ori = CF_ori
        self.Sigma = Sigma
        self.tau = tau
        self.CF_intep1d_instance = CF_intep1d_instance
        self.log_bins_mine = log_bins_mine
        self.unfolder_A_matrix = unfolder_A_matrix
        self.P_bound_3D_array = P_bound_3D_array
        self.density_3d_mine = density_3d_mine


        # 存储每个节点的"最大亮度"，用于判定谁是主星
        self.node_max_flux = np.zeros(self.n_nodes)
        # 存储每颗原始恒星的最终概率
        self.final_probs = np.zeros(self.n_stars)
        
        # 初始化叶子节点亮度
        self.node_max_flux[:self.n_stars] = self.fluxes

    def precompute_fluxes(self):
        """自底向上：计算每个聚类节点包含的最亮星的亮度"""
        for i in range(len(self.Z)):
            cluster_idx = self.n_stars + i
            left_idx = int(self.Z[i, 0])
            right_idx = int(self.Z[i, 1])
            
            # 当前簇的亮度 = 左右子簇中最亮的那个
            self.node_max_flux[cluster_idx] = max(
                self.node_max_flux[left_idx], 
                self.node_max_flux[right_idx]
            )

    def propagate_probability(self, node_idx, current_prob, P_cal_method="Tobin"):
        """自顶向下：递归分配概率"""
        # 如果是叶子节点 (原始恒星)，记录最终概率并结束
        if node_idx < self.n_stars:
            self.final_probs[node_idx] = current_prob
            return

        # 如果是聚类节点，继续向下拆分
        row_idx = node_idx - self.n_stars
        left_idx = int(self.Z[row_idx, 0])
        right_idx = int(self.Z[row_idx, 1])
        dist = self.Z[row_idx, 2]
        
        # 计算当前这一层的连接概率
        # print(dist * 3600 * self.distance)
        d_pc = dist * 3600 * self.distance / pc2au # 将au距离转换为pc
        d_au = dist * 3600 * self.distance
        # print(dist* 3600 * self.distance,d_pc)
        # p_link = P_companion_given_detection(d_pc,self.CF_ori,self.Sigma,tau=self.tau)
        # print(p_link)

        if self.CF_intep1d_instance is None:
            if P_cal_method == "Tobin":
                p_link = P_companion_given_detection(d_pc,self.CF_ori,self.Sigma,tau=self.tau)
            elif P_cal_method == "Mine":
                p_link = get_final_2d_bound_probability(s_proj=d_au, log_bins=self.log_bins_mine, unfolder_A_matrix=self.unfolder_A_matrix, P_bound_3D_array=self.P_bound_3D_array, P_s_total=self.density_3d_mine)
        else:
            # print(d_pc)
            # 都提供CF插值实例了，就只用tobin method算概率了
            CF_this_dis = self.CF_intep1d_instance(d_au) 
            p_link = P_companion_given_detection(d_pc,CF_this_dis,self.Sigma,tau=self.tau)
        
        # 比较亮度，决定谁乘 1.0，谁乘 p_link
        # Tobin logic: The brighter component is the primary (1.0)
        left_flux = self.node_max_flux[left_idx]
        right_flux = self.node_max_flux[right_idx]
        
        if left_flux >= right_flux:
            # 左边亮：左边是主星 (x 1.0)，右边是伴星 (x p_link)
            self.propagate_probability(left_idx, current_prob * 1.0, P_cal_method=P_cal_method)
            self.propagate_probability(right_idx, current_prob * p_link, P_cal_method=P_cal_method)
        else:
            # 右边亮：右边是主星 (x 1.0)，左边是伴星 (x p_link)
            self.propagate_probability(left_idx, current_prob * p_link, P_cal_method=P_cal_method)
            self.propagate_probability(right_idx, current_prob * 1.0, P_cal_method=P_cal_method)

    def run(self, P_cal_method="Tobin"):
        self.precompute_fluxes()
        # 从根节点开始，初始概率设为 1.0 (假设系统本身存在)
        root_idx = self.n_nodes - 1
        self.propagate_probability(root_idx, 1.0, P_cal_method=P_cal_method)
        return self.final_probs
    

class Cluster_Property:
    def __init__(self, distance, db_path, table_name="final_source_catalogue"):
        self.db_path = db_path
        self.distance = distance
        self.table_name = table_name
        
        # 1. 建立连接
        if isinstance(db_path, sqlite3.Connection):
            self.conn = db_path
        elif isinstance(db_path, str):
            self.conn = sqlite3.connect(db_path)
        
        # 2. 读取所有数据到字典
        print(f"Loading data from table: {table_name}...")
        data_dict = fetch_table_as_arrays(self.conn, self.table_name)
        
        # 3. 将列数据映射为实例变量 (全量映射)
        
        # --- ID & 坐标 ---
        # 对应 'Source ID'
        self.source_id = data_dict.get('Source ID')
        # 对应 'ra', 'ra Error', 'dec', 'dec Error'
        self.ra_array = data_dict.get('ra')
        self.ra_err_array = data_dict.get('ra Error')
        self.dec_array = data_dict.get('dec')
        self.dec_err_array = data_dict.get('dec Error')
        
        # --- Imfit Flux (高斯拟合通量) ---
        # 对应 'Total Flux', 'Total Flux Error'
        self.imfit_flux_array = data_dict.get('Total Flux')
        self.imfit_flux_err_array = data_dict.get('Total Flux Error')
        
        # 对应 'Peak Intensity', 'Peak Intensity Error'
        self.peak_intensity_array = data_dict.get('Peak Intensity')
        self.peak_intensity_err_array = data_dict.get('Peak Intensity Error')
        
        # --- Sum Flux (求和通量) ---
        # 对应 'Sum Flux', 'Sum Flux Error'
        self.sum_flux_array = data_dict.get('Sum Flux')
        self.sum_flux_err_array = data_dict.get('Sum Flux Error')
        
        # --- 几何参数 (FWHM & PA) ---
        # 对应 'deconmajFWHM' 等
        self.maj_fwhm_array = data_dict.get('deconmajFWHM')
        self.maj_fwhm_err_array = data_dict.get('deconmajFWHM Error')
        self.min_fwhm_array = data_dict.get('deconminFWHM')
        self.min_fwhm_err_array = data_dict.get('deconminFWHM Error')
        self.pa_array = data_dict.get('deconPA')
        self.pa_err_array = data_dict.get('deconPA Error')
        
        # --- 图像与统计 ---
        # 对应 'image'
        self.image_names = data_dict.get('image')
        # 对应 'Surrounding MAD Std'
        self.surrounding_mad_std = data_dict.get('Surrounding MAD Std')
        
        # --- 标志位 (Bool Flags) ---
        # 对应 'Surrounding Complex Bool'
        self.surrounding_complex_bool = data_dict.get('Surrounding Complex Bool')
        # 对应 'Manual Fit Bool'
        self.manual_fit_bool = data_dict.get('Manual Fit Bool')
        # 对应 'Asymmetry Bool'
        self.asymmetry_bool = data_dict.get('Asymmetry Bool')
        self.SNR_array = data_dict.get('SNR')

    def apply_pbcor(self,instance,pbcor_ratio):
        # instance 是对应的 Formation_Cluster 实例
        ra_pix,dec_pix = instance.wcs.celestial.all_world2pix(self.ra_array,self.dec_array,0)
        cor_ratio_array = np.zeros_like(self.ra_array)
        for i in range(len(self.ra_array)):
            ra_pix_int = round(ra_pix[i])
            dec_pix_int = round(dec_pix[i])
            cor_ratio_array[i] = pbcor_ratio[dec_pix_int, ra_pix_int]

        sum_flux_array_pbcor = self.sum_flux_array * cor_ratio_array
        sum_flux_err_array_pbcor = self.sum_flux_err_array * cor_ratio_array
        self.sum_flux_array_pbcor = sum_flux_array_pbcor
        self.sum_flux_err_array_pbcor = sum_flux_err_array_pbcor
        imfit_flux_array_pbcor = self.imfit_flux_array * cor_ratio_array
        imfit_flux_err_array_pbcor = self.imfit_flux_err_array * cor_ratio_array
        self.imfit_flux_array_pbcor = imfit_flux_array_pbcor
        self.imfit_flux_err_array_pbcor = imfit_flux_err_array_pbcor
        ins_maxflux = np.nanargmax(imfit_flux_array_pbcor)
        ra_center = self.ra_array[ins_maxflux]
        dec_center = self.dec_array[ins_maxflux]
        self.ra_center = ra_center
        self.dec_center = dec_center
        ra_modi_array = self.ra_array * np.cos(np.radians(self.dec_center))
        sources_points = np.vstack([ra_modi_array,self.dec_array]).T
        self.source_points = sources_points
        distance = self.distance
        delta_ra_array = (self.ra_array - self.ra_center) * np.cos(np.radians(self.dec_center))
        delta_dec_array = self.dec_array - self.dec_center
        distance_deg = np.sqrt(delta_ra_array**2 + delta_dec_array**2)
        distance_pc = np.radians(distance_deg) * distance
        self.distance_to_center_pc = distance_pc
        return None

    def multiplicity_analyse(self,distance,thresh_multi_au=1000
                             ,show=True,criterion='distance',method='centroid',weighted=False,return_numbers=False):
        """
        Performs multiplicity analysis using logarithmically-scaled flux weighting 
        for the centroid calculation in hierarchical clustering.
        """
        # --- 1. 准备坐标数据 ---
        # source points 的单位是度
        ra_modi_array = self.ra_array * np.cos(np.radians(self.dec_center))
        sources_points = np.vstack([ra_modi_array, self.dec_array]).T

        # --- 3. 运行 fclusterdata 并映射回结果 ---
        thresh_multi = thresh_multi_au / distance / 3600 #in degree
        
        # 在加权后的点集上运行聚类
        if weighted:
             # --- 2. 创建对数权重并准备加权聚类 ---
            log_flux = np.log10(self.total_flux_pbcor)

            # 计算对数流量的范围，并定义两个阈值来划分三个等级
            min_log_flux = np.min(log_flux)
            max_log_flux = np.max(log_flux)
            
            # 防止所有流量都相同时出现除零错误
            if np.isclose(min_log_flux, max_log_flux):
                weights = np.ones_like(log_flux, dtype=int)
            else:
                range_log_flux = max_log_flux - min_log_flux
                threshold_1 = min_log_flux + range_log_flux / 3
                threshold_2 = min_log_flux + 2 * range_log_flux / 3

                # 根据阈值分配权重 [1, 2, 3]
                weights = np.ones_like(log_flux, dtype=int) # 默认权重为1
                weights[log_flux > threshold_1] = 2 # 中等亮度
                weights[log_flux > threshold_2] = 3 # 最高亮度
            
            # 根据权重重复数据点以模拟加权
            weighted_points = np.repeat(sources_points, weights, axis=0)
            
            # 创建一个索引，用于将聚类结果从加权空间映射回原始空间
            original_indices = np.repeat(np.arange(len(sources_points)), weights)
            weighted_labels = fclusterdata(weighted_points, t=thresh_multi, criterion=criterion, method=method)
            # 将加权标签映射回原始数据点的标签
            # 这是一个高效的映射方法，确保 labels 的长度与 sources_points 一致
            labels = np.zeros(len(sources_points), dtype=int)
            unique_indices_pos = np.searchsorted(original_indices, np.arange(len(sources_points)))
            labels = weighted_labels[unique_indices_pos]
        else:
            labels = fclusterdata(sources_points, t=thresh_multi, criterion=criterion, method=method)

        self.group_labels_last = labels

        # --- 从这里开始，您的原始代码可以几乎不变地继续工作 ---

        if show:
            # 4. 可视化
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111)
            for k in np.unique(labels):
                group = sources_points[labels == k]
                ax.scatter(group[:, 0], group[:, 1], s=5)
                
                # 画一个最小覆盖圆
                center = group.mean(axis=0) # 注意：这里的center是几何中心，用于画圆
                radius = np.max(np.linalg.norm(group - center, axis=1)) + thresh_multi * 1.2  # 加余量
                if len(group) >= 2:
                    circle = plt.Circle(center, radius, color='C'+str((k-1)%10), fill=False, lw=2, alpha=0.5)
                    ax.add_patch(circle)
            
            ax.set_title("Grouping by Distance (Log-Weighted Centroid), with Circles")
            ax.set_aspect('equal')
            ax.invert_xaxis()
            plt.show()

        # --- 5. 计算 MF 和 CF ---
        non_single = 0
        num_members = np.array([])
        unique_labels = np.unique(labels)
        multiple_systems = [] # only valid when CF_this_distance is None
        for k in unique_labels:
            group_mask = (labels == k)
            group = sources_points[group_mask]
            flux_this_group = self.imfit_flux_array_pbcor[group_mask]  # 要换成imfit的flux
            ra_array_this_group = ra_modi_array[group_mask]
            dec_array_this_group = self.dec_array[group_mask]
        
            # 计算该组内的成员数
            num_in_group = len(group)

            if num_in_group >= 2:
                non_single += 1
                ins_prim_to_smallest = np.argsort(self.imfit_flux_array_pbcor[group_mask])[::-1]  # 从大到小排序的索引
                flux_this_group = flux_this_group[ins_prim_to_smallest]
                idx_primary = np.argmax(flux_this_group)
                ra_primary = ra_array_this_group[idx_primary]
                dec_primary = dec_array_this_group[idx_primary]
                distance_to_primary = np.sqrt((ra_array_this_group - ra_primary)**2 +  # 此处的ra_array_this_group和dec_array_this_group是已经乘了cos(dec_center)的了
                                            (dec_array_this_group - dec_primary)**2)
                
                # P_comp_given_detec_array = np.ones_like(distance_to_primary)
                # if CF_this_distance is not None:
                #     for i in range(len(distance_to_primary)):
                #         if i >= 0:
                #             P_comp_given_detec_array[i] = P_companion_given_detection(d=distance_to_primary[i]*3600*distance/pc2au, PComp=CF_this_distance, Sigma_local=global_Sigma, tau=tau)
                #     P_c_rest = P_comp_given_detec_array[1:]
                #     non_single_cor += 1 - np.prod(1 - P_c_rest)

                multiple_this = {
                    'num_members': num_in_group,
                    'sources_index': group_mask,
                    'distance_to_primary_au': distance_to_primary * 3600 * distance,
                    # 'probability_companion_given_detection': P_comp_given_detec_array,
                    'ra_array_deg_cosdelta': ra_array_this_group,
                    'dec_array_deg': dec_array_this_group
                }
                multiple_systems.append(multiple_this)

            num_members = np.append(num_members, num_in_group)
        
        MF = non_single / len(unique_labels) if len(unique_labels) > 0 else 0
        MF_sigma_interval = calculate_wilson_interval(MF, len(unique_labels), z=1)

        companions = 0
        # companions_cor = 0
        for i in num_members:
            companions += (i - 1)

        # if CF_this_distance is not None:
        #     for idx, num in enumerate(num_members):
        #         if num >= 2:
        #             P_c_rest = multiple_systems[idx]['probability_companion_given_detection'][1:]
        #             companions_cor += np.sum(P_c_rest)

        CF = companions / len(unique_labels) if len(unique_labels) > 0 else 0
        if CF > 0.5:
            CF_sigma_interval = calculate_poisson_interval(CF, len(unique_labels), z=1)
        else:
            CF_sigma_interval = calculate_wilson_interval(CF, len(unique_labels), z=1)
        
        # single_systems = {
        #     'distance_to_center_pc': distance_to_center_single_array,
        #     'disk_radius_au': disk_radius_single_array,
        #     'disk_radius_err_au': disk_radius_err_single_array,
        #     'disk_dust_mass_Mearth': disk_dust_mass_single_array,
        #     'disk_dust_mass_err_Mearth': disk_dust_mass_err_single_array
        # }
        # distance_au_array = distance_array * 3600 * distance

        if return_numbers:
            num_multiple_systems = non_single
            num_all = len(unique_labels)
            num_companions = companions
            return MF, CF, MF_sigma_interval, CF_sigma_interval, multiple_systems, num_multiple_systems, num_all, num_companions
        
        return MF, CF, MF_sigma_interval, CF_sigma_interval, multiple_systems   #, distance_au_array   现在距离统计不从这里出来
    
    def multiplicity_analyse_contamination_corrected(self,distance,thresh_multi_au=1000,CF_this_distance=None,global_Sigma=1000,tau=0.5,
                                                     show=True,criterion='distance',method='centroid',P_cal_method="Tobin", **kwargs):
        """
        Performs multiplicity analysis using logarithmically-scaled flux weighting 
        for the centroid calculation in hierarchical clustering.
        """
        # --- 0. 读入kwargs
        # 获取 Mine 模式下需要的外部参数
        log_bins_mine = kwargs.get('log_bins_mine')
        unfolder_A_matrix = kwargs.get('unfolder_A_matrix')
        P_bound_3D_array = kwargs.get('P_bound_3D_array')
        density_3d_mine = kwargs.get('density_3d_mine')
        get_final_2d_bound_probability = kwargs.get('get_final_2d_bound_probability_func')

        # --- 1. 准备坐标数据 ---
        # source points 的单位是度
        ra_modi_array = self.ra_array * np.cos(np.radians(self.dec_center))
        sources_points = np.vstack([ra_modi_array, self.dec_array]).T

        # --- 3. 运行 fclusterdata 并映射回结果 ---
        thresh_multi = thresh_multi_au / distance / 3600 #in degree
        
        # 在加权后的点集上运行聚类
        labels = fclusterdata(sources_points, t=thresh_multi, criterion=criterion, method=method)

        self.group_labels_last = labels

        # --- 从这里开始，您的原始代码可以几乎不变地继续工作 ---

        if show:
            # 4. 可视化
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111)
            for k in np.unique(labels):
                group = sources_points[labels == k]
                ax.scatter(group[:, 0], group[:, 1], s=5)
                
                # 画一个最小覆盖圆
                center = group.mean(axis=0) # 注意：这里的center是几何中心，用于画圆
                radius = np.max(np.linalg.norm(group - center, axis=1)) + thresh_multi * 1.2  # 加余量
                if len(group) >= 2:
                    circle = plt.Circle(center, radius, color='C'+str((k-1)%10), fill=False, lw=2, alpha=0.5)
                    ax.add_patch(circle)
            
            ax.set_title("Grouping by Distance (Log-Weighted Centroid), with Circles")
            ax.set_aspect('equal')
            ax.invert_xaxis()
            plt.show()

        # --- 5. 计算 MF 和 CF ---
        non_single = 0
        companions = 0  
        systems = 0

        unique_labels = np.unique(labels)
        for k in unique_labels:
            group_mask = (labels == k)
            group = sources_points[group_mask]
            flux_this_group = self.imfit_flux_array_pbcor[group_mask]  # 要换成imfit的flux
    
            num_in_group = len(group)
            final_systems_structure = [] # 存储最终判定出的各系统成员数
            if num_in_group >= 2:                
                # 定义递归函数
                def resolve_system_structure(indices_subset):
                    # 1. 递归基：单星直接返回
                    n_current = len(indices_subset)
                    if n_current < 2:
                        return [1]

                    # 2. 准备子组数据 & 计算 Z 矩阵
                    sub_points = group[indices_subset]
                    sub_fluxes = flux_this_group[indices_subset]
                    Z_sub = linkage(sub_points, method=method)

                    # 3. 计算当前结构下的概率总和 (Step 1: Check Sum)
                    # 注意：HierarchyProbability 每次 run 都会重新归一化根节点概率为 1.0
                    calculator = HierarchyProbability(
                        Z_sub, 
                        sub_fluxes, 
                        self.distance, 
                        CF_ori=CF_this_distance, 
                        Sigma=global_Sigma, 
                        tau=tau,
                        log_bins_mine=log_bins_mine,
                        unfolder_A_matrix=unfolder_A_matrix,
                        P_bound_3D_array=P_bound_3D_array,
                        density_3d_mine=density_3d_mine
                    )
                    probs = calculator.run(P_cal_method=P_cal_method)
                    # print(probs)
                    sum_probs = np.sum(probs)
                    n_effective = int(round(sum_probs))
                    
                    # 确保至少为1 (除非所有概率都极低且四舍五入为0，但主星是1.0所以sum>=1.0)
                    if n_effective < 1: n_effective = 1

                    # --- Tobin Logic 判断开始 ---

                    # 情况 A: 概率和足够高，没有损失成员 (Round(Sum) == N)
                    # 例子: 2星, P=0.8, Sum=1.8 -> 2. 
                    if n_effective == n_current:
                        return [n_effective] # 不拆分，直接返回当前成员数

                    # 情况 B: 概率和降低了，成员“丢失”了 (Round(Sum) < N)
                    # 例子: 2星, P=0.4, Sum=1.4 -> 1. (少了1个)
                    # 例子: 10星, P=0.9, Sum=9.0 -> 9. (少了1个)
                    else:
                        # 此时需要“Check difference”：是哪个连接导致了损失？
                        # 我们遍历 Z 矩阵，找到 P(d) 最小的那个连接
                        
                        min_p_link = 1.0
                        split_node_idx = -1 # 记录在 Z 矩阵哪一行切分

                        # 遍历 Z 矩阵每一行计算 P(d)
                        for i in range(len(Z_sub)):
                            dist_deg = Z_sub[i, 2]
                            dist_au = dist_deg * self.distance * 3600
                            
                            # 计算连接概率
                            if P_cal_method == "Tobin":
                                p_val = P_companion_given_detection(
                                    dist_au / pc2au, 
                                    CF_this_distance, 
                                    global_Sigma, 
                                    tau=tau
                                )
                            elif P_cal_method == "Mine":
                                p_val = get_final_2d_bound_probability(
                                    s_proj=dist_au, 
                                    log_bins=log_bins_mine, 
                                    unfolder_A_matrix=unfolder_A_matrix, 
                                    P_bound_3D_array=P_bound_3D_array, 
                                    P_s_total=density_3d_mine
                                )
                            # print(p_val)
                            
                            # 记录最小值和位置
                            if p_val < min_p_link:
                                min_p_link = p_val
                                split_node_idx = i # 更新下最小概率连接处的索引

                        # 阈值判断：导致成员减少的原因是“弱连接”吗？
                        # Tobin implies: if the link is low probability (<0.5), it is the cause.
                        # 如果 P < 0.5，说明这个连接贡献了主要的“概率损失” (1 - P > 0.5)
                        
                        # 之前是判断是否小于0.5，现在改成最小的直接拆分
                        # if min_p_link < 0.5:

                        # === 决定拆分 (Split) ===
                        # 在 split_node_idx 处切断，分成两个独立的系统
                        
                        # 从 Z 矩阵恢复该节点的左右子树
                        # 注意：Z 矩阵第 i 行生成的簇索引是 n_current + i
                        cluster_idx_in_tree = n_current + split_node_idx
                        
                        # 使用 to_tree 构建完整的树，然后找到对应节点的左右子节点
                        # 这里稍微复杂一点，因为 to_tree 返回根节点。
                        # 简单的办法：Z[i, 0] 和 Z[i, 1] 就是被合并的两个簇的索引
                        
                        idx_left_cluster = int(Z_sub[split_node_idx, 0])
                        idx_right_cluster = int(Z_sub[split_node_idx, 1])
                        
                        # 辅助函数：获取簇索引包含的所有原始叶子索引
                        def get_leaves_from_Z_idx(cluster_idx, n_leafs, Z_matrix):
                            if cluster_idx < n_leafs:
                                return [cluster_idx]
                            else:
                                row = cluster_idx - n_leafs
                                return get_leaves_from_Z_idx(int(Z_matrix[row, 0]), n_leafs, Z_matrix) + \
                                        get_leaves_from_Z_idx(int(Z_matrix[row, 1]), n_leafs, Z_matrix)

                        left_indices_local = get_leaves_from_Z_idx(idx_left_cluster, n_current, Z_sub)
                        right_indices_local = get_leaves_from_Z_idx(idx_right_cluster, n_current, Z_sub)
                        
                        # 映射回 indices_subset 的全局索引
                        left_indices_global = [indices_subset[x] for x in left_indices_local]
                        right_indices_global = [indices_subset[x] for x in right_indices_local]
                        
                        # # 递归处理这两个新分出来的系统
                        # # 精髓递归，如果真拆分，每一部分要么是[1] 要么是[n_effective]，然后list + list = [1, n_effective, ...], 真是太棒了
                        # return resolve_system_structure(left_indices_global) + resolve_system_structure(right_indices_global)  

                        # ==========================================================
                        # 🚨 终极 Bug 修复：找回因为切断内部树枝而变成“孤儿”的星星
                        # ==========================================================
                        # 将左右子树的成员集合起来
                        involved_in_split = set(left_indices_global + right_indices_global)
                        
                        # 找出当前组里，没在这个切断节点里的所有其它星 (比如星 C)
                        everything_else_global =[x for x in indices_subset if x not in involved_in_split]
                        
                        # 1. 递归处理左子树
                        ans = resolve_system_structure(left_indices_global)
                        # 2. 递归处理右子树
                        ans += resolve_system_structure(right_indices_global)
                        
                        # 3. 如果有孤儿星，把它们打包重新送进递归！
                        # 它们会因为不受 A、B 错误质心的干扰，重新计算出正确的物理归宿
                        if len(everything_else_global) > 0:
                            ans += resolve_system_structure(everything_else_global)
                            
                        return ans


                        # else:
                        #     # === 不拆分 (Keep) ===
                        #     # 虽然成员减少了 (n_effective < n_current)，但最弱的连接 P >= 0.5
                        #     # 说明这是“累积概率损失” (Death by a thousand cuts)
                        #     # 这种情况下，我们接受 n_effective 作为该系统的最终成员数
                        #     return [n_effective]

                # 启动递归
                final_systems_structure = resolve_system_structure(list(range(num_in_group)))
            
            else:
                # 只有一颗星
                final_systems_structure = [1]

            # print(final_systems_structure)
            
            # --- 统计更新逻辑 ---
            # 现在 final_systems_structure 包含了该空间组内所有的独立系统
            # 例如: [1, 2] 表示有一个单星和一个双星
            systems += len(final_systems_structure)

            for n_members in final_systems_structure:
                # 统计 MF (System >= 2)
                if n_members >= 2:
                    non_single += 1
                companions += (n_members - 1)
        
        MF = non_single / systems
        MF_sigma_interval = calculate_wilson_interval(MF, systems, z=1)

        CF = companions / systems
        if CF > 0.5:
            CF_sigma_interval = calculate_poisson_interval(CF, systems, z=1)
        else:
            CF_sigma_interval = calculate_wilson_interval(CF, systems, z=1)

        return MF, CF, MF_sigma_interval, CF_sigma_interval

        # num_in_group = len(group)
        # if num_in_group >= 2:
        #     if CF_this_distance is not None:
        #         # fcluster_this = fclusterdata(group, t=thresh_multi, criterion=criterion, method=method)
        #         Z = linkage(group, method=method)
        #         # 4. 执行计算
        #         calculator = HierarchyProbability(Z, flux_this_group, self.distance, CF_ori=CF_this_distance, Sigma=global_Sigma, tau=tau)
        #         probs = calculator.run()
        #         effective_components = np.sum(probs)
        #         int_effective_components = int(round(effective_components))
        #         if int_effective_components >= 2:
        #             non_single += 1
        #         if int_effective_components < num_in_group: # 有可能会分出来一些单独的新系统
        #             for merge in Z:
        #                 P_companion_given_detection(merge[2] * 3600 * self.distance / pc2au, CF_this_distance, global_Sigma, tau=tau)
        #         num_in_group = int_effective_components
        
    
def B_nu(freq,T):
    # return in W m^-2 Hz^-1 sr^-1
    return ((2 * cons.h.value * freq**3)/((cons.c.value)**2)) * (1/(np.exp((cons.h.value * freq)/(cons.k_B.value*T)) - 1))

def estimate_dust_mass_optically_thin(F,d,freq=226074026858.4,kappa=0.899,T=20):
    # F in Jy
    # d in pc
    # freq in Hz
    # kappa in cm^2 g^-1
    # T in K
    # return in Msun
    # default values are for ALMA band 6 : 1.3 mm
    F_SI = F * 1e-26
    d_m = d * cons.pc.value
    kappa_m2_per_kg = kappa * 0.1
    B_nu_SI = B_nu(freq,T)
    
    M_kg = (F_SI * d_m**2) / (kappa_m2_per_kg * B_nu_SI)
    M_sun = M_kg / cons.M_earth.value
    return M_sun

def cal_dust_gas_mass(total_flux_pbcor,total_flux_pbcor_err,instance,distance_pc=140,temperature_K=20,kappa_band6=0.899,gas_to_dust_ratio=100):
    u_total_flux = unumpy.uarray(total_flux_pbcor, total_flux_pbcor_err)
    M_dust_array = estimate_dust_mass_optically_thin(u_total_flux,distance_pc,freq=instance.Freq,kappa=kappa_band6,T=temperature_K)
    M_gas_array = M_dust_array * gas_to_dust_ratio
    return M_dust_array, M_gas_array



# Cluste spatial distribution analysis
from scipy.spatial.distance import pdist
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import astropy.units as u
from matplotlib.ticker import MaxNLocator
import networkx as nx
from pathlib import Path
from adjustText import adjust_text
from tqdm import tqdm

# ps distribution calculation based on Cartwright & Whitworth (2004) Section 3.2 & 3.3
def calculate_ps_distribution(ra_array, dec_array, distance_pc, ra_center=None, dec_center=None, R_cluster=None,
                               i_max=20, visualize=True, cname="Cluster", ax=None):
    """
    计算星团的 p(s) 分布和归一化关联长度 s_bar。
    基于 Cartwright & Whitworth (2004) Section 3.2 & 3.3。

    参数:
    ra_array, dec_array: 源的坐标数组 (度)
    distance_pc: 星团距离 (pc)，用于物理尺度转换
    i_max: bin 的数量 (用户自定义，默认 20)
    visualize: 是否画图

    返回:
    dict: 包含计算结果的字典
        - 's_bar': 归一化关联长度
        - 's_values': bin 的中心值 (x轴)
        - 'p_s': 概率密度值 (y轴)
        - 'R_cluster': 星团半径 (AU)
    """
    
    num_stars = len(ra_array)
    if num_stars < 2:
        print("源数量不足，无法计算 p(s)。")
        return None

    # --- 1. 坐标转换 (Deg -> AU) 并中心化 ---
    # 使用平均位置作为几何中心
    if ra_center is None and dec_center is None:
        ra_mean = np.mean(ra_array)
        dec_mean = np.mean(dec_array)
        ra_center = ra_mean
        dec_center = dec_mean
    
    # 简单的平面对射投影 (Flat approximation)
    cos_dec = np.cos(np.radians(dec_center))
    scale_factor = 3600 * distance_pc # 1度 = 3600角秒 * 距离 = AU
    
    x_au = (ra_array - ra_center) * cos_dec * scale_factor
    y_au = (dec_array - dec_center) * scale_factor
    
    # 组合成 (N, 2) 数组
    coords = np.column_stack([x_au, y_au])

    # --- 2. 计算 R_cluster (星团半径) ---
    # 定义：从平均位置(0,0)到最远恒星的距离
    dist_from_center = np.linalg.norm(coords, axis=1)
    if R_cluster is None:
        R_cluster = np.max(dist_from_center)
        # print(f"Calculated R_cluster: {R_cluster:.2f} AU")

    # --- 3. 计算所有成对距离 (Separations) ---
    # pdist 返回 N*(N-1)/2 个距离
    separations_au = pdist(coords)
    
    # --- 4. 计算 s_bar (归一化关联长度) ---
    # s_bar = mean(separations) / R_cluster
    mean_separation = np.mean(separations_au)
    s_bar = mean_separation / R_cluster

    # --- 5. 计算 p(s) 分布 ---
    # 论文中 p(s) 的自变量 s 是归一化后的距离，即 s = separation / R_cluster
    # s 的范围通常在 0 到 2 之间
    s_normalized = separations_au / R_cluster
    
    # 定义 bins
    # 范围 0 到 2，共 i_max 个 bin

    relative_Rmax = 2 * np.max(dist_from_center) / R_cluster
    bins_edge = np.linspace(0, relative_Rmax, i_max + 1)
    delta_s = bins_edge[1] - bins_edge[0] # bin width
    
    # 计算直方图 (Counts)
    counts, _ = np.histogram(s_normalized, bins=bins_edge)
    
    # 转换为概率密度 p(s)
    # 论文公式 (3): p(s_i) = 2 * N_i / (N_total * (N_total - 1) * delta_s)
    # 其中 N_total * (N_total - 1) / 2 正好是总的成对数量 (len(s_normalized))
    # 所以公式等价于: p(s_i) = (N_i / 总对数) / delta_s
    # 这就是标准的概率密度归一化
    
    total_pairs = len(s_normalized)
    p_s = counts / (total_pairs * delta_s)
    
    # 计算 bin 的中心点 (用于画图 x 轴)
    s_values = (bins_edge[:-1] + bins_edge[1:]) / 2

    # --- 6. 可视化 ---
    if visualize:
        if ax is None:
            fig, ax_plot = plt.subplots(figsize=(8,6))
        else:
            fig = ax.figure
            ax_plot = ax
        # # 绘制 p(s) 数据点/柱状图
        # ax_plot.bar(s_values, p_s, width=delta_s, align='center', 
        #         color='skyblue', edgecolor='black', alpha=0.7, label='Cluster Data')
        
        # 绘制平滑曲线 (可选)
        ax_plot.plot(s_values, p_s, 'k-o', markersize=7, linewidth=1,markerfacecolor='red',markeredgecolor='red',label=f'{cname} Data')
        
        # # 绘制参考线 p(s) = 2s (针对 s < 1 的均匀圆盘近似)
        # # 这有助于判断是否存在子结构 (s_bar < 0.8) 或中心聚集 (s_bar > 0.8)
        # s_ref = np.linspace(0, 1, 100)
        # ax_plot.plot(s_ref, 2 * s_ref, 'r--', label=r'Uniform Disk ($p(s)=2s$)')
        
        # 标注 s_bar
        ax_plot.axvline(s_bar, color='k', linestyle='--', linewidth=2, label=r'$\bar{s}$ (Mean)')
        ax_plot.text(0.95,0.95,cname, transform=ax_plot.transAxes, fontsize=15, ha='right', va='top')


        ax_plot.set_xlabel(r'Normalized Separation $s$')
        ax_plot.set_ylabel(r'$p(s)$')
        # ax_plot.settitle(f'Separation Distribution p(s)\n$\overline{{s}} = {s_bar:.2f}$, $i_{{max}} = {i_max}$')
        # ax_plot.set_xlim(0.05, 1.95)
        ax_plot.set_xlim(0.0, 2.0)
        ax_plot.set_ylim(0.0, 1.89)
        # ax_plot.legend()
        # ax_plot.grid(True, linestyle=':', alpha=0.6)
        ax_plot.minorticks_on()
        ax_plot.tick_params(which='major', direction='in',top=True,right=True)
        ax_plot.tick_params(which='minor', direction='in',top=True,right=True)
        if ax is None:
            plt.show()
    else:
        fig = None

    return {
        's_bar': s_bar,
        's_values': s_values,
        'p_s': p_s,
        'R_cluster_au': R_cluster,
        'fig': fig
    }

# all source MST
def create_and_visualize_mst2(ra_array, dec_array, instance=None, distance_pc=1000, cluster_name="Cluster", visualize=True, manual_center=None, ax=None):
    """
    基于 AU 物理单位生成 MST 并进行可视化。
    坐标原点 (0,0) 为图像中心。
    
    参数:
    ra_array, dec_array: 绝对坐标数组 (度)
    instance: 包含 head 信息的对象 (需有 CRVAL1, CRVAL2)
    distance_pc: 源的距离 (pc)
    visualize: 是否绘图
    """
    
    # --- 1. 坐标转换 (Deg -> AU) ---
    if manual_center is not None:
        ra_center, dec_center = manual_center
    elif instance is not None:
        ra_center = instance.head['CRVAL1']
        dec_center = instance.head['CRVAL2']
    else:
        ra_center = np.mean(ra_array)
        dec_center = np.mean(dec_array)
    cos_dec = np.cos(np.radians(dec_center))

    # 1. 计算角度 Offset (单位：度)
    # ra_offset 包含 cos(delta) 修正
    d_ra_deg = (ra_array - ra_center) * cos_dec
    d_dec_deg = dec_array - dec_center
    
    # 2. 转换为 AU
    # 1 deg = 3600 arcsec, 1 arcsec * dist(pc) = 1 AU
    # 公式: theta(deg) * 3600 * d(pc) = L(AU)
    if isinstance(distance_pc, (int, float)):
        scale_factor = 3600 * distance_pc
        xlabel = 'R.A. Offset (au)'
        ylabel = 'Dec. Offset (au)'
    else:
        scale_factor = 3600 #* float(distance_pc)
        xlabel = 'R.A. Offset (arcsec)'
        ylabel = 'Dec. Offset (arcsec)'

    x_au = d_ra_deg * scale_factor
    y_au = d_dec_deg * scale_factor
    
    num_points = len(ra_array)
    # 合并坐标 (N, 2)
    points_au = np.vstack([x_au, y_au]).T

    # --- 2. 构建完全图 ---
    G = nx.Graph()
    for i in range(num_points):
        G.add_node(i, pos=points_au[i])

    # 预计算所有边的权重 (欧氏距离, AU)
    edges_to_add = []
    for i in range(num_points):
        for j in range(i + 1, num_points):
            dist_au = np.linalg.norm(points_au[i] - points_au[j])
            edges_to_add.append((i, j, dist_au))
            
    G.add_weighted_edges_from(edges_to_add)

    # --- 3. 计算 MST ---
    # print("正在计算 MST (Prim算法)...")
    MST = nx.minimum_spanning_tree(G, algorithm='prim', weight='weight')
    # print(f"MST 计算完成: {MST.number_of_edges()} 条边。")

    # --- 4. 演讲级可视化 (AU 模式) ---
    if visualize:
        # plt.style.use('default') 
        if ax is None:
            fig, ax_plot = plt.subplots(figsize=(10, 10)) # 正方形画布
        else:
            fig = ax.figure
            ax_plot = ax
        
        # A. 提取 MST 边
        lines = []
        for u, v in MST.edges():
            pos_u = G.nodes[u]['pos'] # [x_au, y_au]
            pos_v = G.nodes[v]['pos']
            lines.append([pos_u, pos_v])
            
        # B. 绘制边
        lc = LineCollection(lines, colors='k', linewidths=1.5, alpha=1.0, zorder=1) #'#B0B0B0'
        ax_plot.add_collection(lc)
        
        # C. 绘制节点
        ax_plot.plot(x_au, y_au, 
                   'o', markersize=5, markerfacecolor='red', markeredgecolor='red', linestyle='None', 
                    zorder=2, label='Sources'
                   ) #'#007799'
        
        # D. 坐标轴设置
        # ax_plot.set_xlabel(r'$\Delta$ RA $\cos(\delta)$ (AU)', fontsize=12, fontfamily='serif')
        # ax_plot.set_ylabel(r'$\Delta$ Dec (AU)', fontsize=12, fontfamily='serif')
        # ax_plot.set_xlabel('R.A. Offset (au)', fontsize=12, fontfamily='serif')
        # ax_plot.set_ylabel('Dec. Offset (au)', fontsize=12, fontfamily='serif')
        ax_plot.set_xlabel(xlabel)
        ax_plot.set_ylabel(ylabel)
        
        # 1. RA 翻转 (东左西右)
        ax_plot.invert_xaxis()
        
        # 2. 强制正方形视场，并将 (0,0) 置于中心
        # 找出绝对值最大的坐标，并加一点余量 (10%)
        max_limit = np.max(np.abs(points_au)) * 1.1
        
        # 设置两轴范围一致，确保是正方形且中心为0
        ax_plot.set_xlim(max_limit, -max_limit) # 注意 x 是反转的: 正 -> 负
        ax_plot.set_ylim(-max_limit, max_limit)
        
        # 3. 强制 Aspect Ratio 为 1
        points_au = np.vstack([x_au, y_au]).T
        ax_plot.set_aspect('equal')
        
        # E. 添加中心十字丝 (参考线)
        ax_plot.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5, zorder=0)
        ax_plot.axvline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5, zorder=0)

        # F. 比例尺 (Scale Bar)
        # 自动找一个漂亮的整数值 (比如 1000, 2000, 5000)
        if isinstance(distance_pc, (int, float)):
            span_au = 2 * max_limit
            scale_len_au = 10 ** np.floor(np.log10(span_au * 0.2))
            # 简单的取整优化逻辑
            if scale_len_au * 5 < span_au * 0.3: scale_len_au *= 5
            elif scale_len_au * 2 < span_au * 0.3: scale_len_au *= 2
            
            scale_txt = f"{int(scale_len_au)} au"
            
            # 放置在左下角 (基于当前的 max_limit)
            # 左边界是 max_limit (因为翻转了), 下边界是 -max_limit
            x_start = max_limit - (max_limit * 2) * 0.08 # 向内缩 8%
            y_start = -max_limit + (max_limit * 2) * 0.08
            
            ax_plot.plot([x_start, x_start - scale_len_au], [y_start, y_start], 
                    color='black', linewidth=2, zorder=10)
            ax_plot.text(x_start - scale_len_au/2, y_start + max_limit * 0.03, 
                    scale_txt, ha='center', va='bottom', fontsize=10, fontweight='bold')
        else:
            # 设置真实距离的比例尺，从fc读入距离
            span_au = 10000# * instance.distance
            span_arcsec = span_au / instance.distance # 实际绘制单位是arcsec

            scale_txt = f"{int(span_au)} au"
            
            # 使用transAxes坐标系，在0.95, 0.05处放置
            ax_plot.text(0.95, 0.05, scale_txt, transform=ax_plot.transAxes, 
                         ha='right', va='bottom', fontsize=15)#, fontweight='bold')
            
            # 配合下面的 ax_plot.set_xlim(-20, 20) (视野宽 40 arcsec)，计算比例尺相对长度
            span_axes = span_arcsec / 40.0
            ax_plot.plot([0.95, 0.95 - span_axes], [0.03, 0.03], 
                         color='black', linewidth=2, zorder=10, transform=ax_plot.transAxes)


        # G. 标题与刻度
        title_str = f'MST Structure of {cluster_name}\nDistance = {distance_pc} pc'
        ax_plot.text(0.95, 0.95, cluster_name, transform=ax_plot.transAxes, fontsize=15, ha='right', va='top')
        # ax_plot.set_title(title_str, fontsize=14, fontweight='bold', pad=15)#, fontfamily='serif')
        
        # ax_plot.tick_params(direction='in', length=6, width=1, labelsize=10, top=True, right=True)
        if isinstance(distance_pc, (int, float)):
            pass
        else:
            ax_plot.set_xlim(-20, 20)
            ax_plot.set_ylim(-20, 20)
            ax_plot.invert_xaxis()
        
        ax_plot.minorticks_on()
        ax_plot.tick_params(which='major', top=True, bottom=True, left=True, right=True,direction='in') 
        ax_plot.tick_params(which='minor', top=True, bottom=True, left=True, right=True,direction='in') 

        if ax is None:
            plt.tight_layout()
            plt.show()
    else:
        fig = None

    return MST,fig

def analyze_mass_segregation(ra_array, dec_array, mass_array
                             , instance=None, distance_pc=1000
                             , visualize=True, cluster_name="Cluster"
                             , num_min=3
                             , manual_center=None):
    """
    计算并可视化质量隔离度 (Mass Segregation Ratio, MSR) 随 N_MST 变化的图表。
    使用 Allison et al. 2009 的方法:
    提取前 N 个最重目标的 MST 长度 (l_massive)，以及重复抽取 N 个随机目标的平均 MST 长度 (l_random)。
    MSR = <l_random> / l_massive。
    """
    
    # 提取质量 nominal 值并降序排序
    if hasattr(mass_array[0], 'nominal_value'):
        mass_array_val = unumpy.nominal_values(mass_array)
    else:
        mass_array_val = np.array(mass_array)
        
    sort_idx = np.argsort(-mass_array_val)
    ra_array_sorted = ra_array[sort_idx]
    dec_array_sorted = dec_array[sort_idx]
    
    msr_list = []
    msr_err_list = []
    
    # num_min = 2
    N_list = np.arange(num_min, len(ra_array), 1)
    
    def random_MST_length_statistic(N_MST, N_random):
        l_norm_list = np.zeros(N_random)
        for i in range(N_random):
            idx = np.random.choice(len(ra_array), size=N_MST, replace=False)
            sampled_ra = ra_array[idx]
            sampled_dec = dec_array[idx]
            mst_this, _ = create_and_visualize_mst2(
                sampled_ra, sampled_dec, 
                instance=instance, distance_pc=distance_pc, 
                visualize=False, manual_center=manual_center
            )
            distance_this = sum(d['weight'] for u, v, d in mst_this.edges(data=True))
            l_norm_list[i] = distance_this
        return np.mean(l_norm_list), np.std(l_norm_list)

    for N_MST in tqdm(N_list, desc="Calculating MSR"):
        mst_massive, _ = create_and_visualize_mst2(
            ra_array_sorted[:N_MST], dec_array_sorted[:N_MST], 
            instance=instance, distance_pc=distance_pc, 
            visualize=False, manual_center=manual_center
        )
        l_massive = sum(d['weight'] for u, v, d in mst_massive.edges(data=True))
        
        N_random = 100 if N_MST >= len(ra_array) / 2 else 1000
        mean_l_norm, std_l_norm = random_MST_length_statistic(N_MST, N_random)
        
        msr = mean_l_norm / l_massive
        msr_err = std_l_norm / l_massive
        msr_list.append(msr)
        msr_err_list.append(msr_err)
        
    msr_list = np.array(msr_list)
    msr_err_list = np.array(msr_err_list)
    
    if visualize:
        fontsize = 20
        fig, ax = plt.subplots(figsize=(9, 7), facecolor='white')
        
        ax.errorbar(N_list, msr_list, yerr=msr_err_list,
                    marker='o', markersize=9, linestyle='None',
                    markerfacecolor='None', color='black', capsize=5, 
                    zorder=3, elinewidth=1, label=f'{cluster_name}')
        ax.hlines(1.0, 0, len(ra_array), linestyles='--', linewidth=2.5, color='red', zorder=2)
        
        ax.tick_params(labelsize=fontsize, which='major', top=True, bottom=True, left=True, right=True, direction='in')
        ax.tick_params(which='minor', top=True, bottom=True, left=True, right=True, direction='in')
        
        ax.set_xlabel(r'$N_\mathrm{MST}$', fontsize=fontsize)
        ax.set_ylabel(r'Mass Segregation Ratio $\Lambda_\mathrm{MSR}$', fontsize=fontsize)
        ax.minorticks_on()
        ax.set_xlim(0, len(ra_array))
        ax.legend(prop={'size': 20}, loc='upper right')
        
        plt.show()
    else:
        fig = None
        
    return N_list, msr_list, msr_err_list, fig

# calculate m_bar based on Cartwright & Whitworth (2004) Section 3.2 & 3.3
def calculate_m_bar_strict(mst_graph, ra_array, dec_array, distance_pc,
                           ra_center=None,dec_center=None,R_cluster=None):
    """
    严格基于 Cartwright & Whitworth (2004) 定义计算归一化平均边长 m_bar。
    
    定义引用: 
    "the mean length of the branches of the tree, divided by 
    (N_total * A)^0.5 / (N_total - 1)"

    参数:
    mst_graph: networkx.Graph 对象 (边的权重属性名为 'weight'，单位需为 AU)
    ra_array, dec_array: 源的坐标数组 (度)
    distance_pc: 星团距离 (pc)

    返回:
    m_bar: 归一化平均边长
    debug_info: 包含中间变量的字典
    """
    
    # 1. 获取源数量 N_total
    N_total = mst_graph.number_of_nodes()
    if N_total < 2:
        print("源数量不足，无法计算 m_bar")
        return None, {}

    # 2. 计算 MST 总边长 (Sum of edge lengths)
    total_edge_length = mst_graph.size(weight='weight')
    
    # 3. 计算这一项: "mean length of the branches of the tree"
    # MST 的边数固定为 N_total - 1
    mean_edge_length = total_edge_length / (N_total - 1)
    
    # 4. 计算星团面积 A
    # 坐标转换 (Deg -> AU)，以平均位置为中心
    if ra_center is None and dec_center is None:
        ra_mean = np.mean(ra_array)
        dec_mean = np.mean(dec_array)
        ra_center = ra_mean
        dec_center = dec_mean
    
    cos_dec = np.cos(np.radians(dec_center))
    scale_factor = 3600 * distance_pc # 1度 -> AU
    
    x_au = (ra_array - ra_center) * cos_dec * scale_factor
    y_au = (dec_array - dec_center) * scale_factor
    
    # R_cluster: 从几何中心到最远源的距离
    if R_cluster is None:
        dist_from_center = np.sqrt(x_au**2 + y_au**2)
        R_cluster = np.max(dist_from_center)
    
    # Area: 论文定义的投影面积 A = pi * R^2
    Area = np.pi * R_cluster**2
    
    # 5. 计算分母: (N_total * A)^0.5 / (N_total - 1)
    numerator_of_factor = np.sqrt(N_total * Area)
    denominator_of_factor = N_total - 1
    
    normalization_factor = numerator_of_factor / denominator_of_factor
    
    # 6. 计算最终的 m_bar
    if normalization_factor == 0:
        m_bar = 0
    else:
        m_bar = mean_edge_length / normalization_factor

    # 打包中间结果以便检查
    debug_info = {
        'N_total': N_total,
        'Sum_Edge_Length': total_edge_length,
        'Mean_Edge_Length': mean_edge_length, # 分子
        'Area_AU2': Area,
        'Normalization_Factor': normalization_factor, # 分母
        'R_cluster_AU': R_cluster
    }

    return m_bar, debug_info




# chance alignment probability calculation
def calculate_tobin_distances_plot_corrected(points, brightness, Z, hp_instance, cut_threshold=10000, plot_pairs=False):
    """
    根据 Tobin 2022 附录 A.2 逻辑，计算距离分布及其对应的真实存在概率。
    
    参数:
    - points: (n, 2) 坐标数组 (单位: AU)
    - brightness: (n,) 亮度数组
    - Z: linkage 矩阵 (method='centroid' 或 'average')
    - hp_instance: 已经调用过 run() 且收敛的 HierarchyProbability 实例
    - cut_threshold: 距离截断阈值 (默认 10000 AU)
    - plot_pairs: 是否绘图
    
    返回:
    - logged_distances: 合并分离度数组 (Z[:, 2])
    - logged_probs: 对应的合并概率
    - logged_pairs: 用于绘图的最亮星连接对
    """
    n_stars = len(points)
    
    # 从 hp_instance 获取我们需要的概率和节点亮度信息
    final_probs = hp_instance.final_probs
    node_max_flux = hp_instance.node_max_flux
    
    logged_distances =[]
    logged_probs = []
    logged_pairs =[]
    
    valid_nodes = set(range(n_stars))
    
    # 用于绘图：记录每个节点（无论是叶子还是簇）内最亮星的原始索引
    brightest_in_cluster = {i: i for i in range(n_stars)}

    # 辅助函数：递归获取某个节点下的所有原始单星 (Leaves) 索引
    def get_leaves(node_id):
        if node_id < n_stars:
            return [node_id]
        else:
            row = node_id - n_stars
            return get_leaves(int(Z[row, 0])) + get_leaves(int(Z[row, 1]))

    # --- 遍历合并树 ---
    for i, row in enumerate(Z):
        dist = row[2]
        new_cluster_idx = n_stars + i
        left_idx = int(row[0])
        right_idx = int(row[1])
        
        # 如果任一子节点在此前被跳过，则当前节点也跳过
        if left_idx not in valid_nodes or right_idx not in valid_nodes:
            continue
            
        # 截断大于阈值的无物理意义的合并
        if dist > cut_threshold:
            continue
            
        valid_nodes.add(new_cluster_idx)
        
        # 1. 根据节点最大亮度，判断谁是主星分支，谁是伴星分支
        if node_max_flux[left_idx] >= node_max_flux[right_idx]:
            companion_idx = right_idx
        else:
            companion_idx = left_idx
            
        # 2. 计算这个距离对应的概率：伴星分支中所有单星的最小 final_prob
        companion_leaves = get_leaves(companion_idx)
        sep_prob = np.min(final_probs[companion_leaves])
        
        # 记录 Tobin 逻辑的距离和概率
        logged_distances.append(dist)
        logged_probs.append(sep_prob)

        # --- 以下为绘图准备逻辑 ---
        b1 = brightest_in_cluster[left_idx]
        b2 = brightest_in_cluster[right_idx]
        logged_pairs.append((b1, b2))

        # 更新新簇的最亮星信息
        if brightness[b1] >= brightness[b2]:
            brightest_in_cluster[new_cluster_idx] = b1
        else:
            brightest_in_cluster[new_cluster_idx] = b2

    logged_distances = np.array(logged_distances)
    logged_probs = np.array(logged_probs)

    # --- 绘图部分 ---
    if plot_pairs and len(logged_pairs) > 0:
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        
        # 绘制星星 (大小随亮度变化)
        sizes = brightness / np.max(brightness) * 100 + 20
        ax.scatter(points[:, 0], points[:, 1], s=sizes, c='none', edgecolors='k', zorder=3, label='Stars')
        
        # 绘制统计连线，用颜色的深浅或线条粗细表示概率大小
        for idx, (id_start, id_end) in enumerate(logged_pairs):
            p1 = points[id_start]
            p2 = points[id_end]
            prob = logged_probs[idx]
            
            # 概率越高，线越实；概率越低，线越虚/透明
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='blue', alpha=prob*0.8 + 0.2, linewidth=1.5 * prob + 0.5)
            
            # 可选：在连线中点标出概率
            mid_x, mid_y = (p1[0] + p2[0])/2, (p1[1] + p2[1])/2
            ax.text(mid_x, mid_y, f"p={prob:.2f}", color='red', fontsize=8)
            
        title_str = f"Tobin (2022) Probabilistic Mergers\nTotal Valid Pairs: {len(logged_pairs)}"
        ax.set_title(title_str)
        ax.set_xlabel("X (AU)")
        ax.set_ylabel("Y (AU)")
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    return logged_distances, logged_probs, logged_pairs
def calculate_tobin_distances_plot_corrected_noprob(points, brightness, Z, cut_threshold=10000, plot_pairs=False):
    """
    根据 Tobin 2022 附录 A.2 逻辑，计算距离分布，不带概率。
    
    参数:
    - points: (n, 2) 坐标数组 (单位: AU)
    - brightness: (n,) 亮度数组
    - Z: linkage 矩阵 (method='centroid' 或 'average')
    - cut_threshold: 距离截断阈值 (默认 10000 AU)
    - plot_pairs: 是否绘图
    
    返回:
    - logged_distances: 合并分离度数组 (Z[:, 2])
    - logged_pairs: 用于绘图的最亮星连接对
    """
    n_stars = len(points)
    
    logged_distances =[]
    logged_pairs =[]
    
    valid_nodes = set(range(n_stars))
    
    # 用于绘图：记录每个节点（无论是叶子还是簇）内最亮星的原始索引
    brightest_in_cluster = {i: i for i in range(n_stars)}

    # --- 遍历合并树 ---
    for i, row in enumerate(Z):
        dist = row[2]
        new_cluster_idx = n_stars + i
        left_idx = int(row[0])
        right_idx = int(row[1])
        
        # 如果任一子节点在此前被跳过，则当前节点也跳过
        if left_idx not in valid_nodes or right_idx not in valid_nodes:
            continue
        
        # 截断大于阈值的无物理意义的合并
        if dist > cut_threshold:
            continue
            
        valid_nodes.add(new_cluster_idx)
        
        # 记录 Tobin 逻辑的距离
        logged_distances.append(dist)

        # --- 以下为绘图准备逻辑 ---
        b1 = brightest_in_cluster[left_idx]
        b2 = brightest_in_cluster[right_idx]
        logged_pairs.append((b1, b2))

        # 更新新簇的最亮星信息
        if brightness[b1] >= brightness[b2]:
            brightest_in_cluster[new_cluster_idx] = b1
        else:
            brightest_in_cluster[new_cluster_idx] = b2

    logged_distances = np.array(logged_distances)

    # --- 绘图部分 ---
    if plot_pairs and len(logged_pairs) > 0:
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        
        # 绘制星星 (大小随亮度变化)
        sizes = brightness / np.max(brightness) * 100 + 20
        ax.scatter(points[:, 0], points[:, 1], s=sizes, c='none', edgecolors='k', zorder=3, label='Stars')
        
        # 绘制统计连线
        for idx, (id_start, id_end) in enumerate(logged_pairs):
            p1 = points[id_start]
            p2 = points[id_end]
            
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='blue', alpha=0.5, linewidth=1)
            
        title_str = f"Tobin (2022) Distances\nTotal Valid Pairs: {len(logged_pairs)}"
        ax.set_title(title_str)
        ax.set_xlabel("X (AU)")
        ax.set_ylabel("Y (AU)")
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    return logged_distances, logged_pairs


class ProjectionBayesianUnfolder_v1:
    def __init__(self, s_array_intial, bins=np.logspace(1.5, 4.5, 25), show_initial=True):
        self.s_array = s_array_intial
        self.s_array_intial = s_array_intial
        self.bins = bins
        P_s = np.histogram(s_array_intial, bins=bins)[0]    
        P_s = P_s / np.sum(P_s) # 归一化
        self.P_s = P_s
        self.P_s_initial = P_s
        self.s_centers = (bins[:-1] + bins[1:]) / 2
        if show_initial:
            plt.figure()
            plt.plot(self.s_centers, self.P_s, 'ko-', label='Initial P(s)')
            plt.xscale('log')
            plt.xlabel('s (au)')
            plt.ylabel('P(s)')
            plt.title('Initial Guess for True 3D Separation Distribution')
            plt.legend()
            plt.show()
    
    def P_s_proj_given_s(self, s_proj, s_true):
        if s_proj >= s_true:
            return 0
        else:
            return s_proj / (s_true * np.sqrt(s_true**2 - s_proj**2)) 
            # return 1 - np.sqrt(1 - (s_proj / s_true)**2) # 这个是累积概率，前一个是概率密度函数
    
    def BFactor(self, s_proj):
        # 计算贝叶斯更新的分母：∫ P(s) P(s_proj | s) ds
        integrand = np.array([self.P_s_proj_given_s(s_proj, s) * self.P_s[i] for i, s in enumerate(self.s_centers)])
        return np.trapz(integrand, self.s_centers)

    def P_s_given_s_proj(self, s_proj, s_true):
        # 计算后验分布：P(s | s_proj) ∝ P(s_proj | s) P(s)
        interp1d_func = interp1d(self.s_centers, self.P_s, kind='cubic', fill_value="extrapolate")
        P_s_interp = interp1d_func(s_true)
        numerator = self.P_s_proj_given_s(s_proj, s_true) * P_s_interp
        denominator = self.BFactor(s_proj)
        return numerator / denominator if denominator > 0 else 0

    def P_lessthan_st_given_s_proj(self, s_proj, s_t):
        # 计算 P(s < s_t | s_proj) = ∫_0^{s_t} P(s | s_proj) ds
        s_grid = np.logspace(np.log10(s_proj), np.log10(s_t), 1000)
        integrand = np.array([self.P_s_given_s_proj(s_proj, s) for s in s_grid])
        return np.trapz(integrand, s_grid)

    def P_lessthan_st_given_s_proj_elegant(self, s_proj, s_t):
        """
        使用三角换元消除奇点的完美积分法
        """
        if s_proj >= s_t:
            return 0.0
            
        # 构造一个连续的先验概率分布函数
        # 注意：fill_value=0 表示如果 s 超出我们统计的网格，概率为 0
        P_s_interp = interp1d(self.s_centers, self.P_s, kind='linear', bounds_error=False, fill_value=0.0)
        
        # 积分上限角
        theta_max = np.arccos(s_proj / s_t)
        
        # 构建角度网格 (避免用梯形法，直接在无奇点的角度空间均匀撒点)
        theta_grid_num = np.linspace(0, theta_max, 500)
        theta_grid_den = np.linspace(0, np.pi/2 - 1e-5, 1000) # 接近 90 度时 s趋向无穷大
        
        # 把角度转回 s，查询先验概率
        s_grid_num = s_proj / np.cos(theta_grid_num)
        s_grid_den = s_proj / np.cos(theta_grid_den)
        
        # 在角度空间下，积分就是单纯的 P(s) 对 theta 积分！
        integrand_num = P_s_interp(s_grid_num)
        integrand_den = P_s_interp(s_grid_den)
        
        numerator = np.trapz(integrand_num, theta_grid_num)
        denominator = np.trapz(integrand_den, theta_grid_den)
        
        if denominator <= 0:
            return 0.0
            
        return min(numerator / denominator, 1.0) # 永远在 0~1 之间且严格单调

class ProjectionBayesianUnfolder_v2:
    def __init__(self, s_array_initial, bins=np.logspace(1.5, 4.5, 20), show_initial=True):
        """
        bins: 建议上限设置到星团的物理直径（比如 100,000 au = 0.5 pc），这样才能容纳所有的远端背景污染。
        """
        self.s_array_initial = s_array_initial
        self.bins = bins
        self.n_bins = len(bins) - 1
        
        # 计算 bin 的对数中点作为代表真实 3D 距离的 s_centers
        self.s_centers = np.sqrt(bins[:-1] * bins[1:])
        
        # 计算观测到的投影距离 2D 分布 O(R)
        counts, _ = np.histogram(s_array_initial, bins=bins)
        self.O_prob = counts / np.sum(counts) # 归一化的观测概率
        
        # 初始猜测：假设 3D 真实分布 P(s) 等于观测分布 O(R)
        self.P_s = np.copy(self.O_prob)
        
        # 核心突破：直接构建无奇点的精确解析转移矩阵 A[i, j]
        self.A = self._build_exact_analytical_matrix()
        
        if show_initial:
            plt.figure(figsize=(8, 5))
            plt.step(self.s_centers, self.O_prob, where='mid', label='Observed 2D $P(s_{proj})$', color='orange')
            plt.xscale('log')
            plt.xlabel('Separation (au)')
            plt.ylabel('Probability')
            plt.title('Initial State')
            plt.legend()
            plt.show()

    def _build_exact_analytical_matrix(self):
        """
        神仙代换：直接利用精确的解析 CDF，彻底消灭积分奇点！
        计算 A[i, j] = P(投影距离落在 bin i | 真实 3D 距离在 bin j 的中心)
        """
        A = np.zeros((self.n_bins, self.n_bins))
        
        # 解析累积分布函数：P(s_proj < R | s_true) = 1 - sqrt(1 - (R/s_true)^2)  (当 R <= s_true 时)
        def exact_cdf(R, s_true):
            if R >= s_true:
                return 1.0
            return 1.0 - np.sqrt(1.0 - (R / s_true)**2)

        for j, s_true in enumerate(self.s_centers):
            for i in range(self.n_bins):
                R_lower = self.bins[i]
                R_upper = self.bins[i+1]
                
                # 落在 bin i 内的概率 = CDF(R_upper) - CDF(R_lower)
                # 没有任何梯形积分，没有任何奇点，绝对精确！
                prob_in_bin = exact_cdf(R_upper, s_true) - exact_cdf(R_lower, s_true)
                A[i, j] = prob_in_bin
                
        return A

    def run_EM_iteration(self, max_iter=1000, tol=1e-5, show_result=True):
        """
        执行 EM 期望最大化迭代 (数学上等价于 Richardson-Lucy Deconvolution)
        让星团自己告诉你它的真实 3D 距离分布！
        """
        print("Starting EM Iteration for 3D Deprojection...")
        P_s = np.copy(self.O_prob)
        
        for i in range(max_iter):
            # 1. 给定当前的 3D 分布猜测，预测我们理应看到的 2D 投影分布
            O_pred = self.A @ P_s
            
            # 防止除以 0
            O_pred[O_pred == 0] = 1e-12 
            
            # 2. 计算观测值与预测值的比例反馈
            ratio = self.O_prob / O_pred
            
            # 3. 贝叶斯更新：将反馈通过转置矩阵传导回 3D 空间
            # 这行代码等价于你公式里那个复杂的贝叶斯分母展开和积分求和！
            P_s_new = P_s * (self.A.T @ ratio)
            
            # 确保严格归一化
            P_s_new /= np.sum(P_s_new)
            
            # 检查收敛条件
            diff = np.max(np.abs(P_s_new - P_s))
            P_s = P_s_new
            if diff < tol:
                print(f"Converged perfectly at iteration {i}!")
                break
                
        self.P_s = P_s # 保存收敛后的全局真实 3D 分布
        
        if show_result:
            plt.figure(figsize=(8, 5))
            plt.step(self.s_centers, self.O_prob, where='mid', label='Observed 2D (Apparent)', color='orange', alpha=0.6)
            plt.step(self.s_centers, self.P_s, where='mid', label='Inferred 3D (True)', color='blue', linewidth=2)
            plt.xscale('log')
            plt.xlabel('Separation (au)')
            plt.ylabel('Probability Density')
            plt.title('EM Iteration Result: 2D Projected vs 3D True')
            plt.legend()
            plt.show()
            
        return self.P_s

    def P_bound_given_sproj(self, s_proj, s_t):
        """
        终极判定函数：当我们观测到一个 s_proj，它真实 3D 距离 < s_t 的概率是多少？
        这个函数将用于你的 linkage 树剪枝！
        """
        # 找到 s_proj 落在哪一个 bin 里
        i = np.digitize(s_proj, self.bins) - 1
        if i < 0 or i >= self.n_bins:
            return 0.0 # 超出统计范围，视为不绑定
            
        # 根据贝叶斯定理，提取这个 s_proj 对应的整个真实 3D 距离的后验概率分布
        # P(s_j | s_proj_i) ∝ P(s_proj_i | s_j) * P(s_j)
        posterior_3d = self.A[i, :] * self.P_s
        sum_post = np.sum(posterior_3d)
        
        if sum_post <= 0:
            return 0.0
            
        posterior_3d /= sum_post # 归一化
        
        # 寻找 s_t 对应的边界
        # 把所有中心点 < s_t 的 bin 的概率加起来，就是它受到引力束缚的终极概率！
        bound_mask = self.s_centers < s_t
        P_bound = np.sum(posterior_3d[bound_mask])
        
        return P_bound
    
    def plot_3D_log10_distribution(self):
        """
        绘制 EM 算法算出的真实 3D 距离的 log10 直方图。
        这是超越 Tobin 的结果 (Tobin 只有 2D)。
        """
        # 将 bins 转换为对数空间
        log_bins = np.log10(self.bins)
        log_centers = (log_bins[:-1] + log_bins[1:]) / 2
        bin_widths = np.diff(log_bins)
        
        plt.figure(figsize=(8, 6))
        
        # 绘制直方图条形
        plt.bar(log_centers, self.P_s, width=bin_widths, 
                color='skyblue', edgecolor='black', alpha=0.8, align='center', label='Inferred 3D $P(s)$')
        
        # 可选：把观测的 2D 分布也画上去对比
        plt.step(log_bins, np.append(self.O_prob, self.O_prob[-1]), 
                 where='post', color='red', linestyle='--', linewidth=2, label='Observed 2D $O(R)$')
        
        plt.xlabel(r'$\log_{10}(\mathrm{True\ 3D\ Separation / au})$', fontsize=14)
        plt.ylabel('Probability Fraction', fontsize=14)
        plt.title('Reconstructed 3D Separation Distribution (Log10 Space)', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.show()

        return log_centers, self.P_s, bin_widths

    def get_3D_log10_density(self):
        """
        将 EM 算法内部的 Fraction 转换为严格的对数空间概率密度 (PDF)。
        满足积分守恒：sum(density * d_log_s) = 1.0
        """
        log_bins = np.log10(self.bins)
        bin_widths = np.diff(log_bins) # 每个 bin 的对数宽度
        
        # 核心转换：Density = Fraction / Bin_Width
        density_3d = self.P_s / bin_widths
        density_2d_obs = self.O_prob / bin_widths
        
        return log_bins, density_3d, density_2d_obs

    def plot_3D_log10_density(self):
        """
        绘制严格的 Density 直方图，完美对标理论 PDF
        """
        log_bins, density_3d, density_2d_obs = self.get_3D_log10_density()
        log_centers = (log_bins[:-1] + log_bins[1:]) / 2
        bin_widths = np.diff(log_bins)
        
        plt.figure(figsize=(8, 6))
        
        # 绘制反演后的 3D 真实 Density
        plt.bar(log_centers, density_3d, width=bin_widths, 
                color='skyblue', edgecolor='black', alpha=0.8, align='center', 
                label=r'3D PDF $\frac{dP}{d\log_{10}s}$ after iterations')
        
        # 绘制观测到的 2D 投影 Density 作为对比
        plt.step(log_bins, np.append(density_2d_obs, density_2d_obs[-1]), 
                 where='post', color='red', linestyle='--', linewidth=2, 
                 label=r'Observed 2D PDF')
        
        plt.xlabel(r'$\log_{10}(\mathrm{Separation / au})$', fontsize=14)
        plt.ylabel('Probability Density', fontsize=14) # 这里是严格的 Density
        # plt.title('Strict Probability Density Function (Log10 Space)', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.show()



def generate_mock_cluster_3d(N, sigma_median_pc2):
    """
    根据中位数表面密度，生成一个内部完全随机、无物理双星的 3D 均匀球体星团。
    """
    # 1. 计算等效物理半径 (单位: pc)
    R_pc = np.sqrt(N / (np.pi * sigma_median_pc2))
    
    # 2. 在 3D 球体内均匀随机撒点
    # 半径采样 (r^3 是均匀分布的，这样体积密度才均匀)
    r_pc = R_pc * np.cbrt(np.random.rand(N))
    
    # 角度采样
    theta = np.random.uniform(0, 2 * np.pi, N)
    cos_phi = np.random.uniform(-1, 1, N)
    sin_phi = np.sqrt(1 - cos_phi**2)
    
    # 转换为笛卡尔坐标 (单位: pc)
    x_pc = r_pc * sin_phi * np.cos(theta)
    y_pc = r_pc * sin_phi * np.sin(theta)
    z_pc = r_pc * cos_phi
    
    # 将单位转换为 AU (因为你的 linkage 习惯用 AU 计算)
    pc2au = u.pc.to(u.au)
    points_3d_au = np.vstack([x_pc, y_pc, z_pc]).T * pc2au
    
    return points_3d_au

def extract_chance_3d_separations(N_stars, sigma_median_pc2, real_fluxes, N_iterations=100,cut_threshold=np.inf):
    """
    进行大迭代：多次生成 Mock 星团，跑 3D linkage，收集所有的偶然合并距离。
    """
    all_mock_distances =[]
    
    print(f"Running {N_iterations} Mock Iterations to estimate P_chance(s_3D)...")
    for i in range(N_iterations):
        # 1. 生成无物理双星的 3D 坐标
        mock_points_3d = generate_mock_cluster_3d(N_stars, sigma_median_pc2)
        
        # 2. 随机打乱真实的 flux 赋予这些假星 (保留光度函数特征，消除空间相关性)
        mock_fluxes = np.random.permutation(real_fluxes)
        
        # 3. 在三维空间直接跑 linkage
        # 注意：这里算的是 3D 物理距离的聚类！
        Z_3d = linkage(mock_points_3d, method='centroid')
        
        # 4. 使用你之前写的提取函数 (需确保它能直接接收 3D points)
        # 假设你使用的是那个基于 Z 和 brightest 提取的函数 (这里仅演示调用逻辑)
        # mock_distances = calculate_tobin_distances_3D_compatible(
        #     mock_points_3d, mock_fluxes, Z_3d, cut_threshold=np.inf
        # )
        # mock_distances = calculate_tobin_distances_plot(mock_points_3d, mock_fluxes, Z_3d, cut_threshold=cut_threshold)[0]
        mock_distances = calculate_tobin_distances_plot_corrected_noprob(mock_points_3d, mock_fluxes, Z_3d, cut_threshold=cut_threshold)[0]
        
        all_mock_distances.extend(mock_distances)
        
    return np.array(all_mock_distances)

def get_final_2d_bound_probability(s_proj, log_bins, unfolder_A_matrix, P_bound_3D_array, P_s_total):
    """
    s_proj: 你 2D linkage 里的距离
    log_bins: 你的网格边界
    unfolder_A_matrix: 你之前运行 EM 算法的转移矩阵 (unfolder.A)
    P_bound_3D_array: 刚算出来的 3D 绑定概率
    P_s_total: 你的 density_3d_mine
    """
    i = np.digitize(s_proj, 10**log_bins) - 1
    if i < 0 or i >= len(log_bins)-1:
        return 0.0
        
    # 提取在已知 s_proj 的情况下，3D 真实距离 s 的后验分布
    # (这正是 EM 算法最伟大之处，它告诉你 2D 投影对应的 3D 来源分布)
    posterior_3d = unfolder_A_matrix[i, :] * P_s_total
    
    if np.sum(posterior_3d) == 0:
        return 0.0
        
    posterior_3d /= np.sum(posterior_3d) # 归一化
    
    # 终极全概率积分：把“3D后验”与“3D绑定概率”乘起来求和！
    P_final_bound = np.sum(posterior_3d * P_bound_3D_array)
    
    return P_final_bound

