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
    def __init__(self, Z, fluxes, distance, CF_ori=0.2, Sigma=770, tau=0.5, CF_intep1d_instance=None):
        self.Z = Z
        self.fluxes = fluxes
        self.n_stars = len(fluxes)
        self.n_nodes = self.n_stars + len(Z) # 总节点数 (叶子 + 聚类)
        self.distance = distance
        self.CF_ori = CF_ori
        self.Sigma = Sigma
        self.tau = tau
        self.CF_intep1d_instance = CF_intep1d_instance


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
                p_link = get_final_2d_bound_probability(s_proj=d_au, log_bins=log_bins_mine, unfolder_A_matrix=aaa.A, P_bound_3D_array=P_bound_3D_array, P_s_total=density_3d_mine)
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
                                                     show=True,criterion='distance',method='centroid',P_cal_method="Tobin"):
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
                        tau=tau
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
                                    unfolder_A_matrix=aaa.A, 
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
                        
                        if min_p_link < 0.5:
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


                        else:
                            # === 不拆分 (Keep) ===
                            # 虽然成员减少了 (n_effective < n_current)，但最弱的连接 P >= 0.5
                            # 说明这是“累积概率损失” (Death by a thousand cuts)
                            # 这种情况下，我们接受 n_effective 作为该系统的最终成员数
                            return [n_effective]

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