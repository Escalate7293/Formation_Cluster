import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fclusterdata, linkage
import astropy.units as u
pc2au = 3600 * 180 / np.pi


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



def calculate_wilson_interval(mf, n_sys, z=1):
    """Wilson score interval used by the observational workflow."""
    if n_sys <= 0:
        return 0.0, 0.0

    denominator = 1 + (z**2) / n_sys
    center_adjusted = mf + (z**2) / (2 * n_sys)
    discriminant = (mf * (1 - mf) / n_sys) + (z**2) / (4 * n_sys**2)
    root_term = z * np.sqrt(discriminant)
    upper_bound = (center_adjusted + root_term) / denominator
    lower_bound = (center_adjusted - root_term) / denominator

    return lower_bound, upper_bound


def calculate_poisson_interval(cf, n_sys, z=1):
    """Poisson interval used for CF when companion counts are high."""
    if n_sys <= 0:
        return 0.0, 0.0

    standard_error = np.sqrt(cf / n_sys)
    margin_of_error = z * standard_error
    upper_bound = cf + margin_of_error
    lower_bound = max(0.0, cf - margin_of_error)

    return lower_bound, upper_bound


def P_companion_given_detection(d, P_input, Sigma_local, tau=0.75):
    """
    Tobin-style Bayesian probability that a detection is a true companion.

    Parameters
    ----------
    d : float
        Projected separation in pc.
    P_input : float
        Companion-frequency prior at the current separation.
    Sigma_local : float
        Surface density in pc^-2.
    tau : float
        Empirical factor used in the existing workflow.
    """
    true_prior = 1.0 - np.exp(-P_input)
    numerator = tau * true_prior
    denominator = tau * true_prior + (
        1 - np.exp(-tau * Sigma_local * np.pi * d**2)
    ) * (1 - tau * true_prior)
    # print(Sigma_local )
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def get_final_2d_bound_probability(
    s_proj,
    log_bins,
    unfolder_A_matrix,
    P_bound_3D_array,
    P_s_total,
):
    """
    Combine the 3D posterior for a projected separation with a 3D bound prior.

    All separations are in au. ``log_bins`` stores log10 bin edges.
    """
    i = np.digitize(s_proj, 10**log_bins) - 1
    if i < 0 or i >= len(log_bins) - 1:
        return 0.0

    posterior_3d = unfolder_A_matrix[i, :] * P_s_total
    sum_post = np.sum(posterior_3d)
    if sum_post <= 0:
        return 0.0

    posterior_3d = posterior_3d / sum_post
    return np.sum(posterior_3d * P_bound_3D_array)


class HierarchyProbability:
    """
    Lightweight copy of the hierarchy probability calculator used in observations.

    It intentionally lives here so simulation-side workflows do not have to import
    the heavier FITS/DB machinery in ``Cluster_property.py``.
    """

    def __init__(
        self,
        Z,
        fluxes,
        distance,
        CF_ori=0.2,
        Sigma=770,
        tau=0.5,
        CF_intep1d_instance=None,
        log_bins_mine=None,
        unfolder_A_matrix=None,
        P_bound_3D_array=None,
        density_3d_mine=None,
    ):
        self.Z = Z
        self.fluxes = fluxes
        self.n_stars = len(fluxes)
        self.n_nodes = self.n_stars + len(Z)
        self.distance = distance
        self.CF_ori = CF_ori
        self.Sigma = Sigma
        self.tau = tau
        self.CF_intep1d_instance = CF_intep1d_instance
        self.log_bins_mine = log_bins_mine
        self.unfolder_A_matrix = unfolder_A_matrix
        self.P_bound_3D_array = P_bound_3D_array
        self.density_3d_mine = density_3d_mine

        self.node_max_flux = np.zeros(self.n_nodes)
        self.final_probs = np.zeros(self.n_stars)
        self.node_max_flux[: self.n_stars] = self.fluxes

    def precompute_fluxes(self):
        for i in range(len(self.Z)):
            cluster_idx = self.n_stars + i
            left_idx = int(self.Z[i, 0])
            right_idx = int(self.Z[i, 1])
            self.node_max_flux[cluster_idx] = max(
                self.node_max_flux[left_idx],
                self.node_max_flux[right_idx],
            )

    def propagate_probability(self, node_idx, current_prob, P_cal_method="Tobin"):
        if node_idx < self.n_stars:
            self.final_probs[node_idx] = current_prob
            return

        row_idx = node_idx - self.n_stars
        left_idx = int(self.Z[row_idx, 0])
        right_idx = int(self.Z[row_idx, 1])
        dist = self.Z[row_idx, 2]
        d_pc = dist * 3600 * self.distance / pc2au
        d_au = dist * 3600 * self.distance

        if self.CF_intep1d_instance is None:
            if P_cal_method == "Tobin":
                p_link = P_companion_given_detection(
                    d_pc,
                    self.CF_ori,
                    self.Sigma,
                    tau=self.tau,
                )
            elif P_cal_method == "Mine":
                p_link = get_final_2d_bound_probability(
                    s_proj=d_au,
                    log_bins=self.log_bins_mine,
                    unfolder_A_matrix=self.unfolder_A_matrix,
                    P_bound_3D_array=self.P_bound_3D_array,
                    P_s_total=self.density_3d_mine,
                )
            else:
                raise ValueError("P_cal_method must be 'Tobin' or 'Mine'.")
        else:
            CF_this_dis = self.CF_intep1d_instance(d_au)
            p_link = P_companion_given_detection(
                d_pc,
                CF_this_dis,
                self.Sigma,
                tau=self.tau,
            )

        left_flux = self.node_max_flux[left_idx]
        right_flux = self.node_max_flux[right_idx]

        if left_flux >= right_flux:
            self.propagate_probability(left_idx, current_prob, P_cal_method=P_cal_method)
            self.propagate_probability(
                right_idx,
                current_prob * p_link,
                P_cal_method=P_cal_method,
            )
        else:
            self.propagate_probability(
                left_idx,
                current_prob * p_link,
                P_cal_method=P_cal_method,
            )
            self.propagate_probability(right_idx, current_prob, P_cal_method=P_cal_method)

    def run(self, P_cal_method="Tobin"):
        self.precompute_fluxes()
        root_idx = self.n_nodes - 1
        self.propagate_probability(root_idx, 1.0, P_cal_method=P_cal_method)
        return self.final_probs


def convert_positions_to_au(positions, position_unit="au", length_unit_au=None):
    """
    Convert 3D or 2D positions to au.

    ``position_unit='code'`` requires ``length_unit_au``.
    """
    positions = np.asarray(positions, dtype=float)
    unit = position_unit.lower()
    factors = {
        "au": 1.0,
        "pc": pc2au,
        "m": 1.0 / 1.495978707e11,
        "meter": 1.0 / 1.495978707e11,
        "meters": 1.0 / 1.495978707e11,
        "km": 1000.0 / 1.495978707e11,
    }
    if unit == "code":
        if length_unit_au is None:
            raise ValueError("position_unit='code' requires length_unit_au.")
        factor = float(length_unit_au)
    elif unit in factors:
        factor = factors[unit]
    else:
        raise ValueError("position_unit must be one of 'au', 'pc', 'm', 'km', or 'code'.")

    return positions * factor


def project_3d_positions(positions_3d, plane="xy"):
    """Return a 2D projection from 3D positions."""
    positions_3d = np.asarray(positions_3d, dtype=float)
    if positions_3d.ndim != 2 or positions_3d.shape[1] != 3:
        raise ValueError("positions_3d must have shape (N, 3).")

    plane_map = {
        "xy": (0, 1),
        "yz": (1, 2),
        "zx": (2, 0),
        "xz": (0, 2),
    }
    key = plane.lower()
    if key not in plane_map:
        raise ValueError("plane must be one of 'xy', 'yz', 'zx', or 'xz'.")

    i, j = plane_map[key]
    return positions_3d[:, [i, j]]


def calculate_local_surface_density_2d(points_au, kth_neighbor=11):
    """
    Estimate projected local surface densities in pc^-2.

    Uses the same n/(pi r_n^2) convention as the observational workflow, with
    n = kth_neighbor - 1 because the source itself is excluded.
    """
    points_au = np.asarray(points_au, dtype=float)
    if points_au.ndim != 2 or points_au.shape[1] != 2:
        raise ValueError("points_au must have shape (N, 2).")

    n_points = len(points_au)
    if n_points <= 1:
        return np.array([])

    from scipy.spatial import cKDTree

    k_query = min(kth_neighbor + 1, n_points)
    tree = cKDTree(points_au)
    distances_au, _ = tree.query(points_au, k=k_query)
    r_n_pc = distances_au[:, -1] / pc2au
    n_neighbors = k_query - 1

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = n_neighbors / (np.pi * r_n_pc**2)

    return sigma


def estimate_projected_surface_density_pc2(points_au, method="local_median"):
    """
    Return one projected surface density in pc^-2 for a simulated projection.

    ``local_median`` matches the spirit of Tobin-style local surface density.
    For small-N samples it falls back to the global circular area estimate.
    """
    points_au = np.asarray(points_au, dtype=float)
    if len(points_au) <= 1:
        return 0.0

    method = method.lower()
    if method == "local_median":
        sigma = calculate_local_surface_density_2d(points_au)
        finite = sigma[np.isfinite(sigma) & (sigma > 0)]
        if len(finite) > 0 and len(points_au) >= 12:
            return float(np.median(finite))
        method = "global"

    if method == "global":
        center = np.median(points_au, axis=0)
        radius_pc = np.max(np.linalg.norm(points_au - center, axis=1)) / pc2au
        if radius_pc <= 0:
            return np.inf
        return float(len(points_au) / (np.pi * radius_pc**2))

    raise ValueError("surface density method must be 'local_median' or 'global'.")

def standalone_multiplicity_analyse(
    x_array, 
    y_array, 
    imfit_flux_array, 
    thresh_multi_au=1000, 
    length_u2au=3600*1000, # default 3.6e6 (assumes 1000pc and deg as length_u)
    show=True, 
    criterion='distance', 
    method='centroid', 
    return_numbers=False
):
    """
    Standalone version of multiplicity_analyse.
    Works on generic 2D points (x_array, y_array).
    """
    # --- 1. 准备坐标数据 ---
    sources_points = np.vstack([x_array, y_array]).T

    # --- 3. 运行 fclusterdata 并映射回结果 ---
    thresh_multi = thresh_multi_au / length_u2au 
    
    # 在点集上运行聚类
    if len(sources_points) == 0:
        labels = np.array([], dtype=int)
    elif len(sources_points) == 1:
        labels = np.array([1], dtype=int)
    else:
        labels = fclusterdata(sources_points, t=thresh_multi, criterion=criterion, method=method)

    if show:
        # 4. 可视化
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        for k in np.unique(labels):
            group = sources_points[labels == k]
            ax.scatter(group[:, 0], group[:, 1], s=5)
            
            # 画一个最小覆盖圆
            center = group.mean(axis=0)
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
    multiple_systems = [] 
    
    for k in unique_labels:
        group_mask = (labels == k)
        group = sources_points[group_mask]
        flux_this_group = imfit_flux_array[group_mask] 
        x_array_this_group = x_array[group_mask]
        y_array_this_group = y_array[group_mask]
    
        # 计算该组内的成员数
        num_in_group = len(group)

        if num_in_group >= 2:
            non_single += 1
            idx_primary = np.argmax(flux_this_group)
            x_primary = x_array_this_group[idx_primary]
            y_primary = y_array_this_group[idx_primary]
            
            distance_to_primary = np.sqrt((x_array_this_group - x_primary)**2 + 
                                        (y_array_this_group - y_primary)**2)
            
            multiple_this = {
                'num_members': num_in_group,
                'sources_index': group_mask,
                'distance_to_primary_au': distance_to_primary * length_u2au,
                'x_array': x_array_this_group,
                'y_array': y_array_this_group
            }
            multiple_systems.append(multiple_this)

        num_members = np.append(num_members, num_in_group)
    
    MF = non_single / len(unique_labels) if len(unique_labels) > 0 else 0
    MF_sigma_interval = calculate_wilson_interval(MF, len(unique_labels), z=1)

    companions = 0
    for i in num_members:
        companions += (i - 1)

    CF = companions / len(unique_labels) if len(unique_labels) > 0 else 0
    if CF > 0.5:
        CF_sigma_interval = calculate_poisson_interval(CF, len(unique_labels), z=1)
    else:
        CF_sigma_interval = calculate_wilson_interval(CF, len(unique_labels), z=1)

    if return_numbers:
        num_multiple_systems = non_single
        num_all = len(unique_labels)
        num_companions = companions
        return MF, CF, MF_sigma_interval, CF_sigma_interval, multiple_systems, num_multiple_systems, num_all, num_companions
    
    return MF, CF, MF_sigma_interval, CF_sigma_interval, multiple_systems


def standalone_multiplicity_analyse_contamination_corrected(
    x_array, 
    y_array, 
    imfit_flux_array, 
    thresh_multi_au=1000, 
    length_u2au=3600*1000, 
    CF_this_distance=None, 
    global_Sigma=1000, 
    tau=0.5,
    show=True, 
    criterion='distance', 
    method='centroid', 
    P_cal_method="Tobin",
    **kwargs
):
    """
    Standalone version of multiplicity_analyse_contamination_corrected.
    """
    # 依靠 length_u2au 解析 distance_pc，因为 pc = length_u2au / 3600 (若 length_u 为 deg)
    distance_pc = kwargs.get('distance_pc', length_u2au / 3600.0)
    
    # 获取 Mine 模式下需要的外部参数
    log_bins_mine = kwargs.get('log_bins_mine')
    unfolder_A_matrix = kwargs.get('unfolder_A_matrix')
    P_bound_3D_array = kwargs.get('P_bound_3D_array')
    density_3d_mine = kwargs.get('density_3d_mine')
    get_final_2d_bound_probability_func = kwargs.get(
        'get_final_2d_bound_probability_func',
        get_final_2d_bound_probability,
    )

    # --- 1. 准备坐标数据 ---
    sources_points = np.vstack([x_array, y_array]).T

    # --- 3. 运行 fclusterdata 并映射回结果 ---
    thresh_multi = thresh_multi_au / length_u2au 
    
    # 在加权后的点集上运行聚类
    if len(sources_points) == 0:
        labels = np.array([], dtype=int)
    elif len(sources_points) == 1:
        labels = np.array([1], dtype=int)
    else:
        labels = fclusterdata(sources_points, t=thresh_multi, criterion=criterion, method=method)

    if show:
        # 4. 可视化
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        for k in np.unique(labels):
            group = sources_points[labels == k]
            ax.scatter(group[:, 0], group[:, 1], s=5)
            
            # 画一个最小覆盖圆
            center = group.mean(axis=0) 
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
        flux_this_group = imfit_flux_array[group_mask] 
    
        num_in_group = len(group)
        final_systems_structure = []
        if num_in_group >= 2:                
            # 定义递归函数
            def resolve_system_structure(indices_subset):
                n_current = len(indices_subset)
                if n_current < 2:
                    return [1]

                sub_points = group[indices_subset]
                sub_fluxes = flux_this_group[indices_subset]
                Z_sub = linkage(sub_points, method=method)

                calculator = HierarchyProbability(
                    Z_sub, 
                    sub_fluxes, 
                    distance_pc, 
                    CF_ori=CF_this_distance, 
                    Sigma=global_Sigma, 
                    tau=tau,
                    log_bins_mine=log_bins_mine,
                    unfolder_A_matrix=unfolder_A_matrix,
                    P_bound_3D_array=P_bound_3D_array,
                    density_3d_mine=density_3d_mine,
                )
                probs = calculator.run(P_cal_method=P_cal_method)
                sum_probs = np.sum(probs)
                n_effective = int(round(sum_probs))
                
                if n_effective < 1: n_effective = 1

                # --- Tobin Logic 判断开始 ---
                if n_effective == n_current:
                    return [n_effective] 
                else:
                    min_p_link = 1.0
                    split_node_idx = -1

                    for i in range(len(Z_sub)):
                        dist_deg = Z_sub[i, 2]
                        # 对于 Z_sub 中的距离，用 length_u2au 转成 AU
                        dist_au = dist_deg * length_u2au
                        
                        if P_cal_method == "Tobin":
                            p_val = P_companion_given_detection(
                                dist_au / pc2au, 
                                CF_this_distance, 
                                global_Sigma, 
                                tau=tau
                            )
                        elif P_cal_method == "Mine":
                            if get_final_2d_bound_probability_func is None:
                                raise ValueError("get_final_2d_bound_probability_func is required in kwargs when P_cal_method='Mine'")
                            p_val = get_final_2d_bound_probability_func(
                                s_proj=dist_au, 
                                log_bins=log_bins_mine, 
                                unfolder_A_matrix=unfolder_A_matrix, 
                                P_bound_3D_array=P_bound_3D_array, 
                                P_s_total=density_3d_mine
                            )
                        
                        if p_val < min_p_link:
                            min_p_link = p_val
                            split_node_idx = i 

                    # === 决定拆分 (Split) ===
                    idx_left_cluster = int(Z_sub[split_node_idx, 0])
                    idx_right_cluster = int(Z_sub[split_node_idx, 1])
                    
                    def get_leaves_from_Z_idx(cluster_idx, n_leafs, Z_matrix):
                        if cluster_idx < n_leafs:
                            return [cluster_idx]
                        else:
                            row = cluster_idx - n_leafs
                            return get_leaves_from_Z_idx(int(Z_matrix[row, 0]), n_leafs, Z_matrix) + \
                                    get_leaves_from_Z_idx(int(Z_matrix[row, 1]), n_leafs, Z_matrix)

                    left_indices_local = get_leaves_from_Z_idx(idx_left_cluster, n_current, Z_sub)
                    right_indices_local = get_leaves_from_Z_idx(idx_right_cluster, n_current, Z_sub)
                    
                    left_indices_global = [indices_subset[x] for x in left_indices_local]
                    right_indices_global = [indices_subset[x] for x in right_indices_local]

                    involved_in_split = set(left_indices_global + right_indices_global)
                    
                    everything_else_global =[x for x in indices_subset if x not in involved_in_split]
                    
                    ans = resolve_system_structure(left_indices_global)
                    ans += resolve_system_structure(right_indices_global)
                    
                    if len(everything_else_global) > 0:
                        ans += resolve_system_structure(everything_else_global)
                        
                    return ans

            final_systems_structure = resolve_system_structure(list(range(num_in_group)))
        
        else:
            final_systems_structure = [1]

        systems += len(final_systems_structure)

        for n_members in final_systems_structure:
            if n_members >= 2:
                non_single += 1
            companions += (n_members - 1)
    
    MF = non_single / systems if systems > 0 else 0
    MF_sigma_interval = calculate_wilson_interval(MF, systems, z=1)

    CF = companions / systems if systems > 0 else 0
    if CF > 0.5:
        CF_sigma_interval = calculate_poisson_interval(CF, systems, z=1)
    else:
        CF_sigma_interval = calculate_wilson_interval(CF, systems, z=1)

    return MF, CF, MF_sigma_interval, CF_sigma_interval


# Mine correction method
def calculate_tobin_distances_plot(points, brightness, Z, cut_threshold=None, plot_pairs=False):
    """
    根据 Tobin et al. (2016) 的思想计算层级聚类中的距离分布，并可选绘制连线图。
    
    逻辑说明:
    - Case 1 (Leaf-Leaf): 连接两颗孤立星。
    - Case 2 (Leaf-Branch): 连接孤立星与多星系统的主星。
    - Case 3 (Branch-Branch): 仅连接两个多星系统各自的主星。
    
    参数:
    - points: (n, 2) 坐标数组
    - brightness: (n,) 亮度数组
    - Z: linkage 矩阵
    - cut_threshold: 距离截断阈值
    - plot_pairs: 是否绘图 (True/False)
    
    返回:
    - logged_distances: 统计到的距离列表
    - logged_pairs: 统计到的连接点对 (id1, id2)
    """
    n = len(points)
    dist_matrix = squareform(pdist(points))
    if cut_threshold is None:
        cut_threshold = np.inf
    
    logged_distances = []
    logged_pairs = []
    invalid_nodes = set()
    
    # 初始化：对于叶子节点，最亮星就是它自己
    # cluster_members 在此处仅用于维护成员列表（如果需要），本逻辑主要依赖 brightest_in_cluster
    brightest_in_cluster = {i: i for i in range(n)}

    # --- 遍历合并树 ---
    for i, row in enumerate(Z):
        new_cluster_idx = n + i
        idx1, idx2 = int(row[0]), int(row[1])
        merge_dist = row[2]
        
        # 获取两个簇的主星 ID
        # 注意：如果是孤立星(Leaf)，brightest_in_cluster[leaf] 就是 leaf 本身
        b1 = brightest_in_cluster[idx1]
        b2 = brightest_in_cluster[idx2]

        # --- 级联判断逻辑 ---
        # 如果任一子分支已失效，或当前合并距离超限，均视为无效合并
        if idx1 in invalid_nodes or idx2 in invalid_nodes or merge_dist > cut_threshold:
            invalid_nodes.add(new_cluster_idx)
        else:
            # 存活且合规的分支才记录画图及记录距离
            dist = dist_matrix[b1, b2]
            logged_distances.append(dist)
            logged_pairs.append((b1, b2))

        # --- 更新新簇的信息 ---
        # 比较两个子簇主星的亮度，取更亮的作为新簇的主星
        if brightness[b1] >= brightness[b2]:
            brightest_in_cluster[new_cluster_idx] = b1
        else:
            brightest_in_cluster[new_cluster_idx] = b2

    # --- 绘图部分 ---
    if plot_pairs:
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        
        # 1. 绘制星星 (大小随亮度变化)
        # s 参数做了简单的线性映射以便观察: brightness * scale + base_size
        sizes = brightness * 15 + 30
        ax.scatter(points[:, 0], points[:, 1], s=sizes, c='none', edgecolors='k', zorder=3, label='Stars')
        
        # # 2. 标记 ID
        # for idx, (x, y) in enumerate(points):
        #     ax.text(x, y + 0.02, str(idx), fontsize=9, ha='center', zorder=4)
            
        # 3. 绘制统计的连线
        # 使用集合去重，防止视觉上画多次（虽然逻辑上 Tobin 每一层级只算一次，不会有完全重复的层级对）
        for (id_start, id_end) in logged_pairs:
            p1 = points[id_start]
            p2 = points[id_end]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', alpha=0.6, linewidth=1.5)
            
        title_str = f"Tobin (2016) Logic Visualization\nTotal Pairs: {len(logged_pairs)}"
        if cut_threshold < np.inf:
            title_str += f" (Cut < {cut_threshold})"
            
        ax.set_title(title_str)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.show()

    return logged_distances, logged_pairs

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

# 尾部匹配
def decouple_true_binaries(log_bins, density_3d_total, mock_distances_3d, tail_threshold_au=5000):
    """
    通过尾部匹配(Tail-matching)解耦真实双星分布与Mock背景分布
    
    参数:
    - log_bins: 你之前的 log_bins_mine
    - density_3d_total: 你之前的 density_3d_mine (EM反演后的3D总分布)
    - mock_distances_3d: 你生成的 mock 3D 距离数组
    - tail_threshold_au: 物理阈值，认为大于这个距离的连接 100% 是背景污染
    """
    # 计算 log_centers 和 bin_widths
    log_centers = (log_bins[:-1] + log_bins[1:]) / 2
    bin_widths = np.diff(log_bins)
    
    # 1. 计算 Mock 数据的初始 Density
    # 注意要用相同的 bins 以保证对齐
    density_mock_raw, _ = np.histogram(np.log10(mock_distances_3d), bins=log_bins, density=True)
    
    # 2. 尾部匹配 (Tail-Matching Scaling)
    # 找到大于 tail_threshold_au 的索引区域
    tail_idx = np.searchsorted(log_centers, np.log10(tail_threshold_au))
    
    # 计算尾部区域的总面积 (或者直接算和，因为 bin_widths 是一样的)
    area_total_tail = np.sum(density_3d_total[tail_idx:] * bin_widths[tail_idx:])
    area_mock_tail = np.sum(density_mock_raw[tail_idx:] * bin_widths[tail_idx:])
    
    # 计算缩放系数 alpha
    if area_mock_tail > 0:
        alpha = area_total_tail / area_mock_tail
    else:
        alpha = 0.0
        print("警告: Mock数据在尾部没有点！请检查Mock星团的生成半径是否足够大。")
        
    print(f"计算出的背景缩放系数 Alpha = {alpha:.3f}")
    
    # 3. 缩放 Mock 分布，得到真实的污染背景
    density_mock_scaled = density_mock_raw * alpha
    
    # 4. 相减并处理负数 (核心操作)
    # 使用 np.maximum(0, ...) 强制将减出来的负数截断为 0。
    # 物理意义：在统计涨落允许的误差范围内，背景已经吃光了所有信号，没有真实双星的证据。
    density_true_bound = np.maximum(0, density_3d_total - density_mock_scaled)
    
    # # 5. 计算每一段 3D 距离下，是“真实双星”的后验概率！
    # P_bound_3D = np.zeros_like(density_3d_total)
    # valid_mask = density_3d_total > 0
    # # 概率 = 真双星密度 / 总密度
    # P_bound_3D[valid_mask] = density_true_bound[valid_mask] / density_3d_total[valid_mask]
    
    # # 防止因为数值计算浮点数问题超过 1.0
    # P_bound_3D = np.clip(P_bound_3D, 0.0, 1.0)

    # =========================================================
    # 🌟 新增：物理先验强制清零 (Physical Truncation)
    # 既然我们认定 tail_threshold_au 之外全是背景，
    # 那就把这个距离之外的 true_bound 强行抹平，消除统计噪声的残留！
    # =========================================================
    tail_mask = log_centers > np.log10(tail_threshold_au)
    density_true_bound[tail_mask] = 0.0

    # 5. 计算每一段 3D 距离下，是“真实双星”的后验概率！
    P_bound_3D = np.zeros_like(density_3d_total)
    valid_mask = density_3d_total > 0
    P_bound_3D[valid_mask] = density_true_bound[valid_mask] / density_3d_total[valid_mask]
    
    P_bound_3D = np.clip(P_bound_3D, 0.0, 1.0)
    
    # --- 绘图验证 (极其重要，一定要看这个图！) ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    # 画出总分布 (蓝色)
    ax.step(log_bins, np.append(density_3d_total, density_3d_total[-1]), 
             where='post', color='black', linewidth=2, label='Total 3D Distribution')
    
    # 画出缩放后的 Mock 背景 (红色虚线)
    ax.step(log_bins, np.append(density_mock_scaled, density_mock_scaled[-1]), 
             where='post', color='red', linestyle='--', linewidth=2, label='Mock Background')
    
    # 画出相减后得到的纯净物理双星分布 (绿色填充)
    ax.fill_between(log_centers, 0, density_true_bound, step='mid', 
                     color='green', alpha=0.5, label='Bound Multiples')
                     
    # 画一条尾部对齐的垂直参考线
    ax.axvline(np.log10(tail_threshold_au), color='gray', linestyle=':', label=f'Bound Threshold ({tail_threshold_au} au)')
    
    ax.set_xlabel(r'$\log_{10} \left [{\rm Separation}~\mathrm{(au)} \right ]$', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    # ax.set_title('Decoupling True Multiples from Chance Alignments', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14, length=8, right=True, top=True, direction='in')
    ax.legend(fontsize=14)
    # ax.show()
    
    return P_bound_3D, density_true_bound, fig
