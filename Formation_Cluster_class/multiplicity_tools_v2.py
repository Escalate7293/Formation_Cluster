"""Small standalone tools for multiplicity statistics."""

import numpy as np
from scipy.cluster.hierarchy import fclusterdata, linkage
from scipy.special import erf

PC_TO_AU = 3600.0 * 180.0 / np.pi


def calculate_wilson_interval(value, sample_size, z=1.0):
    """Return the Wilson interval for a fraction."""
    denominator = 1.0 + z**2 / sample_size
    center = value + z**2 / (2.0 * sample_size)
    spread = z * np.sqrt(
        value * (1.0 - value) / sample_size
        + z**2 / (4.0 * sample_size**2)
    )
    return (center - spread) / denominator, (center + spread) / denominator


def calculate_poisson_interval(value, sample_size, z=1.0):
    """Return the normal Poisson interval used for CF above 0.5."""
    spread = z * np.sqrt(value / sample_size)
    return max(0.0, value - spread), value + spread


def multiplicity_analyse(
    points_au,
    separation_threshold_au=1000.0,
    method="centroid",
    source_ids=None,
):
    """Calculate uncorrected MF and CF from projected positions in au."""
    points_au = np.asarray(points_au, dtype=float)
    if points_au.ndim != 2 or points_au.shape[1] != 2:
        raise ValueError("points_au must have shape (N, 2).")

    n_sources = len(points_au)
    if source_ids is None:
        source_ids = np.arange(n_sources)
    source_ids = np.asarray(source_ids)
    if len(source_ids) != n_sources:
        raise ValueError("source_ids and points_au must have the same length.")

    if n_sources == 0:
        return {
            "MF": np.nan,
            "CF": np.nan,
            "MF_interval": (np.nan, np.nan),
            "CF_interval": (np.nan, np.nan),
            "n_sources": 0,
            "n_systems": 0,
            "n_multiple_systems": 0,
            "n_companions": 0,
            "order_counts": {},
            "labels": np.array([], dtype=int),
            "systems": [],
        }

    labels = (
        np.array([1], dtype=int)
        if n_sources == 1
        else fclusterdata(
            points_au,
            t=separation_threshold_au,
            criterion="distance",
            method=method,
        )
    )

    systems = []
    orders = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        orders.append(len(indices))
        systems.append(
            {
                "label": int(label),
                "order": len(indices),
                "indices": indices,
                "source_ids": source_ids[indices],
            }
        )

    orders = np.asarray(orders, dtype=int)
    n_systems = len(orders)
    n_multiple = int(np.sum(orders >= 2))
    n_companions = int(np.sum(orders - 1))
    mf = n_multiple / n_systems
    cf = n_companions / n_systems

    return {
        "MF": mf,
        "CF": cf,
        "MF_interval": calculate_wilson_interval(mf, n_systems),
        "CF_interval": (
            calculate_poisson_interval(cf, n_systems)
            if cf > 0.5
            else calculate_wilson_interval(cf, n_systems)
        ),
        "n_sources": n_sources,
        "n_systems": n_systems,
        "n_multiple_systems": n_multiple,
        "n_companions": n_companions,
        "order_counts": {
            int(order): int(np.sum(orders == order)) for order in np.unique(orders)
        },
        "labels": labels,
        "systems": systems,
    }


def P_companion_given_detection(
    d,
    P_input,
    Sigma_local,
    tau=0.75,
    prior_mapping="poisson",
):
    """Return the Tobin companion probability at one projected separation."""
    if prior_mapping == "poisson":
        companion_prior = 1.0 - np.exp(-P_input)
    elif prior_mapping == "identity":
        companion_prior = P_input
    else:
        raise ValueError("prior_mapping must be 'poisson' or 'identity'.")

    numerator = tau * companion_prior
    chance_probability = 1.0 - np.exp(
        -tau * Sigma_local * np.pi * d**2
    )
    denominator = numerator + chance_probability * (1.0 - numerator)
    return numerator / denominator if denominator > 0 else 0.0


class HierarchyProbability:
    """Propagate Tobin probabilities from a linkage root to its leaves."""

    def __init__(
        self,
        Z,
        fluxes,
        distance=None,
        CF_ori=0.2,
        Sigma=770.0,
        tau=0.5,
        prior_mapping="poisson",
        z_distance_unit="au",
    ):
        self.Z = np.asarray(Z, dtype=float)
        self.fluxes = np.asarray(fluxes, dtype=float)
        self.distance = distance
        self.CF_ori = CF_ori
        self.Sigma = Sigma
        self.tau = tau
        self.prior_mapping = prior_mapping
        self.z_distance_unit = z_distance_unit
        self.n_stars = len(self.fluxes)
        self.n_nodes = self.n_stars + len(self.Z)
        self.node_max_flux = np.zeros(self.n_nodes)
        self.final_probs = np.zeros(self.n_stars)
        self.link_probability_by_node = {}
        self.node_max_flux[: self.n_stars] = self.fluxes

    def precompute_fluxes(self):
        """Calculate the brightest leaf in every branch, bottom-up."""
        for row_index, row in enumerate(self.Z):
            node_index = self.n_stars + row_index
            left_index, right_index = int(row[0]), int(row[1])
            self.node_max_flux[node_index] = max(
                self.node_max_flux[left_index], self.node_max_flux[right_index]
            )

    def distance_au(self, linkage_distance):
        if self.z_distance_unit == "au":
            return float(linkage_distance)
        if self.z_distance_unit == "degree":
            return float(linkage_distance * 3600.0 * self.distance)
        raise ValueError("z_distance_unit must be 'au' or 'degree'.")

    def propagate_probability(self, node_index, current_probability):
        """Propagate probabilities from the root to all leaves."""
        if node_index < self.n_stars:
            self.final_probs[node_index] = current_probability
            return

        row_index = node_index - self.n_stars
        left_index = int(self.Z[row_index, 0])
        right_index = int(self.Z[row_index, 1])
        distance_pc = self.distance_au(self.Z[row_index, 2]) / PC_TO_AU
        link_probability = P_companion_given_detection(
            distance_pc,
            self.CF_ori,
            self.Sigma,
            tau=self.tau,
            prior_mapping=self.prior_mapping,
        )
        self.link_probability_by_node[node_index] = link_probability

        if self.node_max_flux[left_index] >= self.node_max_flux[right_index]:
            self.propagate_probability(left_index, current_probability)
            self.propagate_probability(
                right_index, current_probability * link_probability
            )
        else:
            self.propagate_probability(
                left_index, current_probability * link_probability
            )
            self.propagate_probability(right_index, current_probability)

    def run(self, P_cal_method="Tobin"):
        if P_cal_method != "Tobin":
            raise ValueError("This standalone class supports P_cal_method='Tobin'.")
        self.final_probs.fill(0.0)
        self.link_probability_by_node = {}
        self.precompute_fluxes()
        self.propagate_probability(self.n_nodes - 1, 1.0)
        return self.final_probs


def multiplicity_analyse_tobin_once(
    points_au,
    flux_proxy,
    cf_prior,
    sigma_local_pc2,
    separation_threshold_au=1000.0,
    tau=0.45,
    prior_mapping="poisson",
    method="centroid",
    source_ids=None,
):
    """Calculate Tobin-corrected MF and CF for one fixed CF prior."""
    points_au = np.asarray(points_au, dtype=float)
    flux_proxy = np.asarray(flux_proxy, dtype=float)
    if points_au.ndim != 2 or points_au.shape[1] != 2:
        raise ValueError("points_au must have shape (N, 2).")
    if len(flux_proxy) != len(points_au):
        raise ValueError("flux_proxy and points_au must have the same length.")
    if source_ids is None:
        source_ids = np.arange(len(points_au))
    source_ids = np.asarray(source_ids)

    raw = multiplicity_analyse(
        points_au,
        separation_threshold_au,
        method=method,
        source_ids=source_ids,
    )
    if len(points_au) == 0:
        return {**raw, "CF_input": cf_prior}

    final_systems = []

    def get_leaves(node_index, n_leaves, Z):
        if node_index < n_leaves:
            return [node_index]
        row_index = node_index - n_leaves
        return get_leaves(int(Z[row_index, 0]), n_leaves, Z) + get_leaves(
            int(Z[row_index, 1]), n_leaves, Z
        )

    def resolve(indices):
        if len(indices) < 2:
            return [indices]

        Z = linkage(points_au[indices], method=method)
        hierarchy = HierarchyProbability(
            Z,
            flux_proxy[indices],
            CF_ori=cf_prior,
            Sigma=sigma_local_pc2,
            tau=tau,
            prior_mapping=prior_mapping,
            z_distance_unit="au",
        )
        effective_members = max(int(round(hierarchy.run().sum())), 1)
        if effective_members == len(indices):
            return [indices]

        split_row = min(
            range(len(Z)),
            key=lambda row: hierarchy.link_probability_by_node[len(indices) + row],
        )
        left_local = get_leaves(int(Z[split_row, 0]), len(indices), Z)
        right_local = get_leaves(int(Z[split_row, 1]), len(indices), Z)
        left = [indices[index] for index in left_local]
        right = [indices[index] for index in right_local]
        involved = set(left + right)
        remaining = [index for index in indices if index not in involved]
        return resolve(left) + resolve(right) + (resolve(remaining) if remaining else [])

    for raw_system in raw["systems"]:
        final_systems.extend(resolve(raw_system["indices"].tolist()))

    orders = np.array([len(indices) for indices in final_systems], dtype=int)
    n_systems = len(orders)
    n_multiple = int(np.sum(orders >= 2))
    n_companions = int(np.sum(orders - 1))
    mf = n_multiple / n_systems
    cf = n_companions / n_systems

    return {
        "MF": mf,
        "CF": cf,
        "MF_interval": calculate_wilson_interval(mf, n_systems),
        "CF_interval": (
            calculate_poisson_interval(cf, n_systems)
            if cf > 0.5
            else calculate_wilson_interval(cf, n_systems)
        ),
        "CF_input": cf_prior,
        "n_sources": len(points_au),
        "n_systems": n_systems,
        "n_multiple_systems": n_multiple,
        "n_companions": n_companions,
        "order_counts": {
            int(order): int(np.sum(orders == order)) for order in np.unique(orders)
        },
        "labels": raw["labels"],
        "systems": [
            {
                "order": len(indices),
                "indices": np.asarray(indices),
                "source_ids": source_ids[indices],
            }
            for indices in final_systems
        ],
    }


def multiplicity_analyse_tobin_iterative(
    points_au,
    flux_proxy,
    sigma_local_pc2,
    separation_threshold_au=1000.0,
    tau=0.45,
    prior_mapping="poisson",
    initial_cf=None,
    tolerance=0.005,
    max_iterations=10,
    method="centroid",
    source_ids=None,
):
    """Iterate the Tobin CF prior until the corrected CF converges."""
    if initial_cf is None:
        initial_cf = multiplicity_analyse(
            points_au,
            separation_threshold_au,
            method=method,
            source_ids=source_ids,
        )["CF"]

    current_cf = initial_cf
    history = []
    result = None
    for iteration in range(1, max_iterations + 1):
        result = multiplicity_analyse_tobin_once(
            points_au,
            flux_proxy,
            current_cf,
            sigma_local_pc2,
            separation_threshold_au,
            tau,
            prior_mapping,
            method,
            source_ids,
        )
        history.append(
            {
                "iteration": iteration,
                "CF_input": current_cf,
                "MF_output": result["MF"],
                "CF_output": result["CF"],
            }
        )
        if abs(result["CF"] - current_cf) < tolerance:
            break
        current_cf = result["CF"]

    return {
        **result,
        "initial_CF": initial_cf,
        "final_CF_input": history[-1]["CF_input"],
        "n_iterations": len(history),
        "converged": abs(history[-1]["CF_output"] - history[-1]["CF_input"])
        < tolerance,
        "history": history,
    }


def build_loguniform_projection_setup(bins):
    """Build the fiducial log-uniform 3D prior and projection matrix."""
    bins = np.asarray(bins, dtype=float)
    if bins.ndim != 1 or len(bins) < 2 or np.any(bins <= 0):
        raise ValueError("bins must be a positive one-dimensional array.")
    if np.any(np.diff(bins) <= 0):
        raise ValueError("bins must be strictly increasing.")

    log_widths = np.diff(np.log10(bins))
    s_centers = np.sqrt(bins[:-1] * bins[1:])
    projection_matrix = np.zeros((len(s_centers), len(s_centers)))

    def exact_cdf(projected_radius, true_distance):
        if projected_radius >= true_distance:
            return 1.0
        return 1.0 - np.sqrt(1.0 - (projected_radius / true_distance) ** 2)

    for true_index, true_distance in enumerate(s_centers):
        for projected_index, (lower, upper) in enumerate(
            zip(bins[:-1], bins[1:])
        ):
            projection_matrix[projected_index, true_index] = (
                exact_cdf(upper, true_distance)
                - exact_cdf(lower, true_distance)
            )

    return {
        "bins": bins,
        "s_centers": s_centers,
        "log_widths": log_widths,
        "prior_mass": log_widths / log_widths.sum(),
        "projection_matrix": projection_matrix,
    }


def posterior_3d_given_projected_distance(
    s_proj_au,
    projection_setup,
    clip_to_bins=True,
):
    """Return discrete P(s_3D | s_proj) from a projection setup."""
    bins = np.asarray(projection_setup["bins"], dtype=float)
    prior_mass = np.asarray(projection_setup["prior_mass"], dtype=float)
    projection_matrix = np.asarray(
        projection_setup["projection_matrix"], dtype=float
    )
    expected_shape = (len(bins) - 1, len(bins) - 1)
    if prior_mass.shape != (len(bins) - 1,):
        raise ValueError("prior_mass and bins have incompatible shapes.")
    if projection_matrix.shape != expected_shape:
        raise ValueError("projection_matrix and bins have incompatible shapes.")

    s_proj_used = float(s_proj_au)
    was_clipped = not (bins[0] <= s_proj_used < bins[-1])
    if was_clipped and not clip_to_bins:
        raise ValueError(
            f"s_proj_au={s_proj_au} is outside [{bins[0]}, {bins[-1]})."
        )
    s_proj_used = float(
        np.clip(
            s_proj_used,
            bins[0] * (1.0 + 1e-10),
            bins[-1] * (1.0 - 1e-10),
        )
    )

    projected_bin = np.digitize(s_proj_used, bins) - 1
    likelihood = projection_matrix[projected_bin]
    posterior_mass = likelihood * prior_mass
    evidence = posterior_mass.sum()
    if evidence <= 0:
        raise ValueError("The projection posterior has zero evidence.")
    posterior_mass = posterior_mass / evidence

    return {
        "s_proj_au": float(s_proj_au),
        "s_proj_used_au": s_proj_used,
        "was_clipped": was_clipped,
        "projected_bin": int(projected_bin),
        "s_centers": np.asarray(projection_setup["s_centers"], dtype=float),
        "prior_mass": prior_mass / prior_mass.sum(),
        "likelihood": likelihood,
        "posterior_mass": posterior_mass,
        "evidence": float(evidence),
    }


def p_bound_flux_proxy(
    m_left_proxy,
    m_right_proxy,
    s3d_au,
    m_cluster_proxy,
    r_cluster_au,
    virial_factor=0.2,
):
    """Return P(bound | s_3D) using flux as a relative mass proxy."""
    if m_cluster_proxy <= 0 or r_cluster_au <= 0 or virial_factor <= 0:
        raise ValueError(
            "m_cluster_proxy, r_cluster_au, and virial_factor must be positive."
        )

    s3d_au = np.asarray(s3d_au, dtype=float)
    mass_fraction = max(
        (m_left_proxy + m_right_proxy) / m_cluster_proxy,
        0.0,
    )
    x2 = (
        mass_fraction
        * r_cluster_au
        / (2.0 * virial_factor * np.maximum(s3d_au, 1e-30))
    )
    x = np.sqrt(np.maximum(x2, 0.0))
    probability = erf(x) - 2.0 / np.sqrt(np.pi) * x * np.exp(-x2)
    return np.clip(probability, 0.0, 1.0)


def p_bound_projected_flux_proxy(
    s_proj_au,
    m_left_proxy,
    m_right_proxy,
    m_cluster_proxy,
    r_cluster_au,
    projection_setup,
    virial_factor=0.2,
    clip_to_bins=True,
    return_details=False,
):
    """Marginalize P(bound | s_3D) over P(s_3D | s_proj)."""
    posterior = posterior_3d_given_projected_distance(
        s_proj_au,
        projection_setup,
        clip_to_bins=clip_to_bins,
    )
    p_bound_s3d = p_bound_flux_proxy(
        m_left_proxy,
        m_right_proxy,
        posterior["s_centers"],
        m_cluster_proxy,
        r_cluster_au,
        virial_factor=virial_factor,
    )
    weighted_mass = posterior["posterior_mass"] * p_bound_s3d
    probability = float(np.clip(weighted_mass.sum(), 0.0, 1.0))
    if not return_details:
        return probability

    return probability, {
        **posterior,
        "p_bound_s3d": p_bound_s3d,
        "weighted_probability_mass": weighted_mass,
        "p_bound_s_proj": probability,
    }


class HierarchyProbability_v2:
    """Propagate P(bound | s_proj) through one linkage tree."""

    def __init__(
        self,
        Z,
        fluxes,
        r_cluster_au,
        projection_setup,
        total_flux_proxy=None,
        virial_factor=0.2,
        z_distance_unit="au",
        distance_pc=None,
        clip_to_bins=True,
    ):
        self.Z = np.asarray(Z, dtype=float)
        self.fluxes = np.nan_to_num(
            np.asarray(fluxes, dtype=float),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self.r_cluster_au = float(r_cluster_au)
        self.projection_setup = projection_setup
        self.total_flux_proxy = (
            float(self.fluxes.sum())
            if total_flux_proxy is None
            else float(total_flux_proxy)
        )
        self.virial_factor = virial_factor
        self.z_distance_unit = z_distance_unit
        self.distance_pc = distance_pc
        self.clip_to_bins = clip_to_bins
        self.n_stars = len(self.fluxes)
        self.n_nodes = self.n_stars + len(self.Z)
        self.node_max_flux = np.zeros(self.n_nodes)
        self.node_flux_sum = np.zeros(self.n_nodes)
        self.final_probs = np.zeros(self.n_stars)
        self.link_probability_by_node = {}
        self.link_distance_au_by_node = {}
        self.node_max_flux[: self.n_stars] = self.fluxes
        self.node_flux_sum[: self.n_stars] = self.fluxes

    def precompute_fluxes(self):
        """Calculate branch flux sums and brightest members bottom-up."""
        for row_index, row in enumerate(self.Z):
            node_index = self.n_stars + row_index
            left_index, right_index = int(row[0]), int(row[1])
            self.node_flux_sum[node_index] = (
                self.node_flux_sum[left_index]
                + self.node_flux_sum[right_index]
            )
            self.node_max_flux[node_index] = max(
                self.node_max_flux[left_index],
                self.node_max_flux[right_index],
            )

    def distance_au(self, linkage_distance):
        if self.z_distance_unit == "au":
            return float(linkage_distance)
        if self.distance_pc is None:
            raise ValueError("distance_pc is required for angular linkage units.")
        if self.z_distance_unit == "arcsec":
            return float(linkage_distance * self.distance_pc)
        if self.z_distance_unit == "degree":
            return float(linkage_distance * 3600.0 * self.distance_pc)
        raise ValueError("z_distance_unit must be 'au', 'arcsec', or 'degree'.")

    def propagate_probability(self, node_index, current_probability):
        """Propagate probabilities from the root to all leaves."""
        if node_index < self.n_stars:
            self.final_probs[node_index] = current_probability
            return

        row_index = node_index - self.n_stars
        left_index = int(self.Z[row_index, 0])
        right_index = int(self.Z[row_index, 1])
        distance_au = self.distance_au(self.Z[row_index, 2])
        link_probability = p_bound_projected_flux_proxy(
            distance_au,
            self.node_flux_sum[left_index],
            self.node_flux_sum[right_index],
            self.total_flux_proxy,
            self.r_cluster_au,
            self.projection_setup,
            virial_factor=self.virial_factor,
            clip_to_bins=self.clip_to_bins,
        )
        self.link_probability_by_node[node_index] = link_probability
        self.link_distance_au_by_node[node_index] = distance_au

        if self.node_max_flux[left_index] >= self.node_max_flux[right_index]:
            self.propagate_probability(left_index, current_probability)
            self.propagate_probability(
                right_index,
                current_probability * link_probability,
            )
        else:
            self.propagate_probability(
                left_index,
                current_probability * link_probability,
            )
            self.propagate_probability(right_index, current_probability)

    def run(self):
        self.final_probs.fill(0.0)
        self.link_probability_by_node = {}
        self.link_distance_au_by_node = {}
        self.precompute_fluxes()
        self.propagate_probability(self.n_nodes - 1, 1.0)
        return self.final_probs


def multiplicity_analyse_bound_probability(
    points_au,
    flux_proxy,
    r_cluster_au,
    projection_setup,
    separation_threshold_au=1000.0,
    virial_factor=0.2,
    total_flux_proxy=None,
    method="centroid",
    source_ids=None,
    clip_to_bins=True,
):
    """Calculate MF and CF using the fiducial P(bound | s_proj) method."""
    points_au = np.asarray(points_au, dtype=float)
    flux_proxy = np.asarray(flux_proxy, dtype=float)
    if points_au.ndim != 2 or points_au.shape[1] != 2:
        raise ValueError("points_au must have shape (N, 2).")
    if len(flux_proxy) != len(points_au):
        raise ValueError("flux_proxy and points_au must have the same length.")
    if source_ids is None:
        source_ids = np.arange(len(points_au))
    source_ids = np.asarray(source_ids)
    if len(source_ids) != len(points_au):
        raise ValueError("source_ids and points_au must have the same length.")

    valid = (
        np.all(np.isfinite(points_au), axis=1)
        & np.isfinite(flux_proxy)
        & (flux_proxy > 0)
    )
    points_au = points_au[valid]
    flux_proxy = flux_proxy[valid]
    source_ids = source_ids[valid]
    if total_flux_proxy is None:
        total_flux_proxy = float(flux_proxy.sum())

    raw = multiplicity_analyse(
        points_au,
        separation_threshold_au,
        method=method,
        source_ids=source_ids,
    )
    if len(points_au) == 0:
        return {**raw, "n_valid_sources": 0}

    final_systems = []

    def get_leaves(node_index, n_leaves, Z):
        if node_index < n_leaves:
            return [node_index]
        row_index = node_index - n_leaves
        return get_leaves(int(Z[row_index, 0]), n_leaves, Z) + get_leaves(
            int(Z[row_index, 1]), n_leaves, Z
        )

    def resolve(indices):
        if len(indices) < 2:
            return [indices]

        Z = linkage(points_au[indices], method=method)
        hierarchy = HierarchyProbability_v2(
            Z,
            flux_proxy[indices],
            r_cluster_au,
            projection_setup,
            total_flux_proxy=total_flux_proxy,
            virial_factor=virial_factor,
            z_distance_unit="au",
            clip_to_bins=clip_to_bins,
        )
        effective_members = max(int(round(hierarchy.run().sum())), 1)
        if effective_members == len(indices):
            return [indices]

        split_node = min(
            hierarchy.link_probability_by_node,
            key=hierarchy.link_probability_by_node.get,
        )
        split_row = split_node - len(indices)
        left_local = get_leaves(int(Z[split_row, 0]), len(indices), Z)
        right_local = get_leaves(int(Z[split_row, 1]), len(indices), Z)
        left = [indices[index] for index in left_local]
        right = [indices[index] for index in right_local]
        involved = set(left + right)
        remaining = [index for index in indices if index not in involved]
        return resolve(left) + resolve(right) + (resolve(remaining) if remaining else [])

    for raw_system in raw["systems"]:
        final_systems.extend(resolve(raw_system["indices"].tolist()))

    orders = np.array([len(indices) for indices in final_systems], dtype=int)
    n_systems = len(orders)
    n_multiple = int(np.sum(orders >= 2))
    n_companions = int(np.sum(orders - 1))
    mf = n_multiple / n_systems
    cf = n_companions / n_systems

    return {
        "MF": mf,
        "CF": cf,
        "MF_interval": calculate_wilson_interval(mf, n_systems),
        "CF_interval": (
            calculate_poisson_interval(cf, n_systems)
            if cf > 0.5
            else calculate_wilson_interval(cf, n_systems)
        ),
        "n_sources": len(points_au),
        "n_valid_sources": len(points_au),
        "n_systems": n_systems,
        "n_multiple_systems": n_multiple,
        "n_companions": n_companions,
        "order_counts": {
            int(order): int(np.sum(orders == order)) for order in np.unique(orders)
        },
        "labels": raw["labels"],
        "systems": [
            {
                "order": len(indices),
                "indices": np.asarray(indices),
                "source_ids": source_ids[indices],
            }
            for indices in final_systems
        ],
    }


def calculate_mf_cf_with_bound_probability(
    points_au,
    flux_proxy,
    r_cluster_au,
    projection_setup,
    threshold_au=1000.0,
    virial_factor=0.2,
    total_flux_proxy=None,
    method="centroid",
    source_ids=None,
    clip_to_bins=True,
):
    """Compatibility wrapper for the current notebook function name."""
    result = multiplicity_analyse_bound_probability(
        points_au,
        flux_proxy,
        r_cluster_au,
        projection_setup,
        separation_threshold_au=threshold_au,
        virial_factor=virial_factor,
        total_flux_proxy=total_flux_proxy,
        method=method,
        source_ids=source_ids,
        clip_to_bins=clip_to_bins,
    )
    return {
        **result,
        "num_systems": result["n_systems"],
        "num_multiple_systems": result["n_multiple_systems"],
        "num_companions": result["n_companions"],
        "num_valid_sources": result["n_valid_sources"],
    }


def calculate_tobin_distances_with_probabilities_v2(
    points,
    brightness,
    Z,
    hierarchy_probability,
    cut_threshold=10000.0,
    plot_pairs=False,
):
    """Return linkage distances and stage-specific companion probabilities.

    Each merger receives the local link probability multiplied by the minimum
    probability already accumulated inside its companion subtree. Probabilities
    from wider ancestor mergers are never propagated back to smaller mergers.
    Call ``hierarchy_probability.run()`` before using this function.
    """
    points = np.asarray(points, dtype=float)
    brightness = np.asarray(brightness, dtype=float)
    Z = np.asarray(Z, dtype=float)
    n_stars = len(points)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2).")
    if len(brightness) != n_stars:
        raise ValueError("points and brightness must have the same length.")
    if Z.size == 0:
        return np.array([]), np.array([]), []
    if Z.ndim != 2 or Z.shape[1] < 3:
        raise ValueError("Z must be a scipy linkage matrix.")

    node_max_flux = np.asarray(
        hierarchy_probability.node_max_flux,
        dtype=float,
    )
    link_probability_by_node = getattr(
        hierarchy_probability,
        "link_probability_by_node",
        None,
    )
    if not isinstance(link_probability_by_node, dict):
        raise TypeError(
            "hierarchy_probability must provide link_probability_by_node."
        )
    if len(node_max_flux) < n_stars + len(Z):
        raise ValueError("node_max_flux is shorter than the linkage tree.")

    distances = []
    probabilities = []
    pairs = []
    subtree_min_probability = {
        leaf_index: 1.0 for leaf_index in range(n_stars)
    }
    brightest_in_cluster = {
        leaf_index: leaf_index for leaf_index in range(n_stars)
    }

    for row_index, row in enumerate(Z):
        left_index = int(row[0])
        right_index = int(row[1])
        distance = float(row[2])
        node_index = n_stars + row_index

        if (
            left_index not in subtree_min_probability
            or right_index not in subtree_min_probability
        ):
            continue
        if distance > cut_threshold:
            continue
        if node_index not in link_probability_by_node:
            raise RuntimeError(
                f"Missing probability for merger node {node_index}; "
                "call hierarchy_probability.run() first."
            )

        if node_max_flux[left_index] >= node_max_flux[right_index]:
            primary_index, companion_index = left_index, right_index
        else:
            primary_index, companion_index = right_index, left_index

        local_probability = float(link_probability_by_node[node_index])
        if not np.isfinite(local_probability):
            raise ValueError(
                f"Merger node {node_index} has a non-finite probability."
            )
        local_probability = float(np.clip(local_probability, 0.0, 1.0))
        stage_probability = (
            subtree_min_probability[companion_index] * local_probability
        )

        subtree_min_probability[node_index] = min(
            subtree_min_probability[primary_index],
            stage_probability,
        )
        distances.append(distance)
        probabilities.append(stage_probability)

        brightest_left = brightest_in_cluster[left_index]
        brightest_right = brightest_in_cluster[right_index]
        pairs.append((brightest_left, brightest_right))
        brightest_in_cluster[node_index] = (
            brightest_left
            if brightness[brightest_left] >= brightness[brightest_right]
            else brightest_right
        )

    distances = np.asarray(distances, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)

    if plot_pairs and pairs:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(points[:, 0], points[:, 1], s=12, color="black", zorder=3)
        for pair, probability in zip(pairs, probabilities):
            start, end = points[list(pair)]
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                color="tab:blue",
                alpha=0.2 + 0.8 * probability,
                linewidth=0.5,
            )
        ax.set_xlabel("X (au)")
        ax.set_ylabel("Y (au)")
        ax.set_aspect("equal")
        fig.tight_layout()

    return distances, probabilities, pairs


def plot_separation_fraction(
    distances_au,
    probabilities,
    n_sources,
    log10_bins,
    ax=None,
    show_raw=True,
    raw_label="No correction",
    weighted_label="Probability weighted",
    raw_color="0.85",
    weighted_color="0.45",
):
    """Plot raw and weighted companions per detected source in each bin."""
    import matplotlib.pyplot as plt

    distances_au = np.asarray(distances_au, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    log10_bins = np.asarray(log10_bins, dtype=float)
    if distances_au.shape != probabilities.shape:
        raise ValueError("distances_au and probabilities must have the same shape.")
    if n_sources <= 0:
        raise ValueError("n_sources must be positive.")
    if log10_bins.ndim != 1 or len(log10_bins) < 2:
        raise ValueError("log10_bins must contain at least two bin edges.")

    valid = (
        np.isfinite(distances_au)
        & (distances_au > 0)
        & np.isfinite(probabilities)
    )
    log_distances = np.log10(distances_au[valid])
    probabilities = np.clip(probabilities[valid], 0.0, 1.0)
    raw_counts, _ = np.histogram(log_distances, bins=log10_bins)
    weighted_counts, _ = np.histogram(
        log_distances,
        bins=log10_bins,
        weights=probabilities,
    )
    raw_fraction = raw_counts / n_sources
    weighted_fraction = weighted_counts / n_sources

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    widths = np.diff(log10_bins)
    if show_raw:
        ax.bar(
            log10_bins[:-1],
            raw_fraction,
            width=widths,
            align="edge",
            color=raw_color,
            edgecolor="0.7",
            linewidth=0.6,
            label=raw_label,
        )
    ax.bar(
        log10_bins[:-1],
        weighted_fraction,
        width=widths,
        align="edge",
        color=weighted_color,
        edgecolor="0.35",
        linewidth=0.6,
        label=weighted_label,
    )
    ax.set_xlabel(r"Separation ($\log_{10}({\rm au})$)")
    ax.set_ylabel("Companions per detected source per bin")
    ax.legend(frameon=False)

    histogram = {
        "log10_bin_edges": log10_bins,
        "log10_bin_centers": 0.5 * (log10_bins[:-1] + log10_bins[1:]),
        "raw_counts": raw_counts,
        "weighted_counts": weighted_counts,
        "raw_fraction": raw_fraction,
        "weighted_fraction": weighted_fraction,
    }
    return ax, histogram

