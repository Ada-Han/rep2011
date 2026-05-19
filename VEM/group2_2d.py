import argparse
import math
import time

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse.linalg import eigsh

from framework_2D import ASP_VEM2D

# 参数设定区域-----

KAPPA = 0.01
BETA = 1.3025
X_LEFT = 0.0
X_RIGHT = 1.0
Y_BOTTOM = 0.0
Y_TOP = 1.0
A_MAX = 15.0
T_FINAL = 12.0
T_EVAL = 3.0
DELTA = 0.20

DT_SPACE = 1.0 / 240.0
NX_LIST = [8, 16, 32, 64, 128]
NX_REFERENCE = 512
# ----------------

# 定义死亡率
def mu_function(age):
    """group2-- 定常死亡率: mu(a) = 2。"""
    age = np.asarray(age, dtype=float)
    return np.full_like(age, 2.0, dtype=float)

# 初值，与年龄相关
def initial_age_profile(age):
    """年龄方向初值取 u0(a) 。"""
    age = np.asarray(age, dtype=float)
    values = np.zeros_like(age, dtype=float)
    mask = (age >= 0.0) & (age <= 3.0)
    values[mask] = (
        1.0e5
        * (20.0 * age[mask] + 1.0) ** 2
        * (1.0 - age[mask] / 3.0)
        * np.exp(-12.0 * age[mask])
    )
    return values


def build_age_grid(dt, a_max=A_MAX):
    """构造年龄网格，delta a = delta t 时间步长相同，沿特征线。"""
    n_age = int(round(a_max / dt))
    if not np.isclose(n_age * dt, a_max):
        raise ValueError("dt must divide the age interval exactly.")
    return np.linspace(0.0, a_max, n_age + 1)


def composite_trapezoid(values, dt):
    """对离散年龄剖面做复合梯形积分。"""
    return dt * (0.5 * values[0] + np.sum(values[1:-1]) + 0.5 * values[-1])


def normalize_mode_with_mass(model, mode_values):
    """去掉常模态分量，并把离散模态按无穷范数归一化。"""
    ones = np.ones(model.n_nodes)
    mass_ones = model.mass_matrix @ ones
    coefficient = np.dot(mass_ones, mode_values) / np.dot(ones, mass_ones)
    centered_mode = mode_values - coefficient * ones

    max_abs = np.max(np.abs(centered_mode))
    if max_abs <= 0.0:
        raise ValueError("The mode vector is degenerate after mean removal.")
    return centered_mode / max_abs


def target_mode_values(nodes):
    """连续目标模态 phi(x,y)=cos(pi x)cos(pi y) 的节点取值。"""
    x_values = nodes[:, 0]
    y_values = nodes[:, 1]
    return np.cos(np.pi * x_values) * np.cos(np.pi * y_values)

# 复习
def compute_selected_eigenmode(model, num_eigs=4):
    """选出与 cos(pi x)cos(pi y) 最接近的离散 Neumann 本征模态。"""
    target_lambda = 2.0 * np.pi * np.pi
    eigenvalues, eigenvectors = eigsh(
        model.stiffness_matrix,
        k=num_eigs,
        M=model.mass_matrix,
        sigma=target_lambda,
        which="LM",
    )

    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    target = normalize_mode_with_mass(model, target_mode_values(model.nodes))
    correlations = []

    for column_index in range(len(eigenvalues)):
        vector = eigenvectors[:, column_index]
        vector = normalize_mode_with_mass(model, vector)
        correlation = abs(np.dot(target, model.mass_matrix @ vector))
        correlations.append(correlation)

    best_index = int(np.argmax(correlations))
    selected_lambda = float(eigenvalues[best_index])
    selected_mode = normalize_mode_with_mass(model, eigenvectors[:, best_index])

    sign = np.sign(np.dot(target, model.mass_matrix @ selected_mode))
    if sign == 0.0:
        sign = 1.0
    selected_mode = sign * selected_mode

    return selected_lambda, selected_mode


def scalar_birth_update(profile, dt, beta=BETA):
    """对单个模态剖面执行标量出生边界更新。"""
    weighted_sum = np.sum(profile[1:-1]) + 0.5 * profile[-1]
    denominator = 1.0 - 0.5 * beta * dt
    if denominator <= 0.0:
        raise ValueError("The birth boundary denominator must stay positive.")
    return (beta * dt / denominator) * weighted_sum


def solve_mode_profile(initial_profile, dt, lambda_value, mu_function, t_stop):
    """对单个空间模态的年龄剖面做 characteristic 推进。"""
    age_grid = build_age_grid(dt, a_max=A_MAX)
    n_age = len(age_grid) - 1
    n_time = int(round(t_stop / dt))
    if not np.isclose(n_time * dt, t_stop):
        raise ValueError("t_stop must be a multiple of dt.")

    current_profile = initial_profile(age_grid)
    factors = np.zeros(n_age)

    for age_index in range(1, n_age + 1):
        age_midpoint = (age_index - 0.5) * dt
        total_decay = float(mu_function(age_midpoint)) + KAPPA * lambda_value
        numerator = 1.0 / dt - 0.5 * total_decay
        denominator = 1.0 / dt + 0.5 * total_decay
        factors[age_index - 1] = numerator / denominator

    for _ in range(n_time):
        next_profile = np.zeros_like(current_profile)

        for age_index in range(1, n_age + 1):
            next_profile[age_index] = factors[age_index - 1] * current_profile[age_index - 1]

        next_profile[0] = scalar_birth_update(next_profile, dt, beta=BETA)
        current_profile = next_profile

    return age_grid, current_profile


def build_modal_solution(model, dt, mu_function, t_stop, delta=DELTA):
    """构造第二组实验的模态分解解 c(a,t)+d(a,t)psi_h(x,y)。"""
    lambda_value, mode_vector = compute_selected_eigenmode(model)

    def constant_initial(age):
        return initial_age_profile(age)

    def mode_initial(age):
        return delta * initial_age_profile(age)

    age_grid, constant_profile = solve_mode_profile(constant_initial, dt, 0.0, mu_function, t_stop)
    _, mode_profile = solve_mode_profile(mode_initial, dt, lambda_value, mu_function, t_stop)

    return {
        "model": model,
        "age_grid": age_grid,
        "constant_profile": constant_profile,
        "mode_profile": mode_profile,
        "mode_vector": mode_vector,
        "lambda_value": lambda_value,
    }


def node_key(x_value, y_value, digits=12):
    """把节点坐标转成稳定的字典键。"""
    return (round(float(x_value), digits), round(float(y_value), digits))


def restrict_reference_mode(reference_model, reference_mode_vector, coarse_model):
    """把细网格模态限制到粗网格节点上。"""
    reference_map = {}
    for node_index, (x_value, y_value) in enumerate(reference_model.nodes):
        reference_map[node_key(x_value, y_value)] = reference_mode_vector[node_index]

    restricted_values = np.zeros(coarse_model.n_nodes)
    for node_index, (x_value, y_value) in enumerate(coarse_model.nodes):
        key = node_key(x_value, y_value)
        if key not in reference_map:
            raise KeyError("A coarse-grid node was not found on the reference grid.")
        restricted_values[node_index] = reference_map[key]

    return restricted_values


GAUSS_POINTS = [
    (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0),
    (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
    (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
]


def build_modal_spatial_moments(coarse_solution, reference_solution):
    """预计算第二组误差公式里需要的空间矩量。"""
    coarse_model = coarse_solution["model"]
    reference_mode_vector_restricted = restrict_reference_mode(
        reference_solution["model"],
        reference_solution["mode_vector"],
        coarse_model,
    )

    moments = {
        "domain_measure": 0.0,
        "coarse_mode_mean": 0.0,
        "reference_mode_mean": 0.0,
        "coarse_mode_square": 0.0,
        "reference_mode_square": 0.0,
        "cross_mode_product": 0.0,
    }

    for element in coarse_model.elements:
        node_coords = coarse_model.nodes[element]
        coarse_local_values = coarse_solution["mode_vector"][element]
        reference_local_values = reference_mode_vector_restricted[element]

        area = 0.5 * abs(
            (node_coords[1, 0] - node_coords[0, 0]) * (node_coords[2, 1] - node_coords[0, 1])
            - (node_coords[2, 0] - node_coords[0, 0]) * (node_coords[1, 1] - node_coords[0, 1])
        )

        for lam1, lam2, weight in GAUSS_POINTS:
            lam3 = 1.0 - lam1 - lam2
            quadrature_weight = area * weight

            coarse_value = lam1 * coarse_local_values[0] + lam2 * coarse_local_values[1] + lam3 * coarse_local_values[2]
            reference_value = lam1 * reference_local_values[0] + lam2 * reference_local_values[1] + lam3 * reference_local_values[2]

            moments["domain_measure"] += quadrature_weight
            moments["coarse_mode_mean"] += quadrature_weight * coarse_value
            moments["reference_mode_mean"] += quadrature_weight * reference_value
            moments["coarse_mode_square"] += quadrature_weight * coarse_value * coarse_value
            moments["reference_mode_square"] += quadrature_weight * reference_value * reference_value
            moments["cross_mode_product"] += quadrature_weight * coarse_value * reference_value

    return moments


def compute_space_age_relative_error(coarse_solution, reference_solution, dt):
    """计算 t = 3 时刻的空间-年龄相对误差。"""
    coarse_constant = coarse_solution["constant_profile"]
    coarse_mode = coarse_solution["mode_profile"]
    reference_constant = reference_solution["constant_profile"]
    reference_mode = reference_solution["mode_profile"]
    spatial_moments = build_modal_spatial_moments(coarse_solution, reference_solution)

    n_age = len(coarse_constant) - 1
    error_accumulator = 0.0
    reference_accumulator = 0.0

    for age_index in range(n_age + 1):
        weight = 0.5 if age_index in (0, n_age) else 1.0
        delta_constant = coarse_constant[age_index] - reference_constant[age_index]
        coarse_mode_coefficient = coarse_mode[age_index]
        reference_mode_coefficient = reference_mode[age_index]

        error_squared = (
            delta_constant * delta_constant * spatial_moments["domain_measure"]
            + 2.0 * delta_constant * coarse_mode_coefficient * spatial_moments["coarse_mode_mean"]
            - 2.0 * delta_constant * reference_mode_coefficient * spatial_moments["reference_mode_mean"]
            + coarse_mode_coefficient * coarse_mode_coefficient * spatial_moments["coarse_mode_square"]
            + reference_mode_coefficient * reference_mode_coefficient * spatial_moments["reference_mode_square"]
            - 2.0 * coarse_mode_coefficient * reference_mode_coefficient * spatial_moments["cross_mode_product"]
        )
        reference_squared = (
            reference_constant[age_index] * reference_constant[age_index] * spatial_moments["domain_measure"]
            + 2.0
            * reference_constant[age_index]
            * reference_mode_coefficient
            * spatial_moments["reference_mode_mean"]
            + reference_mode_coefficient * reference_mode_coefficient * spatial_moments["reference_mode_square"]
        )

        error_accumulator += weight * max(float(error_squared), 0.0)
        reference_accumulator += weight * max(float(reference_squared), 0.0)

    error_norm = math.sqrt(dt * error_accumulator)
    reference_norm = math.sqrt(dt * reference_accumulator)
    return error_norm / reference_norm


def total_population_field(solution, dt):
    """计算 t = t_eval 时刻的局部总人口场。"""
    constant_total = composite_trapezoid(solution["constant_profile"], dt)
    mode_total = composite_trapezoid(solution["mode_profile"], dt)
    return constant_total + mode_total * solution["mode_vector"]


def build_model(nx_elements):
    """构造单位方形上的二维 P1 网格模型。"""
    mesh_size = 1.0 / nx_elements
    return ASP_VEM2D(
        x_left=X_LEFT,
        x_right=X_RIGHT,
        y_bottom=Y_BOTTOM,
        y_top=Y_TOP,
        hx=mesh_size,
        hy=mesh_size,
        a_max=A_MAX,
        t_final=T_FINAL,
        kappa=KAPPA,
    )


def print_spatial_error_table(results):
    """打印第二组实验的空间误差表。"""
    print("2D Group-2 Baseline Experiment")
    print("=" * 96)
    print("Space-varying initial condition on Omega = (0, 1) x (0, 1)")
    print("u0(x, y, a) = u0(a) * (1 + 0.2 * psi_h(x, y))")
    print("psi_h is the discrete Neumann eigenmode closest to cos(pi x) cos(pi y)")
    print(f"fixed dt = {DT_SPACE:.8f}, evaluation time t = {T_EVAL}")
    print("=" * 96)
    print(f"{'nx=ny':>10} {'h':>12} {'rel_error':>16} {'rate':>10}")
    print("-" * 96)

    continuous_lambda = 2.0 * np.pi * np.pi
    for index, row in enumerate(results):
        rate_string = "-" if index == 0 else f"{row['rate']:.4f}"
        print(
            f"{row['nx']:10d} "
            f"{row['h']:12.6f} "
            f"{row['rel_error']:16.8e} "
            f"{rate_string:>10}"
        )

    print("-" * 96)


def print_timing_summary(total_elapsed):
    """总计算时长"""
    print(f"Total computation time before plot: {total_elapsed:.2f} seconds")


def plot_group2_results(reference_solution, finest_solution, results):
    """绘制第二组实验的收敛图和总人口场。"""
    h_values = np.array([row["h"] for row in results])
    error_values = np.array([row["rel_error"] for row in results])
    second_order_line = error_values[0] * (h_values / h_values[0]) ** 2

    plt.figure(figsize=(7.5, 5.0))
    plt.loglog(h_values, error_values, "o-", linewidth=2.0, markersize=7, label="Computed error")
    plt.loglog(h_values, second_order_line, "k--", linewidth=1.5, label="Reference slope 2")
    plt.xlabel("Mesh size h")
    plt.ylabel("Relative space-age error at t = 3")
    plt.title("Group 2: spatial convergence on the 2D baseline")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("group2_convergence.png", dpi=150, bbox_inches="tight")

    reference_population = total_population_field(reference_solution, DT_SPACE)
    finest_population = total_population_field(finest_solution, DT_SPACE)

    triangulation_reference = mtri.Triangulation(
        reference_solution["model"].nodes[:, 0],
        reference_solution["model"].nodes[:, 1],
        reference_solution["model"].elements,
    )
    triangulation_finest = mtri.Triangulation(
        finest_solution["model"].nodes[:, 0],
        finest_solution["model"].nodes[:, 1],
        finest_solution["model"].elements,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    contour_1 = axes[0].tricontourf(triangulation_reference, reference_population, levels=30, cmap="viridis")
    axes[0].set_title("Reference total population field")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    fig.colorbar(contour_1, ax=axes[0])

    contour_2 = axes[1].tricontourf(triangulation_finest, finest_population, levels=30, cmap="viridis")
    axes[1].set_title(f"Finite mesh total population field (nx = {results[-1]['nx']})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    fig.colorbar(contour_2, ax=axes[1])

    plt.tight_layout()
    plt.savefig("group2_population_fields.png", dpi=150, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Run the second 2D baseline experiment.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show the spatial convergence plot and the total population fields.",
    )
    args = parser.parse_args()

    total_start = time.perf_counter()
    reference_model = build_model(NX_REFERENCE)
    reference_solution = build_modal_solution(reference_model, DT_SPACE, mu_function, T_EVAL, delta=DELTA)

    results = []
    finest_solution = None
    for nx_elements in NX_LIST:
        coarse_model = build_model(nx_elements)
        coarse_solution = build_modal_solution(coarse_model, DT_SPACE, mu_function, T_EVAL, delta=DELTA)

        rel_error = compute_space_age_relative_error(coarse_solution, reference_solution, DT_SPACE)
        row = {
            "nx": nx_elements,
            "h": 1.0 / nx_elements,
            "lambda_h": coarse_solution["lambda_value"],
            "rel_error": rel_error,
            "rate": np.nan,
        }
        results.append(row)

        if nx_elements == NX_LIST[-1]:
            finest_solution = coarse_solution

    for index in range(1, len(results)):
        previous = results[index - 1]
        current = results[index]
        current["rate"] = np.log(previous["rel_error"] / current["rel_error"]) / np.log(previous["h"] / current["h"])

    total_elapsed = time.perf_counter() - total_start
    print_spatial_error_table(results)
    print_timing_summary(total_elapsed)

    if args.plot:
        plot_group2_results(reference_solution, finest_solution, results)
        print("Plots saved to group2_convergence.png and group2_population_fields.png")
        plt.show()


if __name__ == "__main__":
    main()
