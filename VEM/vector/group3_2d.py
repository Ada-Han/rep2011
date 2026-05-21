import argparse
import math
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import splu

from framework_2D import ASP_VEM2D


KAPPA = 0.01
MU_VALUE = 2.0
X_LEFT = 0.0
X_RIGHT = 1.0
Y_BOTTOM = 0.0
Y_TOP = 1.0
A_MAX = 1.0
T_FINAL = 1.0
T_EVAL = 1.0

NX_SPACE_LIST = [8, 16, 32, 64, 128]
DT_SPACE = 1.0 / 480

NX_TIME = 256
DT_TIME_LIST = [1.0 / 10.0, 1.0 / 20.0, 1.0 / 40.0, 1.0 / 80.0]


def exact_space_function(x_value, y_value):
    """制造解关于空间部分 phi(x,y)=cos(pi x)cos(pi y)。"""
    return np.cos(np.pi * x_value) * np.cos(np.pi * y_value)


def exact_solution(nodes, age_value, time_value):
    """制造解 u(x,y,a,t)=cos(pi x)cos(pi y)exp(-(a+t))。"""
    x_values = nodes[:, 0]
    y_values = nodes[:, 1]
    return exact_space_function(x_values, y_values) * np.exp(-(age_value + time_value))


def exact_solution_gradient(points, age_value, time_value):
    """制造解空间梯度。"""
    x_values = points[:, 0]
    y_values = points[:, 1]
    factor = np.exp(-(age_value + time_value))
    gradient_x = -np.pi * np.sin(np.pi * x_values) * np.cos(np.pi * y_values) * factor
    gradient_y = -np.pi * np.cos(np.pi * x_values) * np.sin(np.pi * y_values) * factor
    return np.column_stack([gradient_x, gradient_y])


def source_values(nodes, age_value, time_value, kappa=KAPPA, mu_value=MU_VALUE):
    """源项 f(x,y,a,t)。"""
    coefficient = -2.0 + 2.0 * (np.pi ** 2) * kappa + mu_value
    return coefficient * exact_solution(nodes, age_value, time_value)


def source_values_batch(nodes, age_midpoints, time_value, kappa=KAPPA, mu_value=MU_VALUE):
    """批量计算源项，返回形状为 (n_age, n_nodes) 的矩阵。"""
    coefficient = -2.0 + 2.0 * (np.pi ** 2) * kappa + mu_value
    x_values = nodes[:, 0]
    y_values = nodes[:, 1]
    space_part = exact_space_function(x_values, y_values) # (n_nodes,)
    decay = np.exp(-(age_midpoints + time_value)) # (n_age,)
    source_batch = coefficient * space_part[np.newaxis, :] * decay[:, np.newaxis] # (n_age, n_nodes)
    return source_batch


def build_age_grid(dt, a_max=A_MAX):
    """构造年龄网格 a_i = i * dt。"""
    n_age = int(round(a_max / dt))
    if not np.isclose(n_age * dt, a_max):
        raise ValueError("dt must divide the age interval exactly.")
    return np.linspace(0.0, a_max, n_age + 1)


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


def manufactured_birth_state(model, time_value):
    """制造解对应的年龄 a = 0 精确边界值。"""
    return exact_solution(model.nodes, 0.0, time_value)


def build_initial_states(model, dt):
    """把精确解采样到各年龄节点，构造 t = 0 初值。"""
    age_grid = build_age_grid(dt, a_max=A_MAX)
    initial_states = np.zeros((len(age_grid), model.n_nodes))

    for age_index, age_value in enumerate(age_grid):
        initial_states[age_index, :] = exact_solution(model.nodes, age_value, 0.0)

    return age_grid, initial_states


def space_age_l2_relative_error(model, numerical_states, age_grid, time_value):
    """直接调用 framework_2D 的全向量化 L2 相对误差计算函数（超快）。"""
    return model.compute_l2_error_vectorized(numerical_states, exact_solution, age_grid, time_value)


def space_age_h1_relative_error(model, numerical_states, age_grid, time_value):
    """直接调用 framework_2D 的全向量化 H1 相对误差计算函数（超快）。"""
    return model.compute_h1_error_vectorized(numerical_states, exact_solution_gradient, age_grid, time_value)


def solve_manufactured_problem(model, dt, time_stop, kappa=KAPPA, mu_value=MU_VALUE):
    """求解制造解问题（全向量化高性能版）。"""
    age_grid, current_states = build_initial_states(model, dt)
    n_age = len(age_grid) - 1
    n_time = int(round(time_stop / dt))
    if not np.isclose(n_time * dt, time_stop):
        raise ValueError("time_stop must be a multiple of dt.")

    diffusion_matrix = kappa * model.stiffness_matrix
    reaction_matrix = mu_value * model.mass_matrix
    lhs = (1.0 / dt) * model.mass_matrix + 0.5 * diffusion_matrix + 0.5 * reaction_matrix
    rhs = (1.0 / dt) * model.mass_matrix - 0.5 * diffusion_matrix - 0.5 * reaction_matrix

    linear_solver = splu(lhs.tocsc())

    age_midpoints = age_grid[1:] - 0.5 * dt # (n_age,)

    for time_index in range(1, n_time + 1):
        next_states = np.zeros_like(current_states)
        current_time = time_index * dt
        mid_time = current_time - 0.5 * dt

        # 批量计算源项: shape (n_age, n_nodes)
        source_batch = source_values_batch(model.nodes, age_midpoints, mid_time, kappa=kappa, mu_value=mu_value)
        # 批量计算 load 向量: shape (n_nodes, n_age)
        load_batch = model.mass_matrix @ source_batch.T

        # 批量计算 RHS: shape (n_nodes, n_age)
        rhs_states = rhs @ current_states[:-1, :].T
        RHS_batch = rhs_states + load_batch

        # 批量求解所有年龄层: shape (n_nodes, n_age)
        sol = linear_solver.solve(RHS_batch)

        next_states[1:, :] = sol.T
        next_states[0, :] = manufactured_birth_state(model, current_time)
        current_states = next_states

    return age_grid, current_states


def compute_space_convergence():
    """计算空间收敛表。"""
    results = []

    for nx_elements in NX_SPACE_LIST:
        model = build_model(nx_elements)
        age_grid, numerical_states = solve_manufactured_problem(model, DT_SPACE, T_EVAL)
        l2_error = space_age_l2_relative_error(model, numerical_states, age_grid, T_EVAL)
        h1_error = space_age_h1_relative_error(model, numerical_states, age_grid, T_EVAL)

        results.append(
            {
                "nx": nx_elements,
                "h": 1.0 / nx_elements,
                "l2_error": l2_error,
                "h1_error": h1_error,
                "l2_rate": np.nan,
                "h1_rate": np.nan,
            }
        )

    for index in range(1, len(results)):
        previous = results[index - 1]
        current = results[index]
        current["l2_rate"] = np.log(previous["l2_error"] / current["l2_error"]) / np.log(previous["h"] / current["h"])
        current["h1_rate"] = np.log(previous["h1_error"] / current["h1_error"]) / np.log(previous["h"] / current["h"])

    return results


def compute_time_convergence():
    """计算时间收敛表。"""
    model = build_model(NX_TIME)
    results = []

    for dt in DT_TIME_LIST:
        age_grid, numerical_states = solve_manufactured_problem(model, dt, T_EVAL)
        l2_error = space_age_l2_relative_error(model, numerical_states, age_grid, T_EVAL)

        results.append(
            {
                "dt": dt,
                "l2_error": l2_error,
                "rate": np.nan,
            }
        )

    for index in range(1, len(results)):
        previous = results[index - 1]
        current = results[index]
        current["rate"] = np.log(previous["l2_error"] / current["l2_error"]) / np.log(previous["dt"] / current["dt"])

    return results


def print_space_table(results):
    """打印空间收敛误差表。"""
    print("2D Group-3 Manufactured Solution: spatial convergence (vectorized)")
    print("=" * 104)
    print(
        f"{'nx=ny':>10} {'h':>12} {'L2 error':>16} {'L2 rate':>10} "
        f"{'H1 error':>16} {'H1 rate':>10}"
    )
    print("-" * 104)

    for row in results:
        l2_rate_string = "-" if np.isnan(row["l2_rate"]) else f"{row['l2_rate']:.4f}"
        h1_rate_string = "-" if np.isnan(row["h1_rate"]) else f"{row['h1_rate']:.4f}"
        print(
            f"{row['nx']:10d} "
            f"{row['h']:12.6f} "
            f"{row['l2_error']:16.8e} "
            f"{l2_rate_string:>10} "
            f"{row['h1_error']:16.8e} "
            f"{h1_rate_string:>10}"
        )

    print("-" * 104)


def print_time_table(results):
    """打印时间收敛误差表。"""
    print("2D Group-3 Manufactured Solution: temporal convergence (vectorized)")
    print("=" * 72)
    print(f"{'dt':>12} {'L2 error':>16} {'rate':>10}")
    print("-" * 72)

    for row in results:
        rate_string = "-" if np.isnan(row["rate"]) else f"{row['rate']:.4f}"
        print(
            f"{row['dt']:12.6f} "
            f"{row['l2_error']:16.8e} "
            f"{rate_string:>10}"
        )

    print("-" * 72)


def print_timing_summary(total_elapsed):
    """总计算时间"""
    print(f"Total computation time before plot: {total_elapsed:.2f} seconds")


def plot_group3_results(space_results, time_results):
    """绘制第三组实验的空间与时间收敛图。"""
    h_values = np.array([row["h"] for row in space_results])
    l2_space = np.array([row["l2_error"] for row in space_results])
    h1_space = np.array([row["h1_error"] for row in space_results])

    dt_values = np.array([row["dt"] for row in time_results])
    l2_time = np.array([row["l2_error"] for row in time_results])

    l2_space_ref = l2_space[0] * (h_values / h_values[0]) ** 2
    h1_space_ref = h1_space[0] * (h_values / h_values[0])
    l2_time_ref = l2_time[0] * (dt_values / dt_values[0]) ** 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].loglog(h_values, l2_space, "o-", linewidth=2.0, label="L2 error")
    axes[0].loglog(h_values, h1_space, "s-", linewidth=2.0, label="H1 error")
    axes[0].loglog(h_values, l2_space_ref, "k--", linewidth=1.5, label="slope 2")
    axes[0].loglog(h_values, h1_space_ref, "k:", linewidth=1.5, label="slope 1")
    axes[0].set_xlabel("Mesh size h")
    axes[0].set_ylabel("Relative error")
    axes[0].set_title("Spatial convergence")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].loglog(dt_values, l2_time, "o-", linewidth=2.0, label="L2 error")
    axes[1].loglog(dt_values, l2_time_ref, "k--", linewidth=1.5, label="slope 2")
    axes[1].set_xlabel("Time step dt")
    axes[1].set_ylabel("Relative error")
    axes[1].set_title("Temporal convergence")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="Run the third 2D manufactured-solution experiment.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show the spatial and temporal convergence plots.",
    )
    args = parser.parse_args()

    total_start = time.perf_counter()
    space_results = compute_space_convergence()
    time_results = compute_time_convergence()
    total_elapsed = time.perf_counter() - total_start

    print_space_table(space_results)
    print_time_table(time_results)
    print_timing_summary(total_elapsed)

    if args.plot:
        plot_group3_results(space_results, time_results)
        plt.show()


if __name__ == "__main__":
    main()
