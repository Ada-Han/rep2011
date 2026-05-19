import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from framework import ASP_FEM1D, trapezoidal_birth
from example1 import (
    A_MAX,
    BETA,
    DT_LIST,
    DT_REFERENCE,
    H_COARSE,
    H_REFERENCE,
    KAPPA,
    T_EVAL,
    T_FINAL,
    X_LEFT,
    X_RIGHT,
    build_age_grid,
    composite_trapezoid,
    initial_condition,
    interpolate_reference_to_grid,
    weighted_l2_norm,
)


PAPER_ERRORS = np.array([0.0205, 0.0050, 0.0022, 0.0012, 5.3031e-4, 2.8353e-4])


def mu_example_2(age):
    """
    Example 2 的死亡率函数。

    对应公式:
        mu(a) = 2.457 + 2 a (a - 2)
    """
    age = np.asarray(age, dtype=float)
    return 2.457 + 2.0 * age * (age - 2.0)


def scalar_layer_factor(dt, age_midpoint):
    """
    计算 Example 2 的单个年龄层推进因子。

    对应公式:
        U_i^n = q_i * U_{i-1}^{n-1}
        q_i = (1 / dt - 0.5 * mu(a_{i-1/2})) / (1 / dt + 0.5 * mu(a_{i-1/2}))
    """
    mu_value = float(mu_example_2(age_midpoint))
    numerator = 1.0 / dt - 0.5 * mu_value
    denominator = 1.0 / dt + 0.5 * mu_value
    return numerator / denominator


def solve_uniform_example_2(dt, h, t_stop):
    """
    用空间常数模态求解 Example 2。

    说明:
    - Example 2 的初值、死亡率和出生率都不依赖空间 x。
    - 在零通量边界下，数值解保持空间常数模态。
    - 因此只需推进年龄方向上的标量剖面。
    """
    model = ASP_FEM1D(
        x_left=X_LEFT,
        x_right=X_RIGHT,
        h=h,
        a_max=A_MAX,
        t_final=T_FINAL,
        kappa=KAPPA,
    )

    age_grid = build_age_grid(dt, a_max=A_MAX)
    n_age = len(age_grid) - 1
    n_time = int(round(t_stop / dt))
    if not np.isclose(n_time * dt, t_stop):
        raise ValueError("t_stop must be a multiple of dt.")

    current_profile = initial_condition(age_grid)
    birth_function = trapezoidal_birth(BETA)

    time_grid = np.linspace(0.0, t_stop, n_time + 1)
    total_population = np.zeros(n_time + 1)
    total_population[0] = composite_trapezoid(current_profile, dt)

    for n in range(1, n_time + 1):
        next_profile = np.zeros_like(current_profile)

        for age_index in range(1, n_age + 1):
            age_midpoint = (age_index - 0.5) * dt
            factor = scalar_layer_factor(dt, age_midpoint)
            next_profile[age_index] = factor * current_profile[age_index - 1]

        next_profile[0] = birth_function(next_profile[:, None], dt)[0]
        current_profile = next_profile
        total_population[n] = composite_trapezoid(current_profile, dt)

    state = np.tile(current_profile[:, None], (1, model.n_nodes))
    return {
        "model": model,
        "age_grid": age_grid,
        "profile": current_profile,
        "state": state,
        "time_grid": time_grid,
        "total_population": total_population,
    }


def relative_error_at_t_eval(dt, reference_solution):
    """
    计算 Example 2 在 t = 3 时刻的相对误差。
    """
    coarse_solution = solve_uniform_example_2(dt=dt, h=H_COARSE, t_stop=T_EVAL)
    coarse_age = coarse_solution["age_grid"]
    coarse_profile = coarse_solution["profile"]

    reference_on_coarse = interpolate_reference_to_grid(
        reference_solution["age_grid"],
        reference_solution["profile"],
        coarse_age,
    )

    numerator = weighted_l2_norm(coarse_profile - reference_on_coarse, dt)
    denominator = weighted_l2_norm(reference_on_coarse, dt)
    return numerator / denominator


def compute_convergence_table():
    """
    计算 Example 2 的 Table 2 误差和收敛阶。
    """
    reference_solution = solve_uniform_example_2(
        dt=DT_REFERENCE,
        h=H_REFERENCE,
        t_stop=T_EVAL,
    )

    errors = []
    rates = [np.nan]

    for dt in DT_LIST:
        errors.append(relative_error_at_t_eval(dt, reference_solution))

    errors = np.array(errors)
    for i in range(1, len(DT_LIST)):
        rates.append(np.log(errors[i - 1] / errors[i]) / np.log(DT_LIST[i - 1] / DT_LIST[i]))

    return errors, np.array(rates)


def print_convergence_table(errors, rates):
    """
    打印 Example 2 的 Table 2 对照结果。
    """
    print("Example 2: relative errors at t = 3 years")
    print("-" * 86)
    print(f"{'dt':>10} {'error':>16} {'rate':>12} {'paper_error':>16} {'abs_diff':>16}")
    print("-" * 86)

    for i, dt in enumerate(DT_LIST):
        rate_str = "-" if i == 0 else f"{rates[i]:.4f}"
        abs_diff = abs(errors[i] - PAPER_ERRORS[i])
        print(
            f"{dt:10.6f} "
            f"{errors[i]:16.8e} "
            f"{rate_str:>12} "
            f"{PAPER_ERRORS[i]:16.8e} "
            f"{abs_diff:16.8e}"
        )

    print("-" * 86)


def plot_total_population_time(history):
    """
    复现 Fig. 1 左图：x = 0.1 km 处总人口关于时间的变化。

    说明:
    - 总人口定义为 p(x, t) = integral_0^Am u(x, a, t) da。
    - 对当前 Example 2，解在空间上保持常数，因此 x = 0.1 km 的曲线
      与任意空间位置上的总人口曲线相同。
    """
    plt.figure(figsize=(8, 5))
    plt.plot(
        history["time_grid"],
        history["total_population"],
        linewidth=2.0,
        color="tab:blue",
    )
    plt.xlabel("Time (years)")
    plt.ylabel("Total Population Number")
    plt.title("Total Population Number at x = 0.1 km")
    y_ticks = np.arange(7.2, 9.2 + 1.0e-12, 0.2) * 1.0e4
    plt.ylim(7.2e4, 9.2e4)
    plt.yticks(y_ticks)

    y_formatter = ScalarFormatter(useMathText=True)
    y_formatter.set_scientific(True)
    y_formatter.set_powerlimits((4, 4))
    plt.gca().yaxis.set_major_formatter(y_formatter)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_total_population_surface(history):
    """
    复现 Fig. 1 右图：总人口关于空间和时间的曲面。

    说明:
    - 由于当前算例对空间 x 不显含，曲面在空间方向上为常值。
    - 为了与论文图像结构一致，仍然绘制 p(x, t) 的 space-time 曲面。
    """
    x_grid = np.linspace(X_LEFT, X_RIGHT, 81)
    time_grid = history["time_grid"]

    time_mesh, x_mesh = np.meshgrid(time_grid, x_grid, indexing="xy")
    population_surface = np.tile(history["total_population"], (len(x_grid), 1))

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        x_mesh,
        time_mesh,
        population_surface,
        cmap="seismic",
        vmin=7.2e4,
        vmax=9.2e4,
        rcount=len(x_grid),
        ccount=len(time_grid),
        edgecolor="none",
        linewidth=0.0,
        antialiased=True,
        alpha=1.0,
        shade=True,
    )
    ax.view_init(elev=15, azim=-15)
    ax.set_xlabel("Space (km)")
    ax.set_ylabel("Time (years)")
    ax.set_zlabel("Total Population Number")
    ax.set_title("Total Population Number")
    ax.set_xticks([0.0, 0.1, 0.2])
    ax.set_xticklabels(["0", "0.1", "0.2"])

    z_ticks = np.arange(7.2, 9.2 + 1.0e-12, 0.2) * 1.0e4
    ax.set_zlim(7.2e4, 9.2e4)
    ax.set_zticks(z_ticks)

    z_formatter = ScalarFormatter(useMathText=True)
    z_formatter.set_scientific(True)
    z_formatter.set_powerlimits((4, 4))
    ax.zaxis.set_major_formatter(z_formatter)
    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="Reproduce Example 2 in Section 6.")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Compute the Table 2 data without showing figures.",
    )
    args = parser.parse_args()

    errors, rates = compute_convergence_table()
    print_convergence_table(errors, rates)

    if not args.no_plot:
        history = solve_uniform_example_2(
            dt=DT_REFERENCE,
            h=H_REFERENCE,
            t_stop=T_FINAL,
        )
        plot_total_population_time(history)
        plot_total_population_surface(history)
        plt.show()


if __name__ == "__main__":
    main()
