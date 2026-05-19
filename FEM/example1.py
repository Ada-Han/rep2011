import argparse
import numpy as np
import matplotlib.pyplot as plt

from framework import ASP_FEM1D, trapezoidal_birth


KAPPA = 0.01
BETA = 1.3025
X_LEFT = 0.0
X_RIGHT = 0.2
A_MAX = 15.0
T_FINAL = 12.0
T_EVAL = 3.0
H_COARSE = 0.005
H_REFERENCE = 1.0 / 800.0

DT_LIST = [1.0 / 10.0, 1.0 / 20.0, 1.0 / 30.0, 1.0 / 40.0, 1.0 / 60.0, 1.0 / 80.0]
DT_REFERENCE = 1.0 / 240.0

PAPER_ERRORS = np.array([0.0446, 0.0111, 0.0049, 0.0027, 0.0012, 6.2433e-4])


def mu_example_1(age):
    """
    Example 1 的死亡率函数。

    对应公式:
        mu(a) = 2
    """
    age = np.asarray(age)
    return np.full_like(age, 2.0, dtype=float)


def initial_condition(age):
    """
    Section 6 的初值函数 u0(a)。

    对应公式:
        u0(x, a) = 10^5 * (20a + 1)^2 * (1 - a / 3) * exp(-12a),   a in [0, 3]
                   0,                                               a in (3, Am]

    该初值与空间 x 无关，因此 Example 1 的解在空间上保持常数模态。
    """
    age = np.asarray(age)
    values = np.zeros_like(age, dtype=float)
    mask = (age >= 0.0) & (age <= 3.0)
    values[mask] = (
        1.0e5
        * (20.0 * age[mask] + 1.0) ** 2
        * (1.0 - age[mask] / 3.0)
        * np.exp(-12.0 * age[mask])
    )
    return values


def composite_trapezoid(values, dt):
    """
    年龄方向复合梯形积分。

    对应公式:
        integral_0^Am f(a) da
        ~= dt * [0.5 f_0 + sum_{i=1}^{N-1} f_i + 0.5 f_N]
    """
    return dt * (0.5 * values[0] + np.sum(values[1:-1]) + 0.5 * values[-1])


def weighted_l2_norm(values, dt):
    """
    年龄方向带梯形权重的离散 L2 范数。

    对应公式:
        ||v||_2,h ~= sqrt(dt * [0.5 v_0^2 + sum_{i=1}^{N-1} v_i^2 + 0.5 v_N^2])
    """
    squared = values * values
    return np.sqrt(composite_trapezoid(squared, dt))


def build_age_grid(dt, a_max=A_MAX):
    """
    构造年龄网格 a_i = i * dt。
    """
    n_age = int(round(a_max / dt))
    if not np.isclose(n_age * dt, a_max):
        raise ValueError("dt must divide the age interval exactly.")
    return np.linspace(0.0, a_max, n_age + 1)


def scalar_layer_factor(dt, age_midpoint):
    """
    计算 Example 1 的单个年龄层推进因子。

    实现逻辑:
    1. 由于 Example 1 的初值、死亡率、出生率都与空间无关，
       且边界条件是零通量 Neumann 条件，数值解始终保持空间常数模态。
    2. 对空间常数模态有 S * 1 = 0，因此空间离散系统退化为标量递推。

    对应公式:
        U_i^n = q_i * U_{i-1}^{n-1}
        q_i = (1 / dt - 0.5 * mu(a_{i-1/2})) / (1 / dt + 0.5 * mu(a_{i-1/2}))
    """
    mu_value = float(mu_example_1(age_midpoint))
    numerator = 1.0 / dt - 0.5 * mu_value
    denominator = 1.0 / dt + 0.5 * mu_value
    return numerator / denominator


def solve_uniform_example_1(dt, h, t_stop, store_history=False):
    """
    用空间常数模态求解 Example 1。

    实现逻辑:
    1. 先实例化 framework 中的 1D P1 有限元对象，保持与主框架接口一致。
    2. 由于 Example 1 完全不依赖空间 x，真实数值解在每个空间节点都相同。
    3. 因此只推进年龄方向上的标量剖面，再在需要时扩展成全空间状态。
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

    time_grid = None
    total_population = None
    if store_history:
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

        if store_history:
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


def interpolate_reference_to_grid(reference_age_grid, reference_profile, target_age_grid):
    """
    将参考解从细年龄网格插值到目标年龄网格。
    """
    return np.interp(target_age_grid, reference_age_grid, reference_profile)


def relative_error_at_t_eval(dt, reference_solution):
    """
    计算 t = 3 时刻的相对误差。

    对应公式:
        E = ||u_dt - u_ref|| / ||u_ref||

    这里采用年龄方向带梯形权重的离散 L2 范数。
    由于 Example 1 的解在空间上是常数，上式与空间-年龄 L2 相对误差只差一个会相消的常数因子。
    """
    coarse_solution = solve_uniform_example_1(dt=dt, h=H_COARSE, t_stop=T_EVAL, store_history=False)

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
    计算 Example 1 在 t = 3 时刻的误差和收敛阶。
    """
    reference_solution = solve_uniform_example_1(
        dt=DT_REFERENCE,
        h=H_REFERENCE,
        t_stop=T_EVAL,
        store_history=False,
    )

    errors = []
    rates = [np.nan]

    for dt in DT_LIST:
        error = relative_error_at_t_eval(dt, reference_solution)
        errors.append(error)

    errors = np.array(errors)
    for i in range(1, len(DT_LIST)):
        rate = np.log(errors[i - 1] / errors[i]) / np.log(DT_LIST[i - 1] / DT_LIST[i])
        rates.append(rate)

    rates = np.array(rates)
    return errors, rates, reference_solution


def print_convergence_table(errors, rates):
    """
    打印数值误差表，并与论文 Table 1 做简单对照。
    """
    print("Example 1: relative errors at t = 3 years")
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


def plot_error_curve(errors):
    """
    绘制误差与时间步长的双对数图。
    """
    dt_array = np.array(DT_LIST)
    reference_line = errors[0] * (dt_array / dt_array[0]) ** 2

    plt.figure(figsize=(8, 5))
    plt.loglog(dt_array, errors, "o-", linewidth=2.0, markersize=7, label="Computed error")
    plt.loglog(dt_array, PAPER_ERRORS, "s--", linewidth=1.8, markersize=6, label="Paper Table 1")
    plt.loglog(dt_array, reference_line, "k:", linewidth=1.5, label="Second-order slope")
    plt.xlabel("Time step dt")
    plt.ylabel("Relative error at t = 3")
    plt.title("Example 1: temporal convergence")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()


def plot_total_population(reference_history):
    """
    绘制总人口数随时间变化的曲线。

    由于 Example 1 的解与空间无关，任意 x 位置的总人口数曲线相同。
    """
    plt.figure(figsize=(8, 5))
    plt.plot(
        reference_history["time_grid"],
        reference_history["total_population"],
        linewidth=2.0,
        label="Reference solution at x = 0.1 km",
    )
    plt.xlabel("Time (years)")
    plt.ylabel("Total population number")
    plt.title("Example 1: total population evolution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


def plot_age_profiles(reference_solution):
    """
    绘制 t = 3 时刻不同时间步长下的年龄剖面。
    """
    coarse_dt_values = [1.0 / 10.0, 1.0 / 20.0, 1.0 / 80.0]
    plt.figure(figsize=(8, 5))

    plt.plot(
        reference_solution["age_grid"],
        reference_solution["profile"],
        linewidth=2.5,
        label="Reference dt = 1/240",
    )

    for dt in coarse_dt_values:
        solution = solve_uniform_example_1(dt=dt, h=H_COARSE, t_stop=T_EVAL, store_history=False)
        plt.plot(
            solution["age_grid"],
            solution["profile"],
            linewidth=1.8,
            label=f"dt = 1/{int(round(1.0 / dt))}",
        )

    plt.xlabel("Age (years)")
    plt.ylabel("u(a, t = 3)")
    plt.title("Example 1: age profile at t = 3 years")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="Reproduce Example 1 in Section 6.")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Compute the tables without showing figures.",
    )
    args = parser.parse_args()

    errors, rates, reference_solution = compute_convergence_table()
    print_convergence_table(errors, rates)

    reference_history = solve_uniform_example_1(
        dt=DT_REFERENCE,
        h=H_REFERENCE,
        t_stop=T_FINAL,
        store_history=True,
    )

    if not args.no_plot:
        plot_error_curve(errors)
        plot_total_population(reference_history)
        plot_age_profiles(reference_solution)
        plt.show()


if __name__ == "__main__":
    main()
