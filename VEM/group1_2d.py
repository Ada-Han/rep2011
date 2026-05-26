import argparse
import numpy as np
import matplotlib.pyplot as plt
from framework_2D import ASP_VEM2D


KAPPA = 0.01
BETA = 1.3025
X_LEFT = 0.0
X_RIGHT = 0.2
Y_BOTTOM = 0.0
Y_TOP = 0.2
A_MAX = 15.0
T_FINAL = 12.0
T_EVAL = 3.0
H_COARSE = 0.005

DT_LIST = [1.0 / 10.0, 1.0 / 20.0, 1.0 / 30.0, 1.0 / 40.0, 1.0 / 60.0, 1.0 / 80.0]
DT_REFERENCE = 1.0 / 240.0

PAPER_ERRORS_EXAMPLE_1 = np.array([0.0446, 0.0111, 0.0049, 0.0027, 0.0012, 6.2433e-4])
PAPER_ERRORS_EXAMPLE_2 = np.array([0.0205, 0.0050, 0.0022, 0.0012, 5.3031e-4, 2.8353e-4])


def mu_example_1(age):
    """Example 1 死亡率: mu(a) = 2。"""
    age = np.asarray(age, dtype=float)
    return np.full_like(age, 2.0, dtype=float)


def mu_example_2(age):
    """Example 2 死亡率: mu(a) = 2.457 + 2 a (a - 2)。"""
    age = np.asarray(age, dtype=float)
    return 2.457 + 2.0 * age * (age - 2.0)


def initial_condition(age):
    """论文年龄初值 u0(a)，空间上取常模态。"""
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
    """构造年龄网格 a_i = i * dt， delta a = delta t。"""
    n_age = int(round(a_max / dt))
    if not np.isclose(n_age * dt, a_max):
        raise ValueError("dt must divide the age interval exactly.")
    return np.linspace(0.0, a_max, n_age + 1)


def composite_trapezoid(values, dt):
    """对年龄离散数据做复合梯形积分。"""
    return dt * (0.5 * values[0] + np.sum(values[1:-1]) + 0.5 * values[-1])


def weighted_l2_norm(values, dt):
    """计算年龄方向带梯形权重的离散 L2 范数。"""
    squared_values = values * values
    return np.sqrt(composite_trapezoid(squared_values, dt))


def interpolate_reference_to_grid(reference_age_grid, reference_profile, target_age_grid):
    """把参考年龄剖面插值到目标年龄网格。"""
    return np.interp(target_age_grid, reference_age_grid, reference_profile)


def scalar_birth_update(profile, dt, beta=BETA):
    """空间常模态下标量出生边界更新。"""
    weighted_sum = np.sum(profile[1:-1]) + 0.5 * profile[-1]
    denominator = 1.0 - 0.5 * beta * dt
    if denominator <= 0.0:
        raise ValueError("The birth boundary denominator must stay positive.")
    return (beta * dt / denominator) * weighted_sum


def compute_constant_mode_factor(model, dt, age_midpoint, mu_function):
    """二维矩阵提取空间常模态的单层推进因子。"""
    lhs, rhs = model.build_system_matrices(dt, age_midpoint, mu_function)
    ones = np.ones(model.n_nodes)

    lhs_ones = lhs @ ones
    rhs_ones = rhs @ ones

    factor = np.dot(ones, rhs_ones) / np.dot(ones, lhs_ones)
    collinearity_defect = np.linalg.norm(rhs_ones - factor * lhs_ones, np.inf)
    return factor, collinearity_defect


def scalar_layer_factor(dt, age_midpoint, mu_function):
    """空间常模态下的标量年龄层推进因子 q_i。"""
    mu_value = float(mu_function(age_midpoint))
    numerator = 1.0 / dt - 0.5 * mu_value
    denominator = 1.0 / dt + 0.5 * mu_value
    return numerator / denominator


def solve_reduced_constant_mode(model, dt, mu_function, t_stop, store_history=False):
    """求解"""
    # 构造年龄网格 a_i = i * dt。
    age_grid = build_age_grid(dt, a_max=A_MAX)
    # 年龄层总数，不包含 a = 0 这一层。
    n_age = len(age_grid) - 1
    # 需要推进的时间步数。
    n_time = int(round(t_stop / dt))
    # 保证 t_stop 正好落在时间网格点上。
    if not np.isclose(n_time * dt, t_stop):
        raise ValueError("t_stop must be a multiple of dt.")

    # t = 0 时刻的年龄剖面 c_i^0。
    current_profile = initial_condition(age_grid)

    # 空间常向量，对应 u_h(x,y,a,t)=c(a,t)。
    ones = np.ones(model.n_nodes)
    # 区域面积 |Omega| = 1^T M 1，用于全局人口积分。
    domain_measure = float(ones @ (model.mass_matrix @ ones))

    # 默认不存时间历史，只返回终止时刻结果。
    time_grid = None
    global_population = None
    # 需要画图时，记录每个时间层的全局总人口。
    if store_history:
        time_grid = np.linspace(0.0, t_stop, n_time + 1)
        global_population = np.zeros(n_time + 1)
        global_population[0] = domain_measure * composite_trapezoid(current_profile, dt)

    # 预先计算每个年龄层的标量推进因子 q_i。
    layer_factors = np.zeros(n_age)
    for age_index in range(1, n_age + 1):
        # 当前年龄层对应的中点年龄 a_{i-1/2}。
        age_midpoint = (age_index - 0.5) * dt
        # 空间常模态，U_i^n = q_i U_{i-1}^{n-1}。
        layer_factors[age_index - 1] = scalar_layer_factor(dt, age_midpoint, mu_function)

    # 从 t = 0 逐步推进到 t_stop。
    for time_index in range(1, n_time + 1):
        # 存放当前时间层的新年龄剖面。
        next_profile = np.zeros_like(current_profile)

        # 沿特征线推进年龄层。
        for age_index in range(1, n_age + 1):
            next_profile[age_index] = layer_factors[age_index - 1] * current_profile[age_index - 1]

        # 用出生边界补出 a = 0 的新生人口。
        next_profile[0] = scalar_birth_update(next_profile, dt, beta=BETA)
        # 当前时间层成为下一步的上一时间层。
        current_profile = next_profile

        # 如果画图，记录 P(t)=|Omega| * integral c(a,t) da。
        if store_history:
            global_population[time_index] = domain_measure * composite_trapezoid(current_profile, dt)

    # 把标量年龄剖面复制到所有空间节点，形成二维 FEM 状态数组。
    state = np.tile(current_profile[:, None], (1, model.n_nodes))
    # 返回误差计算和画图需要的数据。
    return {
        "age_grid": age_grid,
        "profile": current_profile,
        "state": state,
        "time_grid": time_grid,
        "global_population": global_population,
        "domain_measure": domain_measure,
    }


def relative_error_at_t_eval(model, dt, mu_function, reference_solution):
    """计算 t = 3 时刻的空间-年龄相对误差。"""
    coarse_solution = solve_reduced_constant_mode(model, dt, mu_function, T_EVAL, store_history=False)
    coarse_age = coarse_solution["age_grid"]
    coarse_profile = coarse_solution["profile"]

    reference_on_coarse = interpolate_reference_to_grid(
        reference_solution["age_grid"],
        reference_solution["profile"],
        coarse_age,
    )

    domain_measure = model.domain_measure_by_quadrature()
    numerator_squared = domain_measure * composite_trapezoid((coarse_profile - reference_on_coarse) ** 2, dt)
    denominator_squared = domain_measure * composite_trapezoid(reference_on_coarse ** 2, dt)
    return np.sqrt(numerator_squared) / np.sqrt(denominator_squared)


def compute_error_table(model, mu_function):
    """生成某个算例在 t = 3 的误差和收敛阶表。"""
    reference_solution = solve_reduced_constant_mode(
        model,
        DT_REFERENCE,
        mu_function,
        T_EVAL,
        store_history=False,
    )

    errors = []
    rates = [np.nan]
    for dt in DT_LIST:
        errors.append(relative_error_at_t_eval(model, dt, mu_function, reference_solution))

    errors = np.array(errors)
    for i in range(1, len(DT_LIST)):
        rates.append(np.log(errors[i - 1] / errors[i]) / np.log(DT_LIST[i - 1] / DT_LIST[i]))

    return errors, np.array(rates), reference_solution


def print_error_table(title, errors, rates, paper_errors):
    """误差表"""
    print(title)
    print("-" * 86)
    print(f"{'dt':>10} {'error':>16} {'rate':>12} {'paper_error':>16} {'abs_diff':>16}")
    print("-" * 86)

    for i, dt in enumerate(DT_LIST):
        rate_string = "-" if i == 0 else f"{rates[i]:.4f}"
        abs_diff = abs(errors[i] - paper_errors[i])
        print(
            f"{dt:10.6f} "
            f"{errors[i]:16.8e} "
            f"{rate_string:>12} "
            f"{paper_errors[i]:16.8e} "
            f"{abs_diff:16.8e}"
        )

    print("-" * 86)


def plot_global_population_curves(history_example_1, history_example_2):
    """绘制总人口曲线 P(t)。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    axes[0].plot(history_example_1["time_grid"], history_example_1["global_population"], linewidth=2.0)
    axes[0].set_title("Example 1: global population")
    axes[0].set_xlabel("Time (years)")
    axes[0].set_ylabel("Global population")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history_example_2["time_grid"], history_example_2["global_population"], linewidth=2.0, color="tab:red")
    axes[1].set_title("Example 2: global population")
    axes[1].set_xlabel("Time (years)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()


def build_model():
    """构造二维 P1 模型。"""
    return ASP_VEM2D(
        x_left=X_LEFT,
        x_right=X_RIGHT,
        y_bottom=Y_BOTTOM,
        y_top=Y_TOP,
        hx=H_COARSE,
        hy=H_COARSE,
        a_max=A_MAX,
        t_final=T_FINAL,
        kappa=KAPPA,
    )


def main():
    parser = argparse.ArgumentParser(description="Run group-1 2D FEM baseline experiments.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show global population curves for Example 1 and Example 2.",
    )
    args = parser.parse_args()

    model = build_model()

    print("2D Group-1 实验")
    print("=" * 86)
    model.print_mesh_info()
    print("=" * 86)

    errors_1, rates_1, _ = compute_error_table(model, mu_example_1)
    errors_2, rates_2, _ = compute_error_table(model, mu_example_2)

    print_error_table(
        "Example 1: relative errors at t = 3 years on the 2D baseline",
        errors_1,
        rates_1,
        PAPER_ERRORS_EXAMPLE_1,
    )
    print_error_table(
        "Example 2: relative errors at t = 3 years on the 2D baseline",
        errors_2,
        rates_2,
        PAPER_ERRORS_EXAMPLE_2,
    )

    if args.plot:
        history_1 = solve_reduced_constant_mode(model, DT_REFERENCE, mu_example_1, T_FINAL, store_history=True)
        history_2 = solve_reduced_constant_mode(model, DT_REFERENCE, mu_example_2, T_FINAL, store_history=True)
        plot_global_population_curves(history_1, history_2)
        plt.show()


if __name__ == "__main__":
    main()
