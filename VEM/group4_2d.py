import argparse
#time
import time

import matplotlib.pyplot as plt
import numpy as np

from group3_2d import (
    T_EVAL,
    build_model,
    solve_manufactured_problem,
    space_age_h1_relative_error,
    space_age_l2_relative_error,
)


NX_COUPLED_LIST = [8, 16, 32, 64, 128]
#DT_TO_H_RATIO = 0.5
#DT_TO_H_RATIO = 1.0


def coupled_time_step(nx_elements):
    """取 dt = da 近似 h^1/2，并保证能整除年龄区间和终止时间。"""
    h_value = 1.0 / nx_elements
    target_dt = h_value ** 0.5
    n_steps = int(round(1.0 / target_dt))
    return 1.0 / n_steps


def compute_coupled_convergence():
    """计算空间-时间同步加密收敛表。"""
    results = []

    for nx_elements in NX_COUPLED_LIST:
        h_value = 1.0 / nx_elements
        dt = coupled_time_step(nx_elements)
        model = build_model(nx_elements)
        age_grid, numerical_states = solve_manufactured_problem(model, dt, T_EVAL)
        l2_error = space_age_l2_relative_error(model, numerical_states, age_grid, T_EVAL)
        h1_error = space_age_h1_relative_error(model, numerical_states, age_grid, T_EVAL)

        results.append(
            {
                "nx": nx_elements,
                "h": h_value,
                "dt": dt,
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


def print_coupled_table(results):
    """打印空间-时间同步加密误差表。"""
    print("2D Group-4 Manufactured Solution: coupled space-time convergence")
    print("=" * 116)
    print(
        f"{'nx=ny':>10} {'h':>12} {'dt=da':>12} {'L2 error':>16} {'L2 rate':>10} "
        f"{'H1 error':>16} {'H1 rate':>10}"
    )
    print("-" * 116)

    for row in results:
        l2_rate_string = "-" if np.isnan(row["l2_rate"]) else f"{row['l2_rate']:.4f}"
        h1_rate_string = "-" if np.isnan(row["h1_rate"]) else f"{row['h1_rate']:.4f}"
        print(
            f"{row['nx']:10d} "
            f"{row['h']:12.6f} "
            f"{row['dt']:12.6f} "
            f"{row['l2_error']:16.8e} "
            f"{l2_rate_string:>10} "
            f"{row['h1_error']:16.8e} "
            f"{h1_rate_string:>10}"
        )

    print("-" * 116)


def print_timing_summary(total_elapsed):
    """总计算时间。"""
    print(f"Total computation time before plot: {total_elapsed:.2f} seconds")


def plot_group4_results(results):
    """绘制空间-时间同步加密收敛图。"""
    h_values = np.array([row["h"] for row in results])
    l2_errors = np.array([row["l2_error"] for row in results])
    h1_errors = np.array([row["h1_error"] for row in results])

    l2_reference = l2_errors[0] * (h_values / h_values[0]) ** 2
    h1_reference = h1_errors[0] * (h_values / h_values[0])

    plt.figure(figsize=(7.5, 5.0))
    plt.loglog(h_values, l2_errors, "o-", linewidth=2.0, label="L2 error")
    plt.loglog(h_values, h1_errors, "s-", linewidth=2.0, label="H1 error")
    plt.loglog(h_values, l2_reference, "k--", linewidth=1.5, label="slope 2")
    plt.loglog(h_values, h1_reference, "k:", linewidth=1.5, label="slope 1")
    plt.xlabel("Mesh size h")
    plt.ylabel("Relative error")
    plt.title("Group 4: coupled space-time convergence")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="Run the fourth 2D coupled refinement experiment.")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show the coupled space-time convergence plot.",
    )
    args = parser.parse_args()

    total_start = time.perf_counter()
    results = compute_coupled_convergence()
    total_elapsed = time.perf_counter() - total_start

    print_coupled_table(results)
    print_timing_summary(total_elapsed)

    if args.plot:
        plot_group4_results(results)
        plt.show()


if __name__ == "__main__":
    main()
