import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.sparse.linalg import splu
from framework_2D import ASP_VEM2D
from AMR_grad_recovery import refine_mesh, interpolate_states

image_dir = "/mnt/Data_D/Academic_Works/code/rep2011/image"
os.makedirs(image_dir,exist_ok=True)


def plot_mesh(nodes, elements, ax, title=""):
    """绘制多边形网格。"""
    polygons = [nodes[elem] for elem in elements]
    collection = PolyCollection(polygons, facecolors='none',
                                edgecolors='k', linewidths=0.5)
    ax.add_collection(collection)
    ax.set_xlim(nodes[:, 0].min() - 0.02, nodes[:, 0].max() + 0.02)
    ax.set_ylim(nodes[:, 1].min() - 0.02, nodes[:, 1].max() + 0.02)
    ax.set_aspect('equal')
    ax.set_title(title)


def run(model, dt, n_time, mu, kappa, birth_state):
    """从零初值求解，a=0 始终保持为 birth_state（常 Dirichlet 边界）。"""
    n_age = int(round(model.a_max / dt))
    states = np.zeros((n_age + 1, model.n_nodes))
    states[0, :] = birth_state

    D = kappa * model.stiffness_matrix + mu * model.mass_matrix
    lhs = (1.0 / dt) * model.mass_matrix + 0.5 * D
    rhs = (1.0 / dt) * model.mass_matrix - 0.5 * D
    solver = splu(lhs.tocsc())

    for _ in range(n_time):
        new_states = np.zeros_like(states)
        for i in range(1, n_age + 1):
            new_states[i, :] = solver.solve(rhs @ states[i - 1, :])
        new_states[0, :] = birth_state
        states = new_states

    return states


# ── 参数 ──
NX = 64
KAPPA = 0.01
MU = 2.0
A_MAX = 1.0
T_FINAL = 1.0
DT_TO_H = 4.0

# 高斯峰
CENTER = np.array([0.3, 0.7])
SIGMA = 0.05

MAX_ROUNDS = 10
THETA = 0.5
TOL = 1e-4

# ── 初始网格 ──
h = 1.0 / NX
dt = DT_TO_H * h
n_time = int(round(T_FINAL / dt))
dt_a = dt
n_age = int(round(A_MAX / dt_a))

model = ASP_VEM2D(x_left=0, x_right=1, y_bottom=0, y_top=1,
                  hx=h, hy=h, a_max=A_MAX, t_final=T_FINAL, kappa=KAPPA)

nodes = model.nodes.copy()
elements = [model.elements[i].copy() for i in range(model.n_elements)]

# 初始状态: a=0 高斯峰（常 Dirichlet 边界），其余为零
birth_state = np.exp(-np.sum((nodes - CENTER) ** 2, axis=1) / SIGMA ** 2)
states = np.zeros((n_age + 1, len(nodes)))
states[0, :] = birth_state

# ── AMR 循环 ──
mesh_snapshots = [(nodes.copy(), [e.copy() for e in elements], "Round 0 (initial)")]
eta_snapshots = []

for rd in range(MAX_ROUNDS):
    print(f"=== Round {rd} ===  nodes={len(nodes)}, elements={len(elements)}")

    model = ASP_VEM2D(nodes=nodes, elements=elements,
                      a_max=A_MAX, t_final=T_FINAL, kappa=KAPPA)
    birth_state = np.exp(-np.sum((nodes - CENTER) ** 2, axis=1) / SIGMA ** 2)
    states = run(model, dt, n_time, MU, KAPPA, birth_state)

    eta = model.estimate_gradient_recovery_over_ages(states, dt_a)
    print(f"  max(eta)={np.max(eta):.6e},  mean(eta)={np.mean(eta):.6e}")
    eta_snapshots.append((eta.copy(), [e.copy() for e in elements], nodes.copy()))

    if np.max(eta) < TOL:
        print("  converged, stop.")
        break

    marked = model.mark_elements(eta, theta=THETA)
    print(f"  marked {len(marked)}/{len(elements)} elements")

    new_nodes, new_elements, e2m, cid = refine_mesh(nodes, elements, marked)
    states = interpolate_states(states, elements, len(nodes),
                                len(new_nodes), e2m, cid)
    nodes, elements = new_nodes, new_elements
    mesh_snapshots.append((nodes.copy(), [e.copy() for e in elements],
                           f"Round {rd+1}"))

print(f"\nDone. Final mesh: {len(nodes)} nodes, {len(elements)} elements")

# ── 画图 ──
# 1. 网格对比：初始 vs 最终
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
snap_init = mesh_snapshots[0]
snap_final = mesh_snapshots[-1]
plot_mesh(snap_init[0], snap_init[1], axes[0],
          f"{snap_init[2]}: {len(snap_init[1])} elements")
plot_mesh(snap_final[0], snap_final[1], axes[1],
          f"{snap_final[2]}: {len(snap_final[1])} elements")
plt.tight_layout()
plt.savefig(os.path.join(image_dir, "amr_mesh_comparison.png"), dpi=450)
print(f"Saved: {image_dir}/amr_mesh_comparison.png")

# 2. 每轮误差分布
n_snap = len(eta_snapshots)
fig, axes = plt.subplots(1, n_snap, figsize=(5 * n_snap, 4))
if n_snap == 1:
    axes = [axes]
for idx, (eta_vals, elems, nds) in enumerate(eta_snapshots):
    polygons = [nds[e] for e in elems]
    colors = eta_vals / (np.max(eta_vals) + 1e-30)
    collection = PolyCollection(polygons, facecolors=plt.cm.hot(colors),
                                edgecolors='k', linewidths=0.3)
    axes[idx].add_collection(collection)
    axes[idx].set_xlim(nds[:, 0].min() - 0.02, nds[:, 0].max() + 0.02)
    axes[idx].set_ylim(nds[:, 1].min() - 0.02, nds[:, 1].max() + 0.02)
    axes[idx].set_aspect('equal')
    axes[idx].set_title(f"Round {idx} error (max={np.max(eta_vals):.2e})")
plt.tight_layout()
plt.savefig(os.path.join(image_dir, "amr_error_distribution.png"), dpi=450)
print(f"Saved: {image_dir}/amr_error_distribution.png")

# 3. 解的分布 (a=0, 最终网格)
fig, ax = plt.subplots(1, 1, figsize=(6, 5))
final_nodes = mesh_snapshots[-1][0]
final_elems = mesh_snapshots[-1][1]
sol_a0 = states[0, :]
polygons = [final_nodes[e] for e in final_elems]
vmin, vmax = sol_a0.min(), sol_a0.max()
norm = plt.Normalize(vmin=vmin, vmax=vmax + 1e-30)
colors = plt.cm.viridis(norm(np.array([np.mean(sol_a0[e]) for e in final_elems])))
collection = PolyCollection(polygons, facecolors=colors, edgecolors='k', linewidths=0.3)
ax.add_collection(collection)
ax.set_xlim(final_nodes[:, 0].min() - 0.02, final_nodes[:, 0].max() + 0.02)
ax.set_ylim(final_nodes[:, 1].min() - 0.02, final_nodes[:, 1].max() + 0.02)
ax.set_aspect('equal')
ax.set_title("Solution at a=0 (final mesh)")
sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
plt.colorbar(sm, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(image_dir, "amr_solution_a0.png"), dpi=450)
print(f"Saved: {image_dir}/amr_solution_a0.png")

plt.show()
