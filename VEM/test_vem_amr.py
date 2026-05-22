# 文件名：test_vem_amr.py
import numpy as np
import matplotlib.pyplot as plt
import os
from framework_2D import ASP_VEM2D 

image_dir = "/mnt/Data_D/Academic_Works/code/rep2011/image"
os.makedirs(image_dir, exist_ok=True)
save_path = os.path.join(image_dir, "VEM_AMR_Error_Test.png")

def test_vem_gradient_recovery():
    print("==================================================")
    print("🚀 开始测试: 年龄结构种群模型 VEM 求解 + 梯度恢复自适应评估")
    print("==================================================")

    # 1. 物理参数与网格设置
    dt = 0.1
    t_final = 0.5
    A_max = 2.0  # 最大年龄
    hx = 0.05    # 网格步长（可以调小一点看更精细的马赛克）
    
    # 实例化你强大的 VEM 求解器
    print("1. 正在初始化 VEM 网格与系统矩阵...")
    vem = ASP_VEM2D(
        x_left=0.0, x_right=1.0, 
        y_bottom=0.0, y_top=1.0, 
        hx=hx, hy=hx, 
        a_max=A_max, 
        t_final=t_final, 
        kappa=0.01  # 扩散系数
    )
    vem.print_mesh_info()

    # 2. 定义人口模型的死亡率与出生率
    def mu_function(age):
        return 0.1 * age  # 随年龄增长死亡率略微增加

    def birth_function(current_states, dt):
        # 简单梯形积分算出生率，假设生育率 beta=0.5
        beta = 0.5
        weighted_sum = np.sum(current_states[1:-1, :], axis=0) + 0.5 * current_states[-1, :]
        return (beta * dt / (1.0 - 0.5 * beta * dt)) * weighted_sum

    # 3. 设置初始条件 (造一个局部的人口聚集高峰，这样梯度的变化才会剧烈)
    print("\n2. 设置初始人口分布 (局部高斯峰)...")
    n_age = int(round(A_max / dt)) + 1
    initial_states = np.zeros((n_age, vem.n_nodes))
    
    # 在地图中心偏左放一个人口峰值
    for i in range(vem.n_nodes):
        x, y = vem.nodes[i]
        # 高斯分布，中心在 (0.4, 0.5)
        initial_states[0, i] = 10.0 * np.exp(-50.0 * ((x - 0.4)**2 + (y - 0.5)**2))

    # 4. 开始演化求解
    print("\n3. 开始时间推进求解...")
    current_states = vem.solve(dt, initial_states, mu_function, birth_function, t_stop=t_final)
    print("✅ 求解完成！")

    # =====================================================================
    # 🌟 5. 核心戏码：调用我们今晚写的“梯度恢复后处理”技能！
    # =====================================================================
    print("\n4. 🚀 正在进行 VEM 梯度恢复自适应误差评估...")
    # 我们来评估年龄层 0（新生儿）在 final 时刻的空间分布误差
    target_age_index = 0  
    
    # 这一行就是检验你今晚战果的地方！
    eta_array = vem.compute_gradient_recovery_eta(current_states, target_age_index)

    marked_ids = vem.mark_elements(eta_array, theta=0.1)

    print(" 误差评估完成！")
    print(f"   - 最大误差 (eta_max): {np.max(eta_array):.6f}")
    print(f"   - 最小误差 (eta_min): {np.min(eta_array):.6f}")
    print(f"   - 平均误差 (eta_avg): {np.mean(eta_array):.6f}")

    print(f"为了消灭 10% 的误差，我们需要切碎 {len(marked_ids)} 个网格！")
    print(f"这些网格的 ID 是: {marked_ids}")

    # =====================================================================
    # 6. 可视化：让大弟子享受成就感的时刻
    # =====================================================================
    print("\n5. 📊 正在绘制网格误差热力图...")
    centroids_x = []
    centroids_y = []
    
    # 提取每个网格的中心坐标
    for element in vem.elements:
        coords = vem.nodes[element]
        cx, cy = np.mean(coords, axis=0)
        centroids_x.append(cx)
        centroids_y.append(cy)

    plt.figure(figsize=(10, 8))
    
    # 画数值解的等高线背景（看看人口长什么样）
    X = vem.nodes[:, 0]
    Y = vem.nodes[:, 1]
    U = current_states[target_age_index, :]
    plt.tricontour(X, Y, vem.elements, U, levels=10, colors='gray', alpha=0.3, linewidths=0.5)

    # 画出误差指标的散点热力图
    scatter = plt.scatter(centroids_x, centroids_y, c=eta_array, cmap='jet', s=80, marker='s', edgecolors='none')
    plt.colorbar(scatter, label='Error Indicator $\\eta_K$ (Gradient Recovery)')
    
    plt.title(f'VEM Adaptive Error Indicator at t={t_final}, age_idx={target_age_index}\n(Gray contours show population density)')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    
    # 标记出误差最大的前 10% 的网格（这就是下一步你需要切碎的网格！）
    threshold = np.percentile(eta_array, 90)
    for i, eta in enumerate(eta_array):
        if eta >= threshold:
            plt.plot(centroids_x[i], centroids_y[i], 'r.', markersize=3)
            
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print("✅ 图像已保存为 VEM_AMR_Error_Test.png")
    plt.show()

if __name__ == "__main__":
    test_vem_gradient_recovery()
