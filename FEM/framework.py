import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


class ASP_FEM1D:
    # initialization
    def __init__(
        self,
        x_left=0.0,
        x_right=0.2,
        h=0.005,
        a_max=15.0,
        t_final=12.0,
        kappa=0.01,
    ):

        self.x_left = x_left
        self.x_right = x_right
        self.h = h
        self.a_max = a_max
        self.t_final = t_final
        self.kappa = kappa

        self.nodes, self.elements = self.read_mesh_date()
        self.n_nodes = len(self.nodes)
        self.n_elements = len(self.elements)

        self.mass_matrix = self.build_mass_matrix()
        self.stiffness_matrix = self.build_stiffness_matrix()

    # 1D uniform mesh
    def read_mesh_date(self):
        """
        生成一维均匀网格的节点数组和单元连接关系。

        逻辑:
        1. 用区间长度除以步长 h 得到单元数。
        2. 生成节点 x_i = x_left + i h。
        3. 每个单元由相邻两个节点 [i, i+1] 构成。

        对应离散对象:
        - 空间节点:
            x_i = x_left + i h
        - P1 单元:
            K_e = [x_i, x_{i+1}]
        """
        interval_length = self.x_right - self.x_left
        n_elements = int(round(interval_length / self.h))
        if not np.isclose(n_elements * self.h, interval_length):
            raise ValueError("h must divide the spatial interval exactly.")

        n_nodes = n_elements + 1
        nodes = np.zeros((n_nodes, 1))
        for i in range(n_nodes):
            nodes[i, 0] = self.x_left + i * self.h

        elements = np.zeros((n_elements, 2), dtype=int)
        for i in range(n_elements):
            elements[i] = [i, i + 1]

        return nodes, elements

    # P1 local mass matrix
    def local_mass_matrix(self, element_nodes):
        """
        计算一维 P1 单元上的局部质量矩阵。

        逻辑:
        1. 取出单元两个端点坐标，得到单元长度 h_e。
        2. 直接使用线性单元质量矩阵的解析表达式。

        对应公式:
            M_e = ∫_{K_e} N^T N dx
                = h_e / 6 * [[2, 1],
                             [1, 2]]
        """
        x0 = self.nodes[element_nodes[0], 0]
        x1 = self.nodes[element_nodes[1], 0]
        h_e = x1 - x0
        return (h_e / 6.0) * np.array([[2.0, 1.0], [1.0, 2.0]])

    # P1 local stiffness matrix
    def local_stiffness_matrix(self, element_nodes):
        """
        计算一维 P1 单元上的局部刚度矩阵。

        实现逻辑:
        1. 读取单元长度 h_e。
        2. 利用线性基函数导数为常数，直接写出刚度矩阵解析式。

        对应公式:
            S_e = ∫_{K_e} (N')^T N' dx
                = 1 / h_e * [[ 1, -1],
                             [-1,  1]]
        """
        x0 = self.nodes[element_nodes[0], 0]
        x1 = self.nodes[element_nodes[1], 0]
        h_e = x1 - x0
        return (1.0 / h_e) * np.array([[1.0, -1.0], [-1.0, 1.0]])

    # global mass matrix
    def build_mass_matrix(self):
        """
        组装全局质量矩阵 mass_matrix。

        实现逻辑:
        1. 遍历所有单元，逐个计算局部质量矩阵 M_e。
        2. 根据单元到全局节点的映射关系，将 M_e 累加到全局矩阵 M。
        3. 最后转成 CSR 稀疏格式，便于矩阵运算和线性求解。

        对应公式:
            M = Σ_e A_e^T M_e A_e
        其中 A_e 是局部自由度到全局自由度的装配映射。
        """
        mass_matrix = lil_matrix((self.n_nodes, self.n_nodes))

        for element in self.elements:
            local_matrix = self.local_mass_matrix(element)
            for i, node_i in enumerate(element):
                for j, node_j in enumerate(element):
                    mass_matrix[node_i, node_j] += local_matrix[i, j]

        return mass_matrix.tocsr()

    # global stiffness matrix
    def build_stiffness_matrix(self):
        """
        组装全局刚度矩阵 stiffness_matrix。

        逻辑:
        1. 遍历所有单元，逐个计算局部刚度矩阵 S_e。
        2. 按照单元连接关系把 S_e 累加到全局矩阵 S。
        3. 返回 CSR 稀疏矩阵，供扩散项离散使用。

        对应公式:
            S = Σ_e A_e^T S_e A_e
        """
        stiffness_matrix = lil_matrix((self.n_nodes, self.n_nodes))

        for element in self.elements:
            local_matrix = self.local_stiffness_matrix(element)
            for i, node_i in enumerate(element):
                for j, node_j in enumerate(element):
                    stiffness_matrix[node_i, node_j] += local_matrix[i, j]

        return stiffness_matrix.tocsr()

    # P1 local reaction matrix
    def local_reaction_matrix(self, element_nodes, age_midpoint, mu_function):
        """
        计算一维 P1 单元上的局部反应矩阵。

        逻辑:
        1. 在当前年龄中点 a_{i-1/2} 处计算反应系数 mu(a)。
        2. 由于反应项为 mu(a) u，且 mu 仅依赖年龄而不依赖空间，
           所以单元反应矩阵等于该年龄层系数乘以局部质量矩阵。

        对应公式:
            C_e^{(i)} = mu(a_{i-1/2}) M_e

        其中局部质量矩阵
            M_e = h_e / 6 * [[2, 1],
                             [1, 2]]
        """
        mu_value = mu_function(age_midpoint)
        return mu_value * self.local_mass_matrix(element_nodes)

    # reaction matrix C_i = mu(a_{i-1/2}) M
    def build_reaction_matrix(self, age_midpoint, mu_function):
        """
        构造年龄层 i 对应的反应矩阵 C_i。

        逻辑:
        1. 遍历所有单元，逐个计算局部反应矩阵 C_e^{(i)}。
        2. 按照局部到全局的节点映射进行装配。
        3. 最终得到当前年龄层对应的全局反应矩阵 C_i。

        对应公式:
            C_i = mu(a_{i-1/2}) M
        其中
            a_{i-1/2} = (i - 1/2) dt
        """
        reaction_matrix = lil_matrix((self.n_nodes, self.n_nodes))

        for element in self.elements:
            local_matrix = self.local_reaction_matrix(element, age_midpoint, mu_function)
            for i, node_i in enumerate(element):
                for j, node_j in enumerate(element):
                    reaction_matrix[node_i, node_j] += local_matrix[i, j]

        return reaction_matrix.tocsr()

    # system matrices
    def build_system_matrices(self, dt, age_midpoint, mu_function):
        """
        构造单步推进所需的系统矩阵 lhs 和 rhs。

        逻辑:
        1. 先构造反应矩阵 C_i。
        2. 构造扩散矩阵 K_i = kappa S。
        3. 按照时间-年龄方向的二阶特征离散格式，生成左右两端矩阵。

        对应公式:
            K_i = kappa S
            C_i = mu(a_{i-1/2}) M

            lhs = 1/dt M + 1/2 K_i + 1/2 C_i
            rhs = 1/dt M - 1/2 K_i - 1/2 C_i

        从而单步方程写成:
            lhs * U_i^n = rhs * U_{i-1}^{n-1}
        """
        reaction_matrix = self.build_reaction_matrix(age_midpoint, mu_function)
        diffusion_matrix = self.kappa * self.stiffness_matrix

        lhs = (1.0 / dt) * self.mass_matrix + 0.5 * diffusion_matrix + 0.5 * reaction_matrix
        rhs = (1.0 / dt) * self.mass_matrix - 0.5 * diffusion_matrix - 0.5 * reaction_matrix

        return lhs.tocsr(), rhs.tocsr()

    # age layer update
    def advance_age_layer(self, previous_state, dt, age_index, mu_function):
        """
        推进一个年龄层，计算当前时间层的 U_i^n。

        逻辑:
        1. 根据年龄层编号 i 计算中点年龄 a_{i-1/2}。
        2. 构造该年龄层对应的 lhs 和 rhs。
        3. 先算右端 rhs * U_{i-1}^{n-1}。
        4. 再求解线性系统得到 U_i^n。

        对应公式:
            a_{i-1/2} = (i - 1/2) dt

            (1/dt M + 1/2 K_i + 1/2 C_i) U_i^n
            = (1/dt M - 1/2 K_i - 1/2 C_i) U_{i-1}^{n-1}
        """
        age_midpoint = (age_index - 0.5) * dt
        lhs, rhs = self.build_system_matrices(dt, age_midpoint, mu_function)
        right_hand_side = rhs @ previous_state
        return spsolve(lhs, right_hand_side)

    # birth boundary at a = 0
    def apply_birth_boundary(self, current_states, dt, birth_function):
        """
        在 a = 0 处施加出生边界，返回当前时间层的新生人口向量 U_0^n。

        逻辑:
        1. 当前框架不把出生律写死在类内部。
        2. 具体出生公式由外部传入 birth_function(current_states, dt)。
        3. 这样不同论文或不同算例可以复用同一套有限元骨架。

        对应公式:
            U_0^n = B(U_0^n, U_1^n, ..., U_N^n)
        这里 B 是出生边界算子，由外部回调决定。
        """
        return birth_function(current_states, dt)

    # one time step
    def advance_one_time_level(self, previous_states, dt, mu_function, birth_function):
        """
        完成一个完整时间步 n-1 -> n 的推进。

        实现逻辑:
        1. 已知上一时间层的全部年龄状态 previous_states。
        2. 对 i = 1, 2, ..., N 逐层调用年龄推进公式，得到 U_i^n。
        3. 最后根据出生边界计算 U_0^n。

        对应离散流程:
            先求 U_1^n, U_2^n, ..., U_N^n
            再由出生条件求 U_0^n
        """
        n_age = previous_states.shape[0] - 1
        current_states = np.zeros_like(previous_states)

        for age_index in range(1, n_age + 1):
            current_states[age_index, :] = self.advance_age_layer(
                previous_states[age_index - 1, :],
                dt,
                age_index,
                mu_function,
            )

        current_states[0, :] = self.apply_birth_boundary(current_states, dt, birth_function)
        return current_states

    # time main loop
    def solve(self, dt, initial_states, mu_function, birth_function, t_stop=None):
        """
        时间主循环，从初值推进到指定终止时刻。

        逻辑:
        1. 根据 dt 计算时间步数 N_t 和年龄层数 N_a。
        2. 检查外部给定初值 initial_states 的形状是否与当前网格一致。
        3. 在每个时间步调用 advance_one_time_level 更新整条年龄分布。
        4. 返回终止时刻的离散解。

        对应离散格式:
            U^n = T(U^{n-1}),   n = 1, 2, ..., N_t
        其中 T 由
            - 年龄层推进公式
            - 出生边界公式
        共同组成。
        """
        if t_stop is None:
            t_stop = self.t_final

        n_time = int(round(t_stop / dt))
        n_age = int(round(self.a_max / dt))

        if initial_states.shape != (n_age + 1, self.n_nodes):
            raise ValueError("initial_states has inconsistent shape.")

        current_states = initial_states.copy()

        for _ in range(n_time):
            current_states = self.advance_one_time_level(
                current_states,
                dt,
                mu_function,
                birth_function,
            )

        return current_states


def trapezoidal_birth(beta):
    """
    构造一个基于梯形公式的出生边界函数。

    逻辑:
    1. 外层函数先固定生育系数 beta。
    2. 内层 birth_function 在当前时间层上对年龄积分做梯形离散。
    3. 这里采用的是把 0.5 * beta * dt * U_0^n 移到左端后的显式写法。

    对应公式:
        U_0^n = beta * dt * [Σ_{i=1}^{N-1} U_i^n + 1/2 U_N^n] / (1 - 1/2 beta dt)

    这与原始梯形边界
        U_0^n = beta * dt * [1/2 U_0^n + Σ_{i=1}^{N-1} U_i^n + 1/2 U_N^n]
    是等价的。
    """
    def birth_function(current_states, dt):
        """
        对固定 beta 执行一次出生边界的梯形积分计算。

        实现逻辑:
        1. 对当前时间层的年龄状态做梯形求和。
        2. 将含 U_0^n 的项移到分母，直接解出 U_0^n。

        对应公式:
            weighted_sum = Σ_{i=1}^{N-1} U_i^n + 1/2 U_N^n
            U_0^n = (beta * dt / (1 - 1/2 beta dt)) * weighted_sum
        """
        weighted_sum = np.sum(current_states[1:-1, :], axis=0) + 0.5 * current_states[-1, :]
        denom = 1.0 - 0.5 * beta * dt
        if denom <= 0.0:
            raise ValueError("出生边界分母必须为正数")
        return (beta * dt / denom) * weighted_sum
 
    return birth_function
