import math
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


class ASP_VEM2D:
    def __init__(
        self,
        x_left=0.0,
        x_right=0.2,
        y_bottom=0.0,
        y_top=0.2,
        hx=0.005,
        hy=None,
        a_max=15.0,
        t_final=12.0,
        kappa=0.01,
        nodes=None,
        elements=None,
    ):
        """初始化二维 VEM 框架，装配 M, S。"""
        if hy is None:
            hy = hx

        self.x_left = x_left
        self.x_right = x_right
        self.y_bottom = y_bottom
        self.y_top = y_top
        self.hx = hx
        self.hy = hy
        self.a_max = a_max
        self.t_final = t_final
        self.kappa = kappa

        if nodes is not None and elements is not None:
            self.nodes = np.asarray(nodes, dtype=float)
            self.elements = np.asarray(elements, dtype=int)
            self.auto_generated = False
        else:
            self.nodes, self.elements = self.build_rectangular_mesh()
            self.auto_generated = True

        if self.elements.ndim != 2 or self.elements.shape[1] != 3:
            raise NotImplementedError("This framework currently only supports P1 triangular elements.")

        self.n_nodes = len(self.nodes)
        self.n_elements = len(self.elements)

        self.mass_matrix = self.build_mass_matrix()
        self.stiffness_matrix = self.build_stiffness_matrix()

    def build_rectangular_mesh(self):
        """生成规则矩形区域上的三角形网格。"""
        x_length = self.x_right - self.x_left
        y_length = self.y_top - self.y_bottom

        nx_elements = int(round(x_length / self.hx))
        ny_elements = int(round(y_length / self.hy))

        if not np.isclose(nx_elements * self.hx, x_length):
            raise ValueError("hx must divide the x interval exactly.")
        if not np.isclose(ny_elements * self.hy, y_length):
            raise ValueError("hy must divide the y interval exactly.")

        return self.generate_p1_rectangular_mesh(
            self.x_left,
            self.x_right,
            nx_elements,
            self.y_bottom,
            self.y_top,
            ny_elements,
        )

    def generate_p1_rectangular_mesh(
        self,
        x_left,
        x_right,
        nx_elements,
        y_bottom,
        y_top,
        ny_elements,
    ):
        """把每个小矩形剖成两个 P1 三角形，返回 nodes 和 elements。"""
        nx_nodes = nx_elements + 1
        ny_nodes = ny_elements + 1
        n_nodes = nx_nodes * ny_nodes

        nodes = np.zeros((n_nodes, 2), dtype=float)
        dx = (x_right - x_left) / nx_elements
        dy = (y_top - y_bottom) / ny_elements

        node_id = 0
        for i in range(ny_nodes):
            y_value = y_bottom + i * dy
            for j in range(nx_nodes):
                x_value = x_left + j * dx
                nodes[node_id, :] = [x_value, y_value]
                node_id += 1

        elements = np.zeros((2 * nx_elements * ny_elements, 3), dtype=int)

        element_id = 0
        for i in range(ny_elements):
            for j in range(nx_elements):
                bottom_left = i * nx_nodes + j
                bottom_right = bottom_left + 1
                top_left = (i + 1) * nx_nodes + j
                top_right = top_left + 1

                elements[element_id, :] = [bottom_right, top_right, bottom_left]
                element_id += 1
                elements[element_id, :] = [top_left, bottom_left, top_right]
                element_id += 1

        return nodes, elements

    def print_mesh_info(self):
        """打印网格基本信息。"""
        print("mesh info:")
        if self.auto_generated:
            print(f"  x interval: [{self.x_left}, {self.x_right}]")
            print(f"  y interval: [{self.y_bottom}, {self.y_top}]")
            print(f"  mesh size: hx = {self.hx}, hy = {self.hy}")
            print("  mesh type: auto-generated rectangular P1 triangular mesh")
        else:
            print("  mesh type: user-provided triangular mesh")

        print(f"  number of nodes: {self.n_nodes}")
        print(f"  number of elements: {self.n_elements}")

    def calculate_B_matrix(self, coords, h):
        """矩阵 B (3×N)。"""
        N = len(coords)
        x = coords[:, 0]
        y = coords[:, 1]
        x_next = np.roll(x, -1)
        x_prev = np.roll(x, 1)
        y_next = np.roll(y, -1)
        y_prev = np.roll(y, 1)

        B = np.zeros((3, N), dtype=float)
        B[0, :] = 1.0 / N
        B[1, :] = 0.5 * (y_next - y_prev) / h
        B[2, :] = 0.5 * (x_prev - x_next) / h
        return B

    def calculate_D_matrix(self, coords, xc, yc, h):
        """矩阵 D (N×3)。"""
        N = len(coords)
        D = np.zeros((N, 3), dtype=float)
        D[:, 0] = 1.0
        D[:, 1] = (coords[:, 0] - xc) / h
        D[:, 2] = (coords[:, 1] - yc) / h
        return D

    def calculate_G_matrix(self, B, D):
        """矩阵 G (3×3)。"""
        G = B @ D
        return G

    def calculate_element_area(self, element_nodes):
        """鞋带公式计算多边形单元面积。"""
        node_coords = self.nodes[element_nodes]
        n = len(element_nodes)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += node_coords[i, 0] * node_coords[j, 1]
            area -= node_coords[j, 0] * node_coords[i, 1]
        area = 0.5 * abs(area)
        if area <= np.finfo(float).eps:
            raise ValueError("Degenerate polygonal element detected.")
        return area

    def calculate_element_stiffness(self, element):
        """VEM 单元刚度矩阵: S_e = S_consistent + S_stabilization。"""
        node_coords = self.nodes[element]
        n_vertices = len(element)
        area = self.calculate_element_area(element)
        diameter = self.calculate_element_diameter(node_coords)

        B = self.calculate_B_matrix(node_coords, diameter)
        D = self.calculate_D_matrix(node_coords, *np.mean(node_coords, axis=0), diameter)
        G = self.calculate_G_matrix(B, D)
        Pi_star = np.linalg.solve(G, B)

        # 一致性部分
        Pi_grad = Pi_star[1:3, :]
        S_consistent = (area / diameter**2) * (Pi_grad.T @ Pi_grad)

        # 稳定化部分
        Pi = D @ Pi_star
        I_minus_Pi = np.eye(n_vertices) - Pi
        S_stabilization = I_minus_Pi.T @ I_minus_Pi

        return S_consistent + S_stabilization

    def calculate_element_diameter(self, node_coords):
        """计算单元直径"""
        n = len(node_coords)
        diameter = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(node_coords[i] - node_coords[j])
                if dist > diameter:
                    diameter = dist
        return diameter

    def calculate_element_mass(self, element):
        """VEM 单元质量矩阵(集中质量)"""
        area = self.calculate_element_area(element)
        n_vertices = len(element)
        return (area / n_vertices) * np.eye(n_vertices)

    def calculate_element_reaction(self, element, age_midpoint, mu_function):
        """单元反应矩阵: C_e = mu(a_{i-1/2}) M_e。"""
        mu_value = float(mu_function(age_midpoint))
        return mu_value * self.calculate_element_mass(element)

    def build_mass_matrix(self):
        """全局质量矩阵 M。"""
        mass_matrix = lil_matrix((self.n_nodes, self.n_nodes))

        for element in self.elements:
            element_matrix = self.calculate_element_mass(element)
            for i, node_i in enumerate(element):
                for j, node_j in enumerate(element):
                    mass_matrix[node_i, node_j] += element_matrix[i, j]

        return mass_matrix.tocsr()

    def build_stiffness_matrix(self):
        """全局刚度矩阵 S。"""
        stiffness_matrix = lil_matrix((self.n_nodes, self.n_nodes))

        for element in self.elements:
            element_matrix = self.calculate_element_stiffness(element)
            for i, node_i in enumerate(element):
                for j, node_j in enumerate(element):
                    stiffness_matrix[node_i, node_j] += element_matrix[i, j]

        return stiffness_matrix.tocsr()

    def build_reaction_matrix(self, age_midpoint, mu_function):
        """当前年龄层的全局反应矩阵 C_i。"""
        reaction_matrix = lil_matrix((self.n_nodes, self.n_nodes))

        for element in self.elements:
            element_matrix = self.calculate_element_reaction(element, age_midpoint, mu_function)
            for i, node_i in enumerate(element):
                for j, node_j in enumerate(element):
                    reaction_matrix[node_i, node_j] += element_matrix[i, j]

        return reaction_matrix.tocsr()

    def domain_measure_by_quadrature(self):
        """计算区域面积（各单元面积之和）。"""
        total_measure = 0.0
        for element in self.elements:
            total_measure += self.calculate_element_area(element)
        return total_measure

    def build_system_matrices(self, dt, age_midpoint, mu_function):
        """单层推进矩阵: lhs = M/dt + (K_i+C_i)/2, rhs = M/dt - (K_i+C_i)/2。"""
        reaction_matrix = self.build_reaction_matrix(age_midpoint, mu_function)
        diffusion_matrix = self.kappa * self.stiffness_matrix

        lhs = (1.0 / dt) * self.mass_matrix + 0.5 * diffusion_matrix + 0.5 * reaction_matrix
        rhs = (1.0 / dt) * self.mass_matrix - 0.5 * diffusion_matrix - 0.5 * reaction_matrix
        return lhs.tocsr(), rhs.tocsr()

    def advance_age_layer(self, previous_state, dt, age_index, mu_function):
        """推进年龄层: 从 U_{i-1}^{n-1} 解出 U_i^n。"""
        age_midpoint = (age_index - 0.5) * dt
        lhs, rhs = self.build_system_matrices(dt, age_midpoint, mu_function)
        right_hand_side = rhs @ previous_state
        return spsolve(lhs, right_hand_side)

    def apply_birth_boundary(self, current_states, dt, birth_function):
        """调用外部出生边界函数，得到当前时间层的 U_0^n。"""
        return birth_function(current_states, dt)

    def advance_one_time_level(self, previous_states, dt, mu_function, birth_function):
        """推进一个完整时间层: 先算 i>=1，再补出生边界 i=0。"""
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

    def solve(self, dt, initial_states, mu_function, birth_function, t_stop=None):
        """重复调用单步推进，返回最后的全部年龄层状态。"""
        if t_stop is None:
            t_stop = self.t_final

        n_time = int(round(t_stop / dt))
        current_states = initial_states.copy()

        for _ in range(n_time):
            current_states = self.advance_one_time_level(
                current_states,
                dt,
                mu_function,
                birth_function,
            )

        return current_states

    # ── 数值积分 ──────────────────────────────────────────────────

    def get_quadrature(self):
        """三角形 3 点高斯积分。"""
        return [
            (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0),
            (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
            (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
        ]

    # ── VEM 后处理：基于 Π 投影的解评估 ──────────────────────────

    def evaluate_solution_at_point(self, local_values, point, node_coords, diameter):
        """通过 VEM Π 投影在任意点处评估数值解的值。

        local_values: 单元自由度向量（节点值）
        point: 评估点坐标 [x, y]
        node_coords: 单元顶点坐标 (N×2)
        diameter: 单元直径
        """
        xc, yc = np.mean(node_coords, axis=0)
        B = self.calculate_B_matrix(node_coords, diameter)
        D_vertices = self.calculate_D_matrix(node_coords, xc, yc, diameter)
        G = self.calculate_G_matrix(B, D_vertices)
        Pi_star = np.linalg.solve(G, B)

        # 归一化单项式基在评估点处的值 [1, (x-xc)/h, (y-yc)/h]
        m_point = np.array([1.0, (point[0] - xc) / diameter, (point[1] - yc) / diameter])
        return float(m_point @ Pi_star @ local_values)

    def evaluate_solution_in_element(self, local_values, node_coords, diameter):
        """通过 VEM Π 投影在单元内评估数值解（返回多项式系数）。

        返回 m(x,y) = c0 + c1*(x-xc)/h + c2*(y-yc)/h 的系数 [c0, c1, c2]。
        """
        xc, yc = np.mean(node_coords, axis=0)
        B = self.calculate_B_matrix(node_coords, diameter)
        D_vertices = self.calculate_D_matrix(node_coords, xc, yc, diameter)
        G = self.calculate_G_matrix(B, D_vertices)
        Pi_star = np.linalg.solve(G, B)
        return Pi_star @ local_values

    def evaluate_gradient_in_element(self, local_values, node_coords, diameter):
        """通过 VEM Π 投影评估单元内的梯度（常数向量）。

        返回 [du/dx, du/dy]。
        """
        coefficients = self.evaluate_solution_in_element(local_values, node_coords, diameter)
        # m = c0 + c1*(x-xc)/h + c2*(y-yc)/h  =>  dm/dx = c1/h, dm/dy = c2/h
        return np.array([coefficients[1] / diameter, coefficients[2] / diameter])

    # ── 多边形积分工具 ───────────────────────────────────────────

    def _polygon_centroid(self, coords):
        """用有符号面积公式计算多边形形心。"""
        x = coords[:, 0]
        y = coords[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        cross = x * y_next - x_next * y
        signed_area = 0.5 * np.sum(cross)
        if abs(signed_area) < 1e-15:
            return np.mean(coords, axis=0)
        cx = np.sum((x + x_next) * cross) / (6.0 * signed_area)
        cy = np.sum((y + y_next) * cross) / (6.0 * signed_area)
        return np.array([cx, cy])

    def _integrate_on_polygon(self, node_coords, integrand_func):
        """把多边形从形心剖分成子三角形，在每个子三角形上做高斯积分。

        integrand_func(px, py) -> 被积函数在点 (px, py) 处的值。
        返回积分结果。
        """
        quadrature = self.get_quadrature()
        center = self._polygon_centroid(node_coords)
        n_vertices = len(node_coords)
        result = 0.0

        for i in range(n_vertices):
            a = center
            b = node_coords[i]
            c = node_coords[(i + 1) % n_vertices]

            tri_area = 0.5 * abs(
                (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
            )
            if tri_area < 1e-15:
                continue

            for xi, eta, weight in quadrature:
                lam = 1.0 - xi - eta
                px = lam * a[0] + xi * b[0] + eta * c[0]
                py = lam * a[1] + xi * b[1] + eta * c[1]
                result += 2.0 * tri_area * weight * integrand_func(px, py)

        return result

    # ── 误差计算（基于 VEM Π 投影，多边形形心剖分积分）────────
    # —— 后验误差部分，包括AMR 等

    def compute_l2_error(self, numerical_states, exact_func, age_grid, time_value):
        """空间-年龄 L2 相对误差。

        exact_func(nodes, age, time) -> 节点处精确值数组。
        通过 VEM Π 投影在积分点处评估数值解，多边形形心剖分积分。
        """
        import math

        dt = age_grid[1] - age_grid[0]
        error_accumulator = 0.0
        exact_accumulator = 0.0

        for age_index, age_value in enumerate(age_grid):
            age_weight = 0.5 if age_index in (0, len(age_grid) - 1) else 1.0
            state_vector = numerical_states[age_index, :]

            for element in self.elements:
                node_coords = self.nodes[element]
                diameter = self.calculate_element_diameter(node_coords)
                xc, yc = self._polygon_centroid(node_coords)

                local_values = state_vector[element]
                coefficients = self.evaluate_solution_in_element(local_values, node_coords, diameter)

                def l2_integrand(px, py):
                    m = np.array([1.0, (px - xc) / diameter, (py - yc) / diameter])
                    numerical_value = float(m @ coefficients)
                    exact_value = float(exact_func(np.array([[px, py]]), age_value, time_value)[0])
                    diff = numerical_value - exact_value
                    return diff * diff

                def ref_integrand(px, py):
                    exact_value = float(exact_func(np.array([[px, py]]), age_value, time_value)[0])
                    return exact_value * exact_value

                error_accumulator += age_weight * self._integrate_on_polygon(node_coords, l2_integrand)
                exact_accumulator += age_weight * self._integrate_on_polygon(node_coords, ref_integrand)

        return math.sqrt(dt * error_accumulator) / math.sqrt(dt * exact_accumulator)

    def compute_h1_error(self, numerical_states, exact_grad_func, age_grid, time_value):
        """空间-年龄 H1 半范数相对误差。

        exact_grad_func(points, age, time) -> 梯度数组 (N, 2)。
        通过 VEM Π 投影评估单元内梯度，多边形形心剖分积分。
        """

        dt = age_grid[1] - age_grid[0]
        error_accumulator = 0.0
        exact_accumulator = 0.0

        for age_index, age_value in enumerate(age_grid):
            age_weight = 0.5 if age_index in (0, len(age_grid) - 1) else 1.0
            state_vector = numerical_states[age_index, :]

            for element in self.elements:
                node_coords = self.nodes[element]
                diameter = self.calculate_element_diameter(node_coords)

                local_values = state_vector[element]
                numerical_gradient = self.evaluate_gradient_in_element(local_values, node_coords, diameter)

                def h1_integrand(px, py):
                    exact_gradient = exact_grad_func(np.array([[px, py]]), age_value, time_value)[0]
                    error_gradient = numerical_gradient - exact_gradient
                    return float(np.dot(error_gradient, error_gradient))

                def ref_integrand(px, py):
                    exact_gradient = exact_grad_func(np.array([[px, py]]), age_value, time_value)[0]
                    return float(np.dot(exact_gradient, exact_gradient))

                error_accumulator += age_weight * self._integrate_on_polygon(node_coords, h1_integrand)
                exact_accumulator += age_weight * self._integrate_on_polygon(node_coords, ref_integrand)

        return math.sqrt(dt * error_accumulator) / math.sqrt(dt * exact_accumulator)

    @staticmethod
    def compute_convergence_rates(errors, mesh_sizes):
        """计算收敛阶。errors 和 mesh_sizes 为等长数组，返回 rates（首个元素为 NaN）。"""
        errors = np.asarray(errors, dtype=float)
        mesh_sizes = np.asarray(mesh_sizes, dtype=float)
        rates = np.full(len(errors), np.nan)
        for i in range(1, len(errors)):
            rates[i] = np.log(errors[i - 1] / errors[i]) / np.log(mesh_sizes[i - 1] / mesh_sizes[i])
        return rates
    
    def compute_gradient_recovery_eta(self, current_states, age_index):
        "VEM 自适应 后处理：梯度恢复估计"

        uh = current_states[age_index,:]

        D_diag = np.zeros(self.n_nodes)
        f_x = np.zeros(self.n_nodes)
        f_y = np.zeros(self.n_nodes)

        # 组装全局平滑梯度系统的矩阵
        for element_id, element in enumerate(self.elements):
            node_coords = self.nodes[element]
            Nv = len(element)
            area = self.calculate_element_area(element)
            h_K = self.calculate_element_diameter(node_coords)
            xc, yc = np.mean(node_coords, axis = 0)

            B = self.calculate_B_matrix(node_coords,h_K)
            D_vertics = self. calculate_D_matrix(node_coords,xc,yc,h_K)
            G = self.calculate_G_matrix(B, D_vertics)
            Pi_star = np.linalg.solve(G, B)

            #构建局部对角阵
            D_K_diag = np.ones(Nv) * (area / Nv)

            M_1 = np.zeros((Nv,3))
            M_1[:,1] = 1.0 /h_K

            M_2 = np.zeros((Nv,3))
            M_2 [:,2] = 1.0 /h_K

            uh_local = uh[element]
            fx_local = np.diag(D_K_diag) @ M_1 @ Pi_star @ uh_local
            fy_local = np.diag(D_K_diag) @ M_2 @ Pi_star @ uh_local

            #组装全局
            D_diag[element] += D_K_diag
            f_x[element] += fx_local
            f_y[element] += fy_local

        D_diag[D_diag < 1e-15] = 1e-15
        g_x = f_x / D_diag
        g_y = f_y / D_diag

        eta = np.zeros(self.n_elements)

        for element_id, element in enumerate(self.elements):
            node_coords = self.nodes[element]
            area = self.calculate_element_area(element)
            h_K = self.calculate_element_diameter(node_coords)

            uh_local = uh[element]
            grad_uh_x, grad_uh_y = self.evaluate_gradient_in_element(uh_local, node_coords, h_K)

            mean_gx = np.mean(g_x[element])
            mean_gy = np.mean(g_y[element])

            eta[element_id] = np.sqrt(area * ((grad_uh_x - mean_gx)**2 + (grad_uh_y - mean_gy)**2))

        return eta


def trapezoidal_birth(beta):
    """返回年龄方向复合梯形公式对应的出生边界函数。"""
    def birth_function(current_states, dt):
        """U_0^n = (beta*dt)/(1-beta*dt/2) * [sum_{i=1}^{N-1} U_i^n + U_N^n/2]。"""
        weighted_sum = np.sum(current_states[1:-1, :], axis=0) + 0.5 * current_states[-1, :]
        denominator = 1.0 - 0.5 * beta * dt
        return (beta * dt / denominator) * weighted_sum

    return birth_function
