import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


class ASP_FEM2D:
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
        """初始化二维 P1 三角形 FEM 框架，预装配 M, S。"""
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
        """检查步长并生成规则矩形区域上的三角形网格。"""
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
        """打印当前二维网格的基本规模信息。"""
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
        print("  element type: P1 triangle")

    def get_quadrature(self):
        """固定使用三角形 3 点高斯积分。"""
        return [
            (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0),
            (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
            (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
        ]

    def p1_shape_functions(self, xi, eta):
        """参考三角形上的 P1 形函数: [1-xi-eta, xi, eta]。"""
        return np.array([1.0 - xi - eta, xi, eta], dtype=float)

    def p1_shape_derivatives(self):
        """参考三角形上 P1 形函数对 (xi, eta) 的导数。"""
        dphi_dxi = np.array([-1.0, 1.0, 0.0], dtype=float)
        dphi_deta = np.array([-1.0, 0.0, 1.0], dtype=float)
        return dphi_dxi, dphi_deta

    def calculate_physical_derivatives(self, xi, eta, node_coords):
        """把参考导数映射到物理单元，返回 dphi/dx, dphi/dy 和 |detJ|。"""
        del xi, eta

        dphi_dxi, dphi_deta = self.p1_shape_derivatives()

        dx_dxi = np.dot(dphi_dxi, node_coords[:, 0])
        dy_dxi = np.dot(dphi_dxi, node_coords[:, 1])
        dx_deta = np.dot(dphi_deta, node_coords[:, 0])
        dy_deta = np.dot(dphi_deta, node_coords[:, 1])

        det_jacobian = dx_dxi * dy_deta - dx_deta * dy_dxi
        if abs(det_jacobian) <= np.finfo(float).eps:
            raise ValueError("Degenerate triangular element detected.")

        inverse_det = 1.0 / det_jacobian
        abs_det_jacobian = abs(det_jacobian)

        dphi_dx = inverse_det * (dy_deta * dphi_dxi - dy_dxi * dphi_deta)
        dphi_dy = inverse_det * (-dx_deta * dphi_dxi + dx_dxi * dphi_deta)
        return dphi_dx, dphi_dy, abs_det_jacobian

    def calculate_element_area(self, element_nodes):
        """计算三角形单元面积。"""
        node_coords = self.nodes[element_nodes]
        x1, y1 = node_coords[0]
        x2, y2 = node_coords[1]
        x3, y3 = node_coords[2]

        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        if area <= np.finfo(float).eps:
            raise ValueError("Degenerate triangular element detected.")
        return area

    def calculate_element_mass(self, element_nodes):
        """P1 单元质量矩阵: M_e = area / 12 * [[2,1,1],[1,2,1],[1,1,2]]。"""
        area = self.calculate_element_area(element_nodes)
        return (area / 12.0) * np.array(
            [
                [2.0, 1.0, 1.0],
                [1.0, 2.0, 1.0],
                [1.0, 1.0, 2.0],
            ],
            dtype=float,
        )

    def calculate_element_stiffness(self, element_nodes):
        """P1 单元刚度矩阵: S_e = ∫_K grad(phi_i)·grad(phi_j) dx。"""
        node_coords = self.nodes[element_nodes]
        dphi_dx, dphi_dy, abs_det_jacobian = self.calculate_physical_derivatives(0.0, 0.0, node_coords)
        area = 0.5 * abs_det_jacobian
        return area * (np.outer(dphi_dx, dphi_dx) + np.outer(dphi_dy, dphi_dy))

    def calculate_element_reaction(self, element_nodes, age_midpoint, mu_function):
        """单元反应矩阵: C_e = mu(a_{i-1/2}) M_e。"""
        mu_value = float(mu_function(age_midpoint))
        return mu_value * self.calculate_element_mass(element_nodes)

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
        """当前年龄层对应的全局反应矩阵 C_i。"""
        reaction_matrix = lil_matrix((self.n_nodes, self.n_nodes))

        for element in self.elements:
            element_matrix = self.calculate_element_reaction(element, age_midpoint, mu_function)
            for i, node_i in enumerate(element):
                for j, node_j in enumerate(element):
                    reaction_matrix[node_i, node_j] += element_matrix[i, j]

        return reaction_matrix.tocsr()

    def domain_measure_by_quadrature(self):
        """ 3 点高斯积分计算区域面积 。"""
        total_measure = 0.0
        quadrature = self.get_quadrature()

        for element in self.elements:
            node_coords = self.nodes[element]
            _, _, abs_det_jacobian = self.calculate_physical_derivatives(0.0, 0.0, node_coords)
            for _, _, weight in quadrature:
                total_measure += abs_det_jacobian * weight

        return float(total_measure)

    def build_system_matrices(self, dt, age_midpoint, mu_function):
        """单层推进矩阵: lhs = M/dt + (K_i+C_i)/2, rhs = M/dt - (K_i+C_i)/2。"""
        reaction_matrix = self.build_reaction_matrix(age_midpoint, mu_function)
        diffusion_matrix = self.kappa * self.stiffness_matrix

        lhs = (1.0 / dt) * self.mass_matrix + 0.5 * diffusion_matrix + 0.5 * reaction_matrix
        rhs = (1.0 / dt) * self.mass_matrix - 0.5 * diffusion_matrix - 0.5 * reaction_matrix
        return lhs.tocsr(), rhs.tocsr()

    def advance_age_layer(self, previous_state, dt, age_index, mu_function):
        """推进一个年龄层: 由 U_{i-1}^{n-1} 解出 U_i^n。"""
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
        """重复调用单步推进，返回最后全部年龄层状态。"""
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


def trapezoidal_birth(beta):
    """返回年龄方向复合梯形公式对应的出生边界函数。"""
    def birth_function(current_states, dt):
        """U_0^n = (beta*dt)/(1-beta*dt/2) * [sum_{i=1}^{N-1} U_i^n + U_N^n/2]。"""
        weighted_sum = np.sum(current_states[1:-1, :], axis=0) + 0.5 * current_states[-1, :]
        denominator = 1.0 - 0.5 * beta * dt
        return (beta * dt / denominator) * weighted_sum

    return birth_function
