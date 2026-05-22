import numpy as np

# 计算基于梯度恢复的VEM后验误差指示器

def estimate_gradient_recovery(uh, mesh_data):
    n_dofs = len(uh)

    D_diag = np.zeros(n_dofs)
    f_x = np.zeros(n_dofs)
    f_y = np.zeros(n_dofs)

    for elem_id in range(mesh_data['n_elements']):
        dof_indices = mesh_data['element_dof_indices'][elem_id]
        Nv = len(dof_indices)

        area = mesh_data['areas'][elem_id]
        h_K = mesh_data['diameter'][elem_id]

        D_K_diag = np.ones(Nv) * (area / Nv)
        D_K = np.diag(D_K_diag)

        M_1 = np.zeros((Nv, 3))
        M_1[:, 1] = 1.0 / h_K

        M_2 = np.zeros((Nv, 3))
        M_2[:, 2] = 1.0 / h_K

        S = mesh_data['S_matrices'][elem_id]

        Q_x_local = D_K @ M_1 @ S
        Q_y_local = D_K @ M_2 @ S

        uh_local = uh[dof_indices]

        D_diag[dof_indices] += D_K_diag
        f_x[dof_indices] += Q_x_local @ uh_local
        f_y[dof_indices] += Q_y_local @ uh_local

    g_x = f_x / D_diag
    g_y = f_y / D_diag


    eta = np.zeros(mesh_data['n_elements'])
    for elem_id in range(mesh_data['n_elements']):
        dof_indices = mesh_data['elements'][elem_id]
        area = mesh_data['areas'][elem_id]
        S = mesh_data['S_matrices'][elem_id]
        uh_local = uh[dof_indices]

        coef = S @ uh_local
        grad_uh_x = coef[1] / mesh_data['diameter'][elem_id]
        grad_uh_y = coef[2] / mesh_data['diameter'][elem_id]

        mean_gx = np.mean(g_x[dof_indices])
        mean_gy = np.mean(g_y[dof_indices])

        eta[elem_id] = np.sqrt(area * ((grad_uh_x - mean_gx) ** 2 + (grad_uh_y - mean_gy) ** 2))

    return eta
