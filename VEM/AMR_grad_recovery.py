import numpy as np

def get_edge_key(n1, n2):
    return (min(n1,n2),max(n1,n2))


def polygon_signed_area(coords):
    """计算多边形有符号面积；正值表示顶点按逆时针排列。"""
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def orient_element_ccw(element, nodes):
    """保证单元顶点按逆时针顺序存储，便于 VEM 局部矩阵保持一致。"""
    oriented = [int(v) for v in element]
    if polygon_signed_area(nodes[oriented]) < 0.0:
        oriented.reverse()
    return oriented


def validate_polygon_mesh(nodes, elements, area_tol=1e-14):
    """检查 AMR 前后的多边形网格是否满足当前 VEM 装配的基本要求。"""
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("nodes must be an array with shape (n_nodes, 2).")

    edge_counts = {}
    for elem_idx, element in enumerate(elements):
        elem = np.asarray(element, dtype=int)
        if elem.ndim != 1 or len(elem) < 3:
            raise ValueError(f"Element {elem_idx} must contain at least 3 vertices.")
        if np.any(elem < 0) or np.any(elem >= len(nodes)):
            raise ValueError(f"Element {elem_idx} contains an invalid node index.")
        if len(np.unique(elem)) != len(elem):
            raise ValueError(f"Element {elem_idx} contains duplicate vertices.")

        area = abs(polygon_signed_area(nodes[elem]))
        if area <= area_tol:
            raise ValueError(f"Element {elem_idx} is degenerate or has near-zero area.")

        for i in range(len(elem)):
            edge = get_edge_key(int(elem[i]), int(elem[(i + 1) % len(elem)]))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    nonmanifold_edges = [edge for edge, count in edge_counts.items() if count > 2]
    if nonmanifold_edges:
        raise ValueError(f"Mesh contains non-manifold edges, e.g. {nonmanifold_edges[0]}.")


def refine_mesh(nodes, elements, marked_elements):
    """
    形心-边中点 切割法
    将被标记的 N 边形单元切割为 N 个四边形小单元。
    未标记邻居会把共享边中点插入自己的顶点序列，从而避免悬挂节点。
    """
    nodes = np.asarray(nodes, dtype=float)
    validate_polygon_mesh(nodes, elements)

    # 初始化新节点列表（先包含所有旧节点）
    new_nodes = nodes.tolist()
    edge_to_mid = {}
    marked_set = {int(elem_idx) for elem_idx in marked_elements}
    invalid_marked = [idx for idx in marked_set if idx < 0 or idx >= len(elements)]
    if invalid_marked:
        raise ValueError(f"marked_elements contains invalid element indices: {invalid_marked}")

    for elem_idx in marked_set:
        elem = elements[elem_idx]
        n_vertices = len(elem)

        for i in range(n_vertices):
            n1 = elem[i]
            n2 = elem[(i+1) % n_vertices]
            edge_key = get_edge_key(n1,n2)

            if edge_key not in edge_to_mid:
                mid_coord = (nodes[n1] + nodes[n2]) / 2

                mid_idx = len(new_nodes)
                new_nodes.append(mid_coord)
                edge_to_mid[edge_key] = mid_idx

    new_elements = []
    centroid_to_id = {}

    for elem_idx, elem in enumerate(elements):
            #情况1:标记出来需要切割的单元
        if elem_idx in marked_set:
            n_vertices = len(elem)

            centroid_coord = np.mean([nodes[v] for v in elem], axis=0)
            centroid_idx = len(new_nodes)
            new_nodes.append(centroid_coord)
            centroid_to_id[elem_idx] = centroid_idx

            # 2、获取所有边中点索引
            mid_indices = []
            for i in range(n_vertices):
                n1 = elem[i]
                n2 = elem[(i+1) % n_vertices]
                edge_key = get_edge_key(n1, n2)
                mid_indices.append(edge_to_mid[edge_key])
            
            # 3. 生成N个四边形子单元（逆时针）
            for i in range(n_vertices):
                prev_mid_idx = mid_indices[(i-1) % n_vertices]
                curr_mid_idx = mid_indices[i]
                curr_vertex = elem[i]
                
                sub_elem = [
                    curr_vertex,
                    curr_mid_idx,
                    centroid_idx,
                    prev_mid_idx
                ]
                new_elements.append(sub_elem)
        
        else:
            
            # 情况2：未被标记的单元 → 插入邻接细化边的中点，消除悬挂节点。
            
            new_elem = []
            n_vertices = len(elem)
            
            for i in range(n_vertices):
                n1 = elem[i]
                n2 = elem[(i+1) % n_vertices]
                edge_key = get_edge_key(n1, n2)
                
                # 先加原来顶点
                new_elem.append(n1)
                
                # 如果有中点就加入中点
                if edge_key in edge_to_mid:
                    new_elem.append(edge_to_mid[edge_key])
            
            new_elements.append(new_elem)

    new_nodes = np.array(new_nodes, dtype=float)
    # 统一顶点方向，并在返回前检查退化单元、重复顶点和非流形边。
    new_elements = [orient_element_ccw(elem, new_nodes) for elem in new_elements]
    validate_polygon_mesh(new_nodes, new_elements)

    return new_nodes, new_elements, edge_to_mid, centroid_to_id


def interpolate_states(old_states, elements, old_nodes_count, new_nodes_count,
                       edge_to_mid, centroid_to_id):
    """
    网格细化后，将旧网格上的数值解线性插值到新网格的节点上。

    参数:
    old_states: 旧的全局状态矩阵，维度 (n_age, old_nodes_count)
    elements: 原始网格单元数组（用于获取形心对应的顶点）
    old_nodes_count: 细化前的总节点数
    new_nodes_count: 细化后的总节点数
    edge_to_mid: 边到中点索引的映射字典
    centroid_to_id: 元素索引到形心节点索引的映射字典

    返回:
    new_states: 映射到新网格上的全局状态矩阵，维度 (n_age, new_nodes_count)
    """
    n_age = old_states.shape[0]
    new_states = np.zeros((n_age, new_nodes_count))

    # 旧节点上的解保持不变
    new_states[:, :old_nodes_count] = old_states

    # 边中点：取两端点解的算术平均值
    for (n1, n2), mid_id in edge_to_mid.items():
        new_states[:, mid_id] = 0.5 * (old_states[:, n1] + old_states[:, n2])

    # 形心：取该单元所有顶点解的算术平均值
    for elem_idx, c_id in centroid_to_id.items():
        verts = elements[elem_idx]
        new_states[:, c_id] = np.mean(old_states[:, verts], axis=1)

    return new_states
