import numpy as np

def get_midpoint(n1, n2, old_nodes, new_nodes_list,edge_to_node):
    """
    获取两个节点之间的中点索引。
    如果这条边的中点之前已经生成过，则直接返回已有索引，避免重复生成节点。
    
    参数:
    n1, n2: 边的两个端点全局索引
    old_nodes: 原始的节点坐标数组
    new_nodes_list: 正在不断追加的新节点列表
    edge_to_node: 字典，用于记录 (n1, n2) -> new_node_id 的映射
    
    返回:
    new_node_id: 中点在全局节点列表中的索引
    """
    # 保证边的键值对方向无关 (从小到大排序)
    edge = tuple(sorted((n1, n2)))
    
    if edge in edge_to_node:
        return edge_to_node[edge]
    
    # 如果是全新的边，计算中点坐标
    coord1 = old_nodes[n1]
    coord2 = old_nodes[n2]
    mid_coord = (coord1 + coord2) / 2.0
    
    # 记录新节点
    new_node_id = len(new_nodes_list)
    new_nodes_list.append(mid_coord)
    edge_to_node[edge] = new_node_id
    
    return new_node_id

def refine_mesh(nodes, elements, marked_elements):
    """
    执行基于标记的局部网格细化 (1-to-4 Red Refinement)。
    将所有被标记的三角形单元剖分为 4 个子三角形。
    
    参数:
    nodes: 当前网格节点数组 (N_nodes, 2)
    elements: 当前网格单元数组 (N_elements, 3)
    marked_elements: 需要被细化的单元 ID 列表
    
    返回:
    new_nodes_array: 更新后的网格节点数组
    new_elements_array: 更新后的网格单元数组
    edge_to_node: 边到中点索引的映射字典 (供后续解插值使用)
    """
    # 初始化新节点列表（先包含所有旧节点）
    new_nodes_list = list(nodes)
    new_elements_list = []
    edge_to_node = {}
    
    marked_set = set(marked_elements)
    
    for i, element in enumerate(elements):
        if i in marked_set:
            n1, n2, n3 = element
            
            # 获取三条边的中点索引
            m12 = get_midpoint(n1, n2, nodes, new_nodes_list, edge_to_node)
            m23 = get_midpoint(n2, n3, nodes, new_nodes_list, edge_to_node)
            m31 = get_midpoint(n3, n1, nodes, new_nodes_list, edge_to_node)
            
            # 将原三角形替换为 4 个小的子三角形
            new_elements_list.append([n1, m12, m31])
            new_elements_list.append([n2, m23, m12])
            new_elements_list.append([n3, m31, m23])
            new_elements_list.append([m12, m23, m31])
        else:
            # 未被标记的单元保持原样
            new_elements_list.append(element.tolist())
            
    return np.array(new_nodes_list), np.array(new_elements_list), edge_to_node

def interpolate_states(old_states, old_nodes_count, new_nodes_count, edge_to_node):
    """
    网格细化后，将旧网格上的数值解（人口状态）线性插值到新网格的节点上。
    由于人口模型包含多个年龄层，所以处理的是二维数组。
    
    参数:
    old_states: 旧的全局状态矩阵，维度 (n_age, old_nodes_count)
    old_nodes_count: 细化前的总节点数
    new_nodes_count: 细化后的总节点数
    edge_to_node: refine_mesh 返回的边到中点索引的映射字典
    
    返回:
    new_states: 映射到新网格上的全局状态矩阵，维度 (n_age, new_nodes_count)
    """
    n_age = old_states.shape[0]
    new_states = np.zeros((n_age, new_nodes_count))
    
    # 1. 旧节点上的解保持不变
    new_states[:, :old_nodes_count] = old_states
    
    # 2. 新生成的边中点上的解，取两端点解的算术平均值 (P1 线性插值)
    for (n1, n2), mid_id in edge_to_node.items():
        new_states[:, mid_id] = 0.5 * (old_states[:, n1] + old_states[:, n2])
        
    return new_states