import os
import json
import networkx as nx
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 读取 JSON 文件
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


# 谣言图可视化
def build_graphs(dataset_path):
    graphs = []

    # 读取谣言转发评论文件夹
    rumor_path = os.path.join(dataset_path, 'rumor-repost')
    for filename in os.listdir(rumor_path):
        if filename.endswith('.json'):
            graph = nx.Graph()
            repost_data = read_json_file(os.path.join(rumor_path, filename))
            for repost in repost_data:
                post_id = repost.get('mid', None)
                original_post_id = repost.get('parent', None)
                if post_id is not None and original_post_id is not None:
                    user_id = repost.get('uid', '')
                    graph.add_node(post_id, type='repost', user=user_id, text=repost.get('text', ''))
                    graph.add_edge(post_id, original_post_id)
            graphs.append(graph)

    return graphs


# 计算平均距离、最大距离和平均节点数量
def calculate_graph_metrics(graph):
    average_distance = nx.average_shortest_path_length(graph)
    eccentricities = nx.eccentricity(graph)
    max_distance = max(eccentricities.values())
    average_nodes = graph.number_of_nodes()

    return average_distance, max_distance, average_nodes


# 统计一级、二级、三级节点数量
def count_nodes_at_levels(graph):
    level_counts = {1: 0, 2: 0, 3: 0}

    for node in graph.nodes:
        ego_graph = nx.ego_graph(graph, node, radius=1)
        level_counts[1] += ego_graph.number_of_nodes() - 1

        for neighbor in ego_graph.nodes:
            if neighbor != node:
                ego_graph_level_2 = nx.ego_graph(graph, neighbor, radius=1)
                level_counts[2] += ego_graph_level_2.number_of_nodes() - 1

                for neighbor_level_2 in ego_graph_level_2.nodes:
                    if neighbor_level_2 != node and neighbor_level_2 != neighbor:
                        ego_graph_level_3 = nx.ego_graph(graph, neighbor_level_2, radius=1)
                        level_counts[3] += ego_graph_level_3.number_of_nodes() - 1

    return level_counts



dataset_path = r"Chinese_Rumor_Dataset-master\CED_Dataset"

# 构建图结构
graphs = build_graphs(dataset_path)

# 计算指标的累加值
total_avg_distance = 0
total_max_distance = 0
total_avg_nodes = 0

# 统计节点级别数量的累加值
total_level_counts = {1: 0, 2: 0, 3: 0}

# 逐个可视化图，并累加指标和节点级别数量
for i, graph in enumerate(graphs):
    print(f"Visualization of Graph {i + 1}")

    # 计算图指标
    avg_distance, max_distance, avg_nodes = calculate_graph_metrics(graph)

    # 累加值
    total_avg_distance += avg_distance
    total_max_distance += max_distance
    total_avg_nodes += avg_nodes

    # 统计节点级别数量
    level_counts = count_nodes_at_levels(graph)
    for level in level_counts:
        total_level_counts[level] += level_counts[level]

# 计算平均指标值
overall_avg_distance = total_avg_distance / len(graphs)
overall_max_distance = total_max_distance / len(graphs)
overall_avg_nodes = total_avg_nodes / len(graphs)

# 打印结果
print("Overall Metrics:")
print(f"Overall Average Distance: {overall_avg_distance}")
print(f"Overall Maximum Distance: {overall_max_distance}")
print(f"Overall Average Number of Nodes: {overall_avg_nodes}")
print("\nNode Counts at Different Levels:")
for level in total_level_counts:
    print(f"Level {level} Nodes: {total_level_counts[level]}")
