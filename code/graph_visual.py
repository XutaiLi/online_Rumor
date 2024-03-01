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
def build_digraphs(dataset_path):
    digraphs = []

    # 读取谣言转发评论文件夹
    rumor_path = os.path.join(dataset_path, 'rumor-repost')
    for filename in os.listdir(rumor_path):
        if filename.endswith('.json'):
            digraph = nx.DiGraph()  # 使用 DiGraph 创建有向图
            repost_data = read_json_file(os.path.join(rumor_path, filename))
            for repost in repost_data:
                post_id = repost.get('mid', None)
                original_post_id = repost.get('parent', None)
                if post_id is not None and original_post_id is not None:
                    user_id = repost.get('uid', '')
                    digraph.add_node(post_id, type='repost', user=user_id, text=repost.get('text', ''))
                    digraph.add_edge(original_post_id, post_id)  # 修改这一行，将边设为有向边
            digraphs.append(digraph)

    return digraphs

# 可视化有向图
def visualize_graph(graph):
    pos = nx.spring_layout(graph)  # 定义节点位置布局
    nx.draw(graph, pos, with_labels=True, font_size=8, font_color='black', node_size=50, node_color='skyblue', edge_color='gray', linewidths=0.5)
    plt.show()

# 示例数据集路径
dataset_path = r"C:\Users\李绪泰\OneDrive\桌面\Chinese_Rumor_Dataset-master\CED_Dataset"

# 构建有向图结构
digraphs = build_digraphs(dataset_path)

# 逐个可视化有向图
for i, digraph in enumerate(digraphs):
    print(f"Visualization of Digraph {i + 1}")
    visualize_graph(digraph)
