import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import random
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics as metrics
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import pandas as pd

# 忽略 UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# 显示汉语
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取 JSON 文件
def read_json_file(file_path):
    #uft-8编码方式
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 构建谣言图结构
def build_graph(filename):
    graph = nx.Graph()
    # 读取谣言转发评论文件夹
    if filename.endswith('.json'):
        repost_data = read_json_file(os.path.join(rumor_path, filename))
        for repost in repost_data:
            post_id = repost.get('mid', None)
            original_post_id = repost.get('parent', None)
            if post_id is not None and original_post_id is not None:
                user_id = repost.get('uid', '')
                graph.add_node(post_id, type='repost', user=user_id, text=repost.get('text', ''))
                graph.add_edge(post_id, original_post_id)
    for edge in graph.edges:
        graph.edges[edge]['weight'] = 1
    for node in graph.nodes:
        graph.nodes[node]['label'] = 1

    return graph

# 构建非谣言图结构
def non_build_graph(filename):
    graph = nx.Graph()
    # 读取谣言转发评论文件夹
    if filename.endswith('.json'):
        repost_data = read_json_file(os.path.join(non_rumor_path, filename))
        for repost in repost_data:
            post_id = repost.get('mid', None)
            original_post_id = repost.get('parent', None)
            if post_id is not None and original_post_id is not None:
                user_id = repost.get('uid', '')
                graph.add_node(post_id, type='repost', user=user_id, text=repost.get('text', ''))
                graph.add_edge(post_id, original_post_id)
    for edge in graph.edges:
        graph.edges[edge]['weight'] = 1
    for node in graph.nodes:
        graph.nodes[node]['label'] =0
    return graph


# 生成稀疏邻接矩阵
def generate_sparse_adjacency_matrix(graph):
    adjacency_matrix = nx.adjacency_matrix(graph)
    return adjacency_matrix

#文本信息向量化
def create_data_from_graph(graph):
    # 将节点的 'text' 属性转换为词袋模型表示
    vectorizer = CountVectorizer()
    node_texts = [graph.nodes[node].get('text', '') for node in graph.nodes]
    node_features = torch.FloatTensor(vectorizer.fit_transform(node_texts).toarray())
    node_ids = list(graph.nodes)
    node_id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    edges = list(graph.edges)
    edge_index = torch.LongTensor(
        np.array([[node_id_to_index[edge[0]], node_id_to_index[edge[1]]] for edge in edges]).T)
    return node_features, edge_index

# 从图中创建边权重
def create_edge_weight_from_graph(graph):
    edge_weight = torch.FloatTensor(np.array([graph.edges[edge]['weight'] for edge in graph.edges]))
    return edge_weight

# 从图中创建标签
def create_target_from_graph(graph):
    target = torch.LongTensor([graph.nodes[node]['label'] for node in graph.nodes])
    return target

#定义GCN
class GCN(torch.nn.Module):
    def __init__(self, node_features, input_size,num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(node_features, input_size)
        self.linear = torch.nn.Linear(node_features + input_size, input_size)
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size // 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(input_size // 2, input_size // 4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(input_size // 4, num_classes))
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index, edge_weight):
        '''
        (x, GCN)
        '''
        lst = list()
        lst.append(x)

        x = self.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        lst.append(x)

        x = torch.cat(lst, dim=1)
        x = self.relu(self.linear(x))
        x = F.dropout(x, training=self.training)

        x = self.MLP(x)

        return x


# 数据集路径
dataset_path = r"Chinese_Rumor_Dataset-master\CED_Dataset"
rumor_path = os.path.join(dataset_path, 'rumor-repost')
non_rumor_path = os.path.join(dataset_path, 'non-rumor-repost')

# 训练集大小
train_size = 0.8
# 初始化模型和优化器
node_features = 26288
input_size = 205
num_classes = 2
epochs = 100
learning_rate = 0.01
weight_decay = 5e-4
# 定义损失函数
criterion = torch.nn.CrossEntropyLoss()

# 存储训练和测试损失
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
test_precisions = []
test_recalls = []
test_f1_scores = []

all_graphs = []

# 谣言图,电脑性能实在有限，使用了谣言非谣言共26288条
tag1=0
for filename in os.listdir(rumor_path):
    if filename.endswith('.json'):
        rumor_graph = build_graph(filename)
        all_graphs.append(rumor_graph)  # 使用标签 1 表示谣言图
        tag1+=1
    if tag1>=40:
        break
# 非谣言图
tag2=0
for filename in os.listdir(non_rumor_path):
    if filename.endswith('.json'):
        non_rumor_graph = non_build_graph(filename)
        all_graphs.append(non_rumor_graph)  # 使用标签 0 表示非谣言图
        tag2+=1
    if tag2>=40:
        break

# 将图随机打乱混合在一起
random.shuffle(all_graphs)
# 合并混合后的图
merged_graph = nx.compose_all([graph for graph in all_graphs])

# 生成稀疏邻接矩阵
sparse_adjacency_matrix = generate_sparse_adjacency_matrix(merged_graph)

# 数据集划分
dataset_size = len(merged_graph.nodes)
train_dataset_size = int(train_size * dataset_size)
test_dataset_size = dataset_size - train_dataset_size

train_dataset, test_dataset = random_split(range(dataset_size), [train_dataset_size, test_dataset_size])

# 划分后的数据集生成特征向量
x_train, edge_index_train = create_data_from_graph(merged_graph)
edge_weight_train = create_edge_weight_from_graph(merged_graph)
target_train = create_target_from_graph(merged_graph)

x_test, edge_index_test = create_data_from_graph(merged_graph)
edge_weight_test = create_edge_weight_from_graph(merged_graph)
target_test = create_target_from_graph(merged_graph)

# 初始化模型和优化器
model = GCN(node_features=node_features, input_size=input_size, num_classes=num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 模型训练
for epoch in tqdm(range(epochs)):
    optimizer.zero_grad()
    scores = model(x_train, edge_index_train, edge_weight_train)
    loss = criterion(scores, target_train)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    # 模型评估
    model.eval()

    with torch.no_grad():
        scores_test = model(x_test, edge_index_test, edge_weight_test)
        loss_test = criterion(scores_test, target_test)
        test_losses.append(loss_test.item())

        _, pred_test = scores_test.max(dim=1)

        # 计算准确率
        train_accuracy = metrics.accuracy_score(target_train, scores.argmax(dim=1))
        test_accuracy = metrics.accuracy_score(target_test, pred_test)

        # 计算精确率、召回率和 F1 值
        test_precision = metrics.precision_score(target_test, pred_test)
        test_recall = metrics.recall_score(target_test, pred_test)
        test_f1 = metrics.f1_score(target_test, pred_test)
        classes = ["Non-Rumor", "Rumor"]  # 非谣言为0，谣言为1
        confusion_matrix_result = metrics.confusion_matrix(target_test, pred_test)

        # 创建带标签的混淆矩阵
        confusion_matrix_with_labels = pd.DataFrame(confusion_matrix_result, index=classes, columns=classes)

        print(confusion_matrix_with_labels)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_f1_scores.append(test_f1)

# 绘制损失曲线
plt.plot(train_losses, label='训练损失')
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.legend()
plt.show()

# 输出分类评价指标
print(f"准确率: {train_accuracies[-1]}")
print(f"准确率: {test_accuracies[-1]}")
print(f"精确率: {test_precisions[-1]}")
print(f"召回率: {test_recalls[-1]}")
print("Test F1 Score:", test_f1_scores[-1])