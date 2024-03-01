import pandas as pd
import jieba
import matplotlib.pyplot as plt
import seaborn as sns
# 读取情感词典的 Excel 表格
emotion_lexicon_path = r"情感词汇本体\情感词汇本体.xlsx"
emotion_lexicon = pd.read_excel(emotion_lexicon_path)

# 构建情感词及其强度的字典
emotion_dict = dict(zip(emotion_lexicon['词语'], emotion_lexicon['强度']))

# 读取哈工大停用词表
stopwords_path = 'hit_stopwords.txt'
with open(stopwords_path, 'r', encoding='utf-8') as file:
    stopwords = set([line.strip() for line in file])

# 读取Excel文件
excel_file_path = 'wb_py2.xls'
data = pd.read_excel(excel_file_path)

# 计算每一行文字的情感分数和情感词数量
emotion_scores = []
word_counts = []

for index, row in data.iterrows():
    text = str(row[1])
    cut_text = [word for word in jieba.cut(text) if word not in stopwords]
    total_emotion_score = sum(emotion_dict.get(word, 0)+2 for word in cut_text)
    emotion_scores.append(total_emotion_score)
    word_counts.append(len(cut_text))

# 创建新的DataFrame存储情感分数和情感词数量
result_df = pd.DataFrame({'text': data.iloc[:, 1],  # Assuming the text is in the fourth column
                          'emotion_score': emotion_scores,
                          'word_count': word_counts})

# 计算每一行文字的平均情感分数（除以情感词数量）#考虑到不合适，因而没有使用平均值
result_df['avg_emotion_score'] = result_df['emotion_score']

# 打印每一行文字的平均情感分数
print(result_df[['text', 'avg_emotion_score']])

# 构建情感词及其强度的字典
emotion_dict = dict(zip(emotion_lexicon['词语'], emotion_lexicon['强度']))

file_path = 'all_data.txt'
data = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'text'], encoding='utf-8')

# 计算标签为0的每一行文字的情感分数和情感词数量
emotion_scores_label_0 = []
word_counts_label_0 = []

for index, row in data[data['label'] == 0].iterrows():
    text = row['text']
    cut_text = [word for word in jieba.cut(text) if word not in stopwords]
    total_emotion_score = sum(emotion_dict.get(word, 0) for word in cut_text)
    emotion_scores_label_0.append(total_emotion_score)
    word_counts_label_0.append(len(cut_text))

# 计算标签为1的每一行文字的情感分数和情感词数量
emotion_scores_label_1 = []
word_counts_label_1 = []

for index, row in data[data['label'] == 1].iterrows():
    text = row['text']
    cut_text = [word for word in jieba.cut(text) if word not in stopwords]
    total_emotion_score = sum(emotion_dict.get(word, 0) for word in cut_text)
    emotion_scores_label_1.append(total_emotion_score)
    word_counts_label_1.append(len(cut_text))

# 创建新的DataFrame存储标签为0的情感分数和情感词数量
result_df_label_0 = pd.DataFrame({'text': data[data['label'] == 0]['text'],
                                   'emotion_score': emotion_scores_label_0,
                                   'word_count': word_counts_label_0})

# 创建新的DataFrame存储标签为1的情感分数和情感词数量
result_df_label_1 = pd.DataFrame({'text': data[data['label'] == 1]['text'],
                                   'emotion_score': emotion_scores_label_1,
                                   'word_count': word_counts_label_1})

result_df_label_0['avg_emotion_score'] = result_df_label_0['emotion_score']
result_df_label_1['avg_emotion_score'] = result_df_label_1['emotion_score']


# 使用 seaborn 进行 KDE 可视化
sns.kdeplot(result_df_label_0['avg_emotion_score'], label='rumor', fill=True)
sns.kdeplot(result_df_label_1['avg_emotion_score'], label='non-rumor', fill=True)
# 使用 seaborn 进行 KDE 可视化
sns.kdeplot(result_df['avg_emotion_score'], label='spider Emotion Score', fill=True)
plt.xlabel(' Emotion Score')
plt.ylabel('Density')
plt.title('Distribution of  Emotion Scores')
plt.legend()
plt.show()
