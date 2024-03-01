import pandas as pd
import jieba
import re
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim import corpora, models
import warnings
import time

warnings.filterwarnings("ignore", category=DeprecationWarning)

def preprocess_text(text):
    # Remove English words
    text = re.sub(r'[a-zA-Z]', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

# 读取哈工大停用词表
stopwords_path = 'hit_stopwords.txt'
with open(stopwords_path, 'r', encoding='utf-8') as file:
    stopwords = set([line.strip() for line in file])

file_path = 'all_data.txt'
data = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'text'], encoding='utf-8')

# 对文本进行预处理、分词，并去除停用词
data['tokenized_text'] = data['text'].apply(lambda x: [word for word in jieba.cut(preprocess_text(x)) if word not in stopwords])

# 构建语料库
dictionary = corpora.Dictionary(data['tokenized_text'])
corpus = [dictionary.doc2bow(text) for text in data['tokenized_text']]

# 使用LDA模型训练主题
lda_model = models.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=15)

# 创建主题可视化
lda_vis_data = gensimvis.prepare(lda_model, corpus, dictionary)

# 保存可视化到HTML文件
pyLDAvis.save_html(lda_vis_data, 'lda_visualization2.html')

