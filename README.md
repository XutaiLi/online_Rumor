* text为研究文章
* Chinese_Rumor_Dataset-master文件夹内放有github获取的谣言数据

  数据集中共包含谣言1538条和非谣言1849条，以及其评论/转发约万余条。该数据集分为微博原文与其转发/评论内容。其中所有微博原文（包含谣言与非谣言）在original-microblog文件夹中，剩余两个文件夹non-rumor-repost和rumor-repost分别包含非谣言原文与谣言原文的对应的转发与评论信息。（该数据集中并不区分评论与转发）

* 情感词汇本体文件夹中放有大连理工大学情感强度词典

* all_data.txt是将谣言和非谣言的文本信息和标签读取处理生成的。

* hit_stopwords.txt是哈工大停用词表

* GCN.py为卷积训练分类的python代码

* IDA.py为主题分析代码

* spider.py为爬取微博文本的代码

* svm+rb+nb.py 为对比算法的代码，包括支持向量机，随机森林，朴素贝叶斯

* depth.py为求取图深度，各级节点数量的代码

* emotion.py为计算情感强度代码

* graph_visual.py为构建有向图，并实现可视化的代码

  

  

