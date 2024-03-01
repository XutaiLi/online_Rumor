import os
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt
import random

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def build_dataset(folder_path, label):
    dataset = []
    tag=0
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            data = read_json_file(os.path.join(folder_path, filename))
            for item in data:
                text = item.get('text', '')
                dataset.append({'text': text, 'label': label})
        tag+=1
        if tag>=50:
            break
    return dataset


dataset_path = r"C:\Users\李绪泰\OneDrive\桌面\Chinese_Rumor_Dataset-master\CED_Dataset"
rumor_path = os.path.join(dataset_path, 'rumor-repost')
non_rumor_path = os.path.join(dataset_path, 'non-rumor-repost')


rumor_dataset = build_dataset(rumor_path, label=1)
non_rumor_dataset = build_dataset(non_rumor_path, label=0)
print(rumor_dataset)


all_data = rumor_dataset + non_rumor_dataset
random.shuffle(all_data)


texts = [item['text'] for item in all_data]
labels = [item['label'] for item in all_data]
x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)


vectorizer = CountVectorizer()
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)


svm_model = SVC(kernel='linear')
svm_model.fit(x_train_vectorized, y_train)
svm_predictions = svm_model.predict(x_test_vectorized)
print('svm')


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train_vectorized, y_train)
rf_predictions = rf_model.predict(x_test_vectorized)
print('rf')


nb_model = MultinomialNB()
nb_model.fit(x_train_vectorized, y_train)
nb_predictions = nb_model.predict(x_test_vectorized)
print('nb')


print("SVM Metrics:")
print(metrics.classification_report(y_test, svm_predictions))
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, svm_predictions))


print("Random Forest Metrics:")
print(metrics.classification_report(y_test, rf_predictions))
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, rf_predictions))


print("Naive Bayes Metrics:")
print(metrics.classification_report(y_test, nb_predictions))
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, nb_predictions))
