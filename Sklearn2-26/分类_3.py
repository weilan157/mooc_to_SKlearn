# coding=utf-8
"""
使用knn/决策树/贝叶斯，分类器对人体姿态传感器数据分类
"""
import numpy as np
import pandas as pd

from sklearn.preprocessing import Imputer  # 预处理
from sklearn.model_selection import train_test_split  # 自动生成训练集和测试集
from sklearn.metrics import classification_report  # 预测结果评估

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


def load_dataset(feature_paths, label_paths):
    """
    导入数据
    :param feature_paths: 特征文件
    :param label_paths: 标签文件
    :return: feature: 特征集合；label: 标签集合
    """
    feature = np.ndarray(shape=(0, 41))
    label = np.ndarray(shape=(0, 1))
    for file in feature_paths:
        df = pd.read_table(file, delimiter=',', na_values="?", header=None)
        imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
        imp.fit(df)
        df = imp.transform(df)
        feature = np.concatenate((feature, df))
    for file in label_paths:
        df = pd.read_table(file, header=None)
        label = np.concatenate((label, df))
    label = np.ravel(label)
    return feature, label


if __name__ == "__main__":
    # 设置数据路径
    feature_paths = ['dataset/A/A.feature', 'dataset/B/B.feature',
                     'dataset/C/C.feature', 'dataset/D/D.feature', 'dataset/E/E.feature', ]
    label_paths = ['dataset/A/A.label', 'dataset/B/B.label',
                   'dataset/C/C.label', 'dataset/D/D.label', 'dataset/E/E.label', ]
    # 训练集
    x_train, y_train = load_dataset(feature_paths[:4], label_paths[:4])
    # 测试集
    x_test, y_test = load_dataset(feature_paths[4:], label_paths[4:])
    # 打乱训练集
    x_train, x_, y_train, y_ = train_test_split(x_train, y_train, test_size=0.0)

    # 创建knn（k近邻）分类器，并预测#########################
    print('Start training knn')
    knn = KNeighborsClassifier().fit(x_train, y_train)
    print('Training dong!')
    answer_knn = knn.predict(x_test)
    print('Prediction done!')
    # 创建决策树分类器，并预测###############################
    print('Start training DT')
    dt = DecisionTreeClassifier().fit(x_train, y_train)
    print('Training dong!')
    answer_dt = dt.predict(x_test)
    print('Prediction done!')
    # 创建贝叶斯分类器，并预测###############################
    print('Start training Bayes')
    gnb = GaussianNB().fit(x_train, y_train)
    print('Training dong!')
    answer_gnb = gnb.predict(x_test)
    print('Prediction done!')

    # 计算准确率和召回率
    print("The classification report for knn:")
    print(classification_report(y_test, answer_knn))

    print("The classification report for dt:")
    print(classification_report(y_test, answer_dt))

    print("The classification report for gnb:")
    print(classification_report(y_test, answer_gnb))
