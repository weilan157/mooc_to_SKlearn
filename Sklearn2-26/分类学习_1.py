# coding = utf-8
import numpy as np
from sklearn.cluster import KMeans


def loadData(filePath):
    """
    导入数据
    :param filePath(文件路径)
    :return retData(城市消费水平数据)
    :return retCityName(城市名称)
    """
    fr = open(filePath, "r+")
    lines = fr.readlines()
    retData = []
    retCityName = []
    for line in lines:
        items = line.split(",")
        retCityName.append(items[0])
        retData.append([float(items[item]) for item in range(1, len(items))])
    return retData, retCityName


if __name__ == "__main__":
    data, cityName = loadData("city.txt")  # 导入31省消费水平数据
    km = KMeans(n_clusters=3)  # 初始化KMeans分成3类
    label = km.fit_predict(data)  # 分类数据
    expenses = np.sum(km.cluster_centers_, axis=1)
    # print(expenses)
    CityCluster = [[], [], []]
    for i in range(len(cityName)):
        CityCluster[label[i]].append(cityName[i])
    for i in range(len(CityCluster)):
        print(f'Expenses:{expenses[i]}')
        print(CityCluster[i])
