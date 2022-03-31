import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Node(object):
    def __init__(self):
        self.feature_index = None
        self.split_point = None
        self.deep = None
        self.left_tree = None
        self.right_tree = None
        self.leaf_class = None

'''
函数说明：计算样本集y下的加权基尼指数

Parameters:
    y - 数据样本标签
    D - 样本权重
    
Return:  
    gini - 加权后的基尼指数
'''
def gini(y, D):
    unique_class = np.unique(y)
    total_weight = np.sum(D)

    gini = 1
    for c in unique_class:
        gini -= (np.sum(D[y == c]) / total_weight) ** 2

    return gini

'''
函数说明：计算特征a下样本集y的的基尼指数

Parameters:
    a - 单一特征值
    y - 数据样本标签
    D - 样本权重

Return:
    无
'''
def calcMinGiniIndex(a, y, D):
    feature = np.sort(a)
    total_weight = np.sum(D)

    split_points = [(feature[i] + feature[i + 1]) / 2 for i in range(feature.shape[0] - 1)]

    min_gini = float('inf')
    min_gini_point = None

    for i in split_points:
        yv1 = y[a <= i]
        yv2 = y[a > i]

        Dv1 = D[a <= i]
        Dv2 = D[a > i]
        gini_tmp = (np.sum(Dv1) * gini(yv1, Dv1) + np.sum(Dv2) * gini(yv2, Dv2)) / total_weight

        if gini_tmp < min_gini:
            min_gini = gini_tmp
            min_gini_point = i

    return min_gini, min_gini_point

'''
函数
    param X:
    param y:
    param D:
Returns: 
    特征索引, 分割点
'''
def chooseFeatureToSplit(X, y, D):
    gini0, split_point0 = calcMinGiniIndex(X[:, 0], y, D)
    gini1, split_point1 = calcMinGiniIndex(X[:, 1], y, D)

    if gini0 > gini1:
        return 1, split_point1
    else:
        return 0, split_point0


def createSingleTree(X, y, D, limit_Deep,deep=0):
    '''
    :param X: 训练集特征
    :param y: 训练集标签
    :param D: 训练样本权重
    :param deep: 树的深度
    :return:
    '''

    node = Node()
    node.deep = deep

    # 当前分支下，样本数量小于等于2 或深度达到limit_Deep，直接设置为叶节点
    if (deep == limit_Deep) | (X.shape[0] <= 2):
        pos_weight = np.sum(D[y == 1])
        neg_weight = np.sum(D[y == -1])
        if pos_weight > neg_weight:
            node.leaf_class = 1
        else:
            node.leaf_class = -1

        return node
    # 如果样本全部属于同一种类，直接设置为叶节点
    if(np.sum(D[y == 1]) == 0 | np.sum(D[y == -1]) == 0):
        node.leaf_class = 1 if(np.sum(D[y == 1]) == 0) else 0

    feature_index, split_point = chooseFeatureToSplit(X, y, D)

    node.feature_index = feature_index
    node.split_point = split_point

    left = X[:, feature_index] <= split_point
    right = X[:, feature_index] > split_point

    node.left_tree = createSingleTree(X[left, :], y[left], D[left],limit_Deep, deep + 1)
    node.right_tree = createSingleTree(X[right, :], y[right], D[right],limit_Deep, deep + 1)

    return node


def predictSingle(tree, x):
    '''
    基于基学习器，预测单个样本
    :param tree:
    :param x:
    :return:
    '''
    if tree.leaf_class is not None:
        return tree.leaf_class

    if x[tree.feature_index] > tree.split_point:
        return predictSingle(tree.right_tree, x)
    else:
        return predictSingle(tree.left_tree, x)


def predictBase(tree, X):
    '''
    基于基学习器预测所有样本
    :param tree:
    :param X:
    :return:
    '''
    result = []

    for i in range(X.shape[0]):
        result.append(predictSingle(tree, X[i, :]))

    return np.array(result)


def adaBoostTrain(X, y, tree_num ,limit_Deep):
    '''
    以深度为limit_Deep的决策树作为基学习器，训练adaBoost
    :param X:
    :param y:
    :param tree_num:
    :return:
    '''
    D = np.ones(y.shape) / y.shape  # 初始化权重

    trees = []  # 所有基学习器
    a = []  # 基学习器对应权重

    agg_est = np.zeros(y.shape)

    for _ in range(tree_num):
        tree = createSingleTree(X, y, D , limit_Deep)

        hx = predictBase(tree, X)
        err_rate = np.sum(D[hx != y])

        at = np.log((1 - err_rate) / max(err_rate, 1e-16)) / 2

        agg_est += at * hx
        trees.append(tree)
        a.append(at)

        if (err_rate > 0.5) | (err_rate == 0):  # 错误率大于0.5 或者 错误率为0时，则直接停止
            break

        # 更新每个样本权重
        err_index = np.ones(y.shape)
        err_index[hx == y] = -1

        D = D * np.exp(err_index * at)
        D = D / np.sum(D)

    return trees, a, agg_est


def adaBoostPredict(X, trees, a):
    agg_est = np.zeros((X.shape[0],))

    for tree, am in zip(trees, a):
        agg_est += am * predictBase(tree, X)

    result = np.ones((X.shape[0],))

    result[agg_est < 0] = -1

    return result.astype(int)


def pltAdaBoostDecisionBound(X_, y_, trees, a):
    pos = y_ == 1
    neg = y_ == -1
    x_tmp = np.linspace(0, 1, 600)
    y_tmp = np.linspace(-0.2, 0.7, 600)

    X_tmp, Y_tmp = np.meshgrid(x_tmp, y_tmp)

    Z_ = adaBoostPredict(np.c_[X_tmp.ravel(), Y_tmp.ravel()], trees, a).reshape(X_tmp.shape)
    plt.contour(X_tmp, Y_tmp, Z_, [0], colors='orange', linewidths=1)

    plt.scatter(X_[pos, 0], X_[pos, 1], label='1', color='c')
    plt.scatter(X_[neg, 0], X_[neg, 1], label='0', color='lightcoral')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data_path = r'E:\Code\Machine-Learning\Adaboost\melon.txt'

    data = pd.read_table(data_path, delimiter=' ')

    X = data.iloc[:, :2].values
    y = data.iloc[:, 2].values

    y[y == 0] = -1

    trees, a, agg_est = adaBoostTrain(X, y ,1 ,3)

    print(trees[0].deep)

    predictions = adaBoostPredict(X,trees,a)

    errArr = np.ones(len(predictions))


    print('训练集的错误率:%.3f%%' % float(errArr[predictions != y].sum()/len(errArr)  * 100))

    pltAdaBoostDecisionBound(X, y, trees, a)