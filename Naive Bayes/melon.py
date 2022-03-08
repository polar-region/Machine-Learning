import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target

"""
函数说明:处理训练集得到条件概率

Parameters:
	X - 训练集的数据
    y - 训练集的标签

Returns:
	p1 - 正例和反例的条件概率
    p1_list - 训练集中正例下各属性的条件概率
    p0_list - 训练集中反例下各属性的条件概率
Author:
	polar-region
Blog:
	https://github.com/polar-region/Machine-Learning
Modify:
	2022-03-08
"""
def train_melon(X, y):
    m, n = X.shape  # 读取数据集的二维长度
    p1 = (len(y[y == '是']) + 1) / (m + 2)  # 拉普拉斯平滑

    p1_list = []  # 用于保存正例下各属性的条件概率
    p0_list = []  # 用于保存反例下各属性的条件概率

    X1 = X[y == '是']  # 切分正例中的数据集
    X0 = X[y == '否']  # 切分反例中的数据集

    m1, _ = X1.shape  # 读取正例中数据集的二维长度
    m0, _ = X0.shape  # 读取正例中数据集的二维长度

    # 求各属性的条件概率
    for i in range(n):
        xi = X.iloc[:, i]  # 单独属性的数据集
        p_xi = namedtuple(X.columns[i], ['is_continuous', 'conditional_pro'])  # 用于储存每个变量的情况

        is_continuous = type_of_target(xi) == 'continuous'
        xi1 = X1.iloc[:, i]
        xi0 = X0.iloc[:, i]
        if is_continuous:  # 连续值时，conditional_pro 储存的就是 [mean, var] 即均值和方差
            xi1_mean = np.mean(xi1)
            xi1_var = np.var(xi1)
            xi0_mean = np.mean(xi0)
            xi0_var = np.var(xi0)

            p1_list.append(p_xi(is_continuous, [xi1_mean, xi1_var]))
            p0_list.append(p_xi(is_continuous, [xi0_mean, xi0_var]))
        else:  # 离散值时直接计算各类别的条件概率
            unique_value1 = xi1.unique()  # 取值情况
            nvalue1 = len(unique_value1)  # 取值个数

            unique_value0 = xi0.unique()  # 取值情况
            nvalue0 = len(unique_value0)  # 取值个数

            xi1_value_count = pd.value_counts(xi1)[unique_value1].fillna(0) + 1  # 计算正样本中，该属性每个取值的数量，并且加1，即拉普拉斯平滑
            xi0_value_count = pd.value_counts(xi0)[unique_value0].fillna(0) + 1

            p1_list.append(p_xi(is_continuous, xi1_value_count / (m1 + nvalue1)))
            p0_list.append(p_xi(is_continuous, xi0_value_count / (m0 + nvalue0)))

    return p1, p1_list, p0_list

"""
函数说明:通过预测得到函数结果

Parameters:
    p1 - 正例和反例的条件概率
    p0_list - 训练集中正例下各属性的条件概率
    p1_list - 训练集中正例下各属性的条件概率

Returns:
	求解结果
Author:
	polar-region
Blog:
	https://github.com/polar-region/Machine-Learning
Modify:
	2022-03-08
"""
def predict_melon(x, p1, p1_list, p0_list):
    n = len(x)

    x_p1 = p1
    x_p0 = 1 - p1
    for i in range(n):
        p1_xi = p1_list[i]
        p0_xi = p0_list[i]

        if p1_xi.is_continuous:  # 读取连续属性的数据进行计算
            mean1, var1 = p1_xi.conditional_pro
            mean0, var0 = p0_xi.conditional_pro
            x_p1 *= 1 / (np.sqrt(2 * np.pi) * var1) * np.exp(- (x[i] - mean1) ** 2 / (2 * var1 ** 2))
            x_p0 *= 1 / (np.sqrt(2 * np.pi) * var0) * np.exp(- (x[i] - mean0) ** 2 / (2 * var0 ** 2))
        else:  # 读取离散属性的数据进行计算
            x_p1 *= p1_xi.conditional_pro[x[i]]
            x_p0 *= p0_xi.conditional_pro[x[i]]
    
    print(x_p0,x_p1)

    if x_p1 > x_p0:
        return '是'
    else:
        return '否'

if __name__ == '__main__':
    data_path = r'melon.csv'
    data = pd.read_csv(data_path, index_col=0)  # 读取数据集

    X = data.iloc[:, :-1]  # 训练集的数据集
    y = data.iloc[:, -1]  # 训练集的标签集
    p1, p1_list, p0_list = train_melon(X, y)

    x_test = X.iloc[0, :]   # 书3-3题目测1 其实就是第一个数据

    print(predict_melon(x_test, p1, p0_list, p1_list))
