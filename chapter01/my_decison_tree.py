"""
实现决策树，并与sklearn中的决策树对比
"""

from sklearn.datasets import load_breast_cancer
import pandas as pd
from numpy import log2
from collections import Counter
from sklearn.model_selection import train_test_split
import sklearn.metrics as ms
from sklearn.tree import DecisionTreeClassifier


def get_cross_entropy(elements):
    """
    交叉熵函数
    :param elements: [1,1,0,1,0,...]
    :return: 输入序列的信息熵
    """
    counter = Counter(elements)
    num_sum = len(elements)
    probs = [counter[i]/num_sum for i in set(elements)]
    return -1 * sum(p*log2(p) for p in probs if p != 0)

def split_data_by_feature_value(data,spliter_feature):
    """
    通过输入一个pandas数据，根据特征名和特征值将数据分割成两部分
    :param data: pandas的Dataframe
    :param spliter_feature: 是元组，分别为特征名和特征值
    :return: 切割后的两个数据
    """
    f, v = spliter_feature
    data1 = data[data[f] < v]
    data2 = data[data[f] >= v]
    data1 = data1.drop(f,axis=1)
    data2 = data2.drop(f,axis=1)
    return data1, data2

def no_have_feature(data):
    """
    当data没有特征了，只有"target"的时候，无法再切割了，
    此时对"target"里的数据进行投票，即哪种输出多就输出哪种
    :param data:
    :return:
    """
    fields = data.columns
    if len(fields) == 1:
        counter = Counter(data["target"])
        if counter[0] >= counter[1]: # 0类别的数量比1类别的数量多
            return "0"
        else:
            return "1"

def find_min_spliter(data):
    """
    输入一个data，通过递归，选择每层能使增益最大的特征名和特征值，并保持到一个字典中
    :param data: 要生成决策树的数据
    :return: 一个字典，从头到尾，哪个特征值的特征名能使当前的交叉熵最小
    """
    gain_data = get_cross_entropy(data["target"])
    if gain_data == 0:  # 如果输入的data的"target"交叉熵为0，则结果非常纯，就不用继续切割了，返回此时的target值
        return data["target"].tolist()[0]

    if no_have_feature(data):  # 如果输入的data没有特征值了，就不再切割了，直接返回投票的结果
        return int(no_have_feature(data))

    fields = data.columns.tolist()[:-1] # data的特征值，不包含最后的target这一列
    spliter_feature = None #能使增益最大的特征名和特征值
    min_E = float("inf") #增益最大时的交叉熵
    for f in fields:
        value = set(data[f])
        for v in value:
            target1 = data[data[f] < v]["target"]
            target2 = data[data[f] >= v]["target"]
            len_t1, len_t2 = len(target1), len(target2)
            len_sum = len_t1 + len_t2
            if len_t1 == 0: # 如果target1为空，那target1交叉熵不用再计算了
                E1 = 0
            else:
                E1 = get_cross_entropy(target1) * len_t1 / len_sum

            if len_t2 == 0: # 如果target2为空，那target2交叉熵不用再计算了
                E2 = 0
            else:
                E2 = get_cross_entropy(target2) * len_t2 / len_sum

            E = E1 + E2
            if E < min_E: #如果此特征名的特征值比之前的效果都好，那就把当前的特征名的特征值记为最好的选择
                min_E = E
                spliter_feature = (f,v)

    data1, data2 = split_data_by_feature_value(data, spliter_feature)
    if not len(data1): # 如果划分完数据后，data1为空，就不再对data1继续划分了
        return {
                    spliter_feature: {
                                        "no" : find_min_spliter(data2)
                                    }
                }
    if not len(data2): # 如果划分完数据后，data2为空，就不再对data2继续划分了
        return {
                    spliter_feature: {
                                        "yes": find_min_spliter(data1)
                                    }
                }
    return {
                spliter_feature: {
                                "yes": find_min_spliter(data1),
                                "no" : find_min_spliter(data2)
                                }
            }


def predict(model, x):
    """
    通过字典和输入的向量的各种特征值，获得字典的最后的值
    :param model: 字典
    :param x: 包含特征值的一个向量
    :return: 预测结果
    """
    if isinstance(model, int):
        return model
    while model:
        spliter_feature = model.keys()
        f, v = list(spliter_feature)[0]
        values = list(model.values())[0]

        if float(x[f]) < v:
            return predict(values["yes"], x)
        else:
            return predict(values["no"], x)

def get_yhat(model,data):
    """
    输入多个样本数据，输出预测结果
    :param model: 字典
    :param data: 多个样本数据
    :return: 每个样本预测的列表
    """
    yhat = []
    for i in range(len(data)):
        yhat.append(predict(model, data[i:i+1]))
    return yhat

# 1 准备数据
cancer = load_breast_cancer()
cancer_data = cancer["data"]
cancer_target = cancer["target"]
cancer_names = cancer["feature_names"]
data = pd.DataFrame(cancer_data,columns=cancer_names)
data["target"] = cancer_target
train, test = train_test_split(data, test_size=0.2)

# 2 构建自己定义的模型，并评估
print("正在构建模型，用时约1分钟，请稍等...")
model = find_min_spliter(train)
yhat = get_yhat(model,test)
y = test["target"]

print("自定义模型的混淆矩阵为:")
print(ms.confusion_matrix(y, yhat))
print("自定义模型的f1_score为:")
print(ms.f1_score(y, yhat))

# 3 用sklearn里的API定义模型，并评估
model_from_sk = DecisionTreeClassifier()
model_from_sk.fit(train.iloc[:,:-1],train["target"])
yhat_from_sk = model_from_sk.predict(test.iloc[:,:-1])

print("sklearn模型的混淆矩阵为:")
print(ms.confusion_matrix(y, yhat_from_sk))
print("sklearn模型的f1_score为:")
print(ms.f1_score(y, yhat_from_sk))
