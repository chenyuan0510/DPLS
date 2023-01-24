"""
内容：单变量kpls的复写
时间：20221220
简介：k为一阶范式关系矩阵
     pls中寻优n（小于样本数）
"""
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import pearsonr
import numpy as np
import pandas as pd


# 当数据集噪音很小时会出现UserWarning: Y residual is constant at iteration 2
# warnings.warn(f"Y residual is constant at iteration {k}")的警告，所以这里测试时用warnings关闭了警告

import warnings
warnings.filterwarnings("ignore", message="Y residual is constant at iteration")


def get_test_dataset():
    t_x = np.random.rand(100)
    t_y = t_x * 2
    return t_x, t_y

def score(_x, _y, n=2, num_cv=5):
    """
    cv_r2（n交叉测试的person相关系数）计算
    :param _x: 一维向量（np.array）
    :param _y: 一维向量（np.array）
    :param n: pls选择的特征数
    :param num_cv: 交叉测试折数
    :return:
    """
    # 欧式距离核矩阵
    k = euclidean_distances(_x.reshape(-1, 1), squared=False)
    # pls建模
    mypls = PLSRegression(n_components=n)
    # 交叉测试R2 计算
    my_cv = KFold(n_splits=num_cv)
    # 需注意的是这里的y是二维数组，n行1列，应为pls输出的是二维数组
    pred_y = np.zeros((_y.shape[0], 1))
    # 进行交叉测试，需注意的是从矩阵k内索引的范围
    for split_idx in my_cv.split(k):
        mypls.fit(k[split_idx[0], :][:, split_idx[0]], _y[split_idx[0]])
        pred_y[split_idx[1]] = mypls.predict(k[split_idx[1], :][:, split_idx[0]])
    cv_r2 = pearsonr(pred_y.reshape(-1), _y)[0]**2
    return cv_r2


def _kpls(train_x, train_y, num_cv=5):
    """
    计算x,y的cv_r2,如果同时还输出测试集的化还会进行建模预测。
    :param train_x: 用来的计算cv_r2的x,一维向量（np.array）
    :param train_y: 用来的计算cv_r2的y,一维向量（np.array）
    :param num_cv:
    :return:
    """
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    max_n = 20
    # # 并行计算参数n的寻优
    # n_result = Parallel(n_jobs=1)(
    #             delayed(score)(x, y, n=i, num_cv=5) for i in range(1, max_n+1)
    #         )
    # 数据预处理
    sc = StandardScaler()
    Scale_x = sc.fit_transform(train_x.reshape(-1, 1)).reshape(-1)
    n_result = []
    for i in range(1, max_n + 1):
        try:
            n_result.append(score(Scale_x, train_y, n=i, num_cv=num_cv))
        except:
            break
    best_n = np.argmax(n_result) + 1
    best_score = np.max(n_result)


    # 结果输出
    result = {
        # 寻优的最好的n
        "n": best_n,
        # 最佳参数情况下的cv_r2
        "score": np.round(best_score,6)
    }

    return result

def dpls_score(x, y):
    return _kpls(x, y)["score"]

def dpls(data,y=None):
    if y is not None:
        res = data.apply(dpls_score, axis=0, y=y)
    else:
        res = pd.DataFrame()
        data = pd.DataFrame(data)
        for i in range(data.shape[1]):
            res[i] = data.apply(dpls_score, axis=0, y=data.iloc[:, i])
    return res

if __name__ == "__main__":
    x, y = get_test_dataset()
    test = _kpls(x, y)
    print(test)