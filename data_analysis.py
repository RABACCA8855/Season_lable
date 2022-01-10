"""
计算rank_distribution.csv的排名方差和均值
结果输出到analysis.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from kneed import KneeLocator  ##寻求拐点位置的函数
from matplotlib import style
style.use('seaborn-whitegrid')


def Normalize(array):
    '''
    Normalize the list
    '''
    mx = np.nanmax(array)
    mn = np.nanmin(array)
    t = (array - mn) / (mx - mn)
    return t

def cal_slope(knees,y):
    """
    :param knees: 拐点的索引set集合
    :param y: y值list
    :return: 该拐点的平均斜率
    """
    result = {}
    for i in knees:
        print(y[i-1])
        if i not in [1,12]:
            pre, knee, after = y[i-2],y[i-1],y[i]
            result.update({str(i): round(abs(knee - pre) + abs(knee - after) / 2, 3)})
    return result


def main():
    data = pd.read_csv('data/rank_distribution.csv')
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    data_01 = np.array(data[months]).tolist()
    new_data = []
    for line in data_01:
        # 计算均值
        mean = round(np.mean(line), 4)
        # 排名归一到[0,1],计算方差
        line = Normalize(line)
        var = round(np.var(line), 4)
        new_data.append((var, mean))
    new_df = pd.DataFrame(new_data, columns=['方差', '排名均值'])
    result = pd.concat([data, new_df], axis=1, ignore_index=False)
    result.to_csv('data/analysis.csv', index=False)

    single = result[100:101][months].values.tolist()
    x = [int(i) for i in months]
    y = single[0]
    print(x)
    print(y)

    # pd.DataFrame({'x': x, 'y': y}).plot(x='x', y='y')
    # plt.legend()
    # plt.show()

    kneedle = KneeLocator(x,
                          y,
                          curve='concave',
                          direction='decreasing',
                          online=True)
    # 可以打印出拐点的索引坐标
    # print(kneedle.all_knees)
    # print(kneedle.all_knees_y)

    knees = kneedle.all_knees
    slope = cal_slope(knees,y)
    print(slope)

    # print(round(kneedle.all_knees_y, 3))
    # 打印拐点的标记图
    kneedle.plot_knee()
    # kneedle.plot_knee_normalized()
    plt.legend()
    plt.show()

    # 数据可视化探索，图太丑不可用
    # var_df = result['方差']
    # res = var_df.sort_values()
    # # s2 = res.sort_index()
    # print(res)
    #
    # # s2.plot(kind='line', grid=True, label='S1', title='xxx')
    # res.plot(label='S1', title='xxx')
    # # s2.plot(kind="line", linestyle='-.', color='k', marker='o')
    # plt.legend()
    # plt.show()

    # rank_df = result['排名均值'].sort_values()
    # print(rank_df)
    # rank_df.plot(label='S1', title='xxx')
    # plt.legend()
    # plt.show()

    # b = sorted(dict(Counter([round(_, 3) for _ in a['2_month'] if not pd.isna(_) & int(round(_, 3)) != 0])).items(),
    #            key=lambda x: x[0], reverse=False)
    # pd.DataFrame({'x': [_ for _ in dict(b).keys()], 'y': [_ for _ in dict(b).values()]}).plot(x='x', y='y')


if __name__ == '__main__':
    main()
