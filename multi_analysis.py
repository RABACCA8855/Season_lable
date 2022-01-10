# -*- coding: utf-8 -*-
# @Time    : 2022/1/4 15:42
# @Author  : Jiajun LUO
# @File    : multi_analysis.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from kneed import KneeLocator  ##寻求拐点位置的函数
import time, json, re, urllib.parse, logging, argparse, os
from multiprocessing import Pool
from tqdm import tqdm
from matplotlib import style

style.use('seaborn-whitegrid')

months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
months_int = [int(i) for i in months]


def Normalize(array):
    '''
    Normalize the list
    '''
    mx = np.nanmax(array)
    mn = np.nanmin(array)
    t = (array - mn) / (mx - mn)
    return t


def cal_slope(knees, y):
    """
    :param knees: 拐点的索引set集合
    :param y: y值list
    :return: 该拐点的平均斜率
    """
    result = {}
    for i in knees:
        # print(y[i - 1])
        if i not in [1, 12]:
            # 获取拐点前的坐标和拐点下一个坐标
            pre, knee, after = y[i - 2], y[i - 1], y[i]
            # 由于月份的差值是等值1，需要计算拐点前后的梯度幅度后取一个平均值，可以代表该拐点的最终梯度
            # 梯度越大，代表波峰比较陡峭；梯度越小，代表波峰比较平滑
            result.update({str(i): round(abs(knee - pre) + abs(knee - after) / 2, 3)})
        elif i == 12:
            pre, knee = y[i - 2], y[i - 1]
            result.update({str(i): round(abs(knee - pre) , 3)})
    return result


def analysis_func(line):
    # print(line)
    # 计算均值
    mean = round(np.mean(line), 4)

    # 排名归一到[0,1],计算方差
    line = Normalize(line)
    var = round(np.var(line), 4)

    # 计算拐点和斜率
    x = months_int
    y = line
    kneedle = KneeLocator(x,
                          y,
                          curve='concave',
                          direction='decreasing',
                          online=True)
    knees = kneedle.all_knees
    slope = cal_slope(knees, y)
    # print(slope)
    return [mean, var, slope]


def main():
    start = time.time()
    data = pd.read_csv('data/rank_distribution.csv')

    data_months = np.array(data[months]).tolist()
    new_data = []  # 存放12个月的平均排名列表

    # 以下的代码是为了查看执行的进度条，和上面的代码速度上
    with Pool(12) as pool:
        batch = []
        for line in tqdm(data_months):
            if len(batch) == 1000:
                new_data.extend(pool.map(analysis_func, batch))
                batch = []
            batch.append(line)

        # batch按照整百等分后，还有余数需要处理
        if len(batch) != 0:
            new_data.extend(pool.map(analysis_func, batch))

    # print(new_data)
    new_df = pd.DataFrame(new_data, columns=['均值', '方差', '拐点/斜率'])
    res = pd.concat([data[['Search Term']+months], new_df], axis=1, ignore_index=False)
    res.to_csv('result/analysis_2000.csv', index=False)
    end = time.time()
    print('cost time (min):', round((end - start) / 60))


if __name__ == '__main__':
    main()
