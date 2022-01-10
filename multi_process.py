# -*- coding: utf-8 -*-
# @Time    : 2021/12/30 13:38
# @Author  : Jiajun LUO
# @File    : multi_process.py
# @Software: PyCharm

import pandas as pd
import numpy as np
from numpy import *
import time, json, re, urllib.parse, logging, argparse, os
from multiprocessing import Pool
from tqdm import tqdm

# 生成月份清单给最终的df打列名
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']


def cal_rank_mean(line):
    """
    :param text: 每个搜索词对应的count值
    :return:
    """
    rank_dict = {}
    tmp_list = line.split()
    # 有时候月份信息里会出现‘0’的情况,需要拿掉
    tmp = [i for i in tmp_list if i != '0']
    # print(tmp_list)
    # 每条count记录用空格分开后，进行统计
    for item in tmp:
        rank = item.split('%')[0].replace(',', '')
        month = item.split('%')[1]
        rank = np.array(rank, dtype='int64')
        if month not in rank_dict:
            rank_dict[month] = [rank]
        else:
            rank_dict[month].append(rank)

    # 按月存储每月的平均值 {01:4.333,02:98.111...}
    month_dict = {}
    for key, value in rank_dict.items():
        month_dict[key] = round(mean(value), 4)
    # print(month_dict)

    # 对没有排名数据的月份补零
    for month in months:
        if month not in month_dict:
            month_dict.update({month: 0})
    # 根据月份从小到大排序
    s = dict(sorted(month_dict.items(), key=lambda x: int(x[0])))
    return s


def main():
    start = time.time()
    data = pd.read_csv('/data/stef/guanguo/A_B_test/1year.csv', index_col=0)
    data_count = data['count'].values.tolist()
    new_data = []  # 存放12个月的平均排名列表

    # pool = Pool(12)
    # new_data = pool.map(cal_rank_mean,data_count)

    # 以下的代码是为了查看执行的进度条，和上面的代码速度上
    with Pool(12) as pool:
        batch = []
        for line in tqdm(data_count):
            if len(batch) == 1000:
                new_data.extend(pool.map(cal_rank_mean, batch))
                batch = []
            batch.append(line)

        # batch按照整百等分后，还有余数需要处理
        if len(batch) != 0:
            new_data.extend(pool.map(cal_rank_mean, batch))

    # print(new_data)
    new_df = pd.DataFrame(new_data)
    res = pd.concat([data[['Search Term', 'length']], new_df], axis=1, ignore_index=False)
    res.to_csv('result/rank_all.csv', index=False)
    end = time.time()
    print('cost time (min):', round((end - start) / 60))


if __name__ == '__main__':
    main()
