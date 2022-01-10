"""
处理中间文件1year.csv，转化为月份平均排名分布文件rank_distribution.csv
"""


import pandas as pd
import numpy as np
from numpy import *



def main():
    data_01 = pd.read_csv('data/small_case.csv', index_col=0)
    # 生成月份清单给最终的df打列名
    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    new_data = []
    data_02 = np.array(data_01).tolist()

    # 取每一个关键词的月份数据count求月份排名的平均值
    for line in data_02:
        # 存储数据格式 {‘01’:[1,1,2,...]，‘02’：[3,4,..]}
        rank_dict = {}
        tmp_list = line[1].split()
        # print(tmp_list)
        # 每条count记录用空格分开后，进行统计
        for item in tmp_list:
            # 有时候月份信息里会出现‘0’的情况
            if item == '0':
                break
            rank = item.split('%')[0].replace(',', '')
            month = item.split('%')[1]
            if month not in rank_dict:
                rank_dict[month] = [int(rank)]
            else:
                rank_dict[month].append(int(rank))

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
        # print(s)
        new_data.append(s)

    new_df = pd.DataFrame(new_data)

    res = pd.concat([data_01['Search Term'], new_df], axis=1, ignore_index=False)
    res.to_csv('data/rank_distribution.csv', index=False)

if __name__ == '__main__':
    main()