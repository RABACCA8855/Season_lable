# -*- coding: utf-8 -*-
# @Time    : 2022/1/5 10:36
# @Author  : Jiajun LUO
# @File    : words_extract.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import time, json, re, urllib.parse, logging, argparse, os
from nltk.corpus import stopwords
from nltk.util import ngrams, everygrams
from multiprocessing import Pool
from tqdm import tqdm
from collections import defaultdict
from functools import partial
from sklearn.feature_extraction.text import CountVectorizer


class NgramCount:
    def __init__(self):
        self.terms = []  # 存放去除停用词的搜索词集
        self.ngram_dict = defaultdict(int)  # 放ngrams频次排序的字典
        self.ngrams_list = []  # 存放所有ngram短语的列表
        self.stop_words = set(stopwords.words('english'))  # 179个停用词

    def clean_data(self, data):
        """
        :param data: 原始ARA文件的筛选出的所有搜索词汇
        :return: 生成存放去除停用词的搜索词集
        """
        for line in data:
            words = line.split()
            for word in words[::-1]:
                if word in self.stop_words:
                    words.remove(word)
            new_word = ' '.join(words)
            self.terms.append(new_word)
        # pd.DataFrame(terms,columns=['Search Term']).to_csv('corpus/del_stopwords.csv', index=False)
        # print(self.terms)

    def split_term(self, s):
        """
        :param s: 单条搜索词字符串
        :return: 搜索词生成的ngram word list
        """
        # print(s)
        s = s.lower()
        # s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
        tokens = [token for token in s.split(" ") if token != ""]
        # output = list(ngrams(tokens, 2))
        output =list(everygrams(tokens, min_len=1, max_len=3))
        result = [" ".join(i) for i in output]
        # 下面代码可以检测可能ngram切错的词汇
        # if 'r co' in result:
        #     print(s)
        #     print(tokens)
        #     print(output)
        # print(result)
        return result

    def generate_ngram(self):
        """
        产生ngrams的所有短语
        """
        for line in tqdm(self.terms):
            self.ngrams_list.extend(self.split_term(line))

        # # ngrams列表嵌套合并为单链表
        # self.ngrams_list = [j for i in res for j in i]
        self.ngrams_list = list(set(self.ngrams_list))  # 去重
        print('ngram total: ', len(self.ngrams_list))

    def process_one(self, item):
        """
        :param item: 单个ngram短语
        :return:
        这里不能用多线程处理，多线程无法同时优化全局变量的值，因为self.ngram_dict的计算是依赖序列关系的
        多线程只能处理单条记录，无法用并行结果合并
        """
        for word in self.terms:
            if item in word:
                # print(item, ':', word)
                self.ngram_dict[item] += 1

    def generate_dict(self):
        """
        :return: 生成放ngrams频次排序的字典
        """

        for line in tqdm(self.ngrams_list):
            self.process_one(line)

        # with Pool(12) as pool:
        #     batch = []
        #     for line in tqdm(self.ngrams_list):
        #         if len(batch) == 1000:
        #             pool.map(self.process_one, batch)
        #             print("1000: ",self.ngram_dict)
        #             batch = []
        #         batch.append(line)
        #
        #     # batch按照整百等分后，还有余数需要处理
        #     if len(batch) != 0:
        #         pool.map(self.process_one, batch)
        return self.ngram_dict


def main():
    # data = pd.read_csv('data/rank_10w.csv')
    # data['Search Term'].to_csv('corpus/original_words.csv', index=False)
    data = pd.read_csv('corpus/original_words.csv')
    data = data['Search Term'].values.tolist()  # 所有词汇

    # print(data[:2000])
    data = data[:2000]
    NC = NgramCount()
    NC.clean_data(data)
    NC.generate_ngram()
    ngram_dict = NC.generate_dict()

    # print(ngram_dict)
    d = sorted(ngram_dict.items(), key=lambda item: item[1], reverse=True)
    res = pd.DataFrame(d, columns=[['Ngram Term', 'count']])
    # print(res)
    res.to_csv('corpus/everygrams_count.csv')

    # # NC = NgramCount()
    # sent = 'mytv enter code tv sign'.split()
    # print(type(sent))
    # res = list(everygrams(sent,min_len=1, max_len=3))
    # print(res)


if __name__ == '__main__':
    main()
