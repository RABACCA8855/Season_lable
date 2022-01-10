# -*- coding: utf-8 -*-
# @Time    : 2021/12/30 14:46
# @Author  : Jiajun LUO
# @File    : amazon_title_process.py
# @Software: PyCharm

from multiprocessing import Pool
import pickle,re
from tqdm import tqdm,trange



def tokenize_func(text):
    _puc_reg = re.compile(r"\W+")
    result = text.split()
    words = [w.strip() for w in result  # 文本过滤，过滤停顿词，并通过正则匹配过滤需要的文本
             if w.strip() and not _puc_reg.search(w)]
    return words

def main():
    # 读取title数据，并用spacy包做批量分词
    title_list = pickle.load(open("data/amazon_title.pkl", "rb"))
    title_token_list = []
    with Pool(12) as pool:
        batch = []
        for title in tqdm(title_list):
            if len(batch) == 1000:
                title_token_list.extend(pool.map(tokenize_func, batch))
                batch = []
            batch.append(title)

        if len(batch) != 0:
            title_token_list.extend(pool.map(tokenize_func, batch))
    pickle.dump(title_token_list, open("data/amazon_title_token.pkl", "wb"))

if __name__ == '__main__':
    main()
