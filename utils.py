'''
@Author: feiyun
@Github: https://github.com/feiyunamy
@Blog: https://blog.feiyunamy.cn
@Date: 2019-07-11 10:46:19
@LastEditors: feiyun
@LastEditTime: 2019-07-11 21:17:23
'''
import os
import json
import re
from tqdm import tqdm
from langconv import Converter
import time
from datetime import timedelta
import random

def merge_json():
    w_5_path = './data/5_traditional.txt'
    w_7_path = './data/7_traditional.txt'
    root = 'data'
    file_list = [i for i in os.listdir(os.path.join(root, 'json')) if 'poet' in i]
    for one in tqdm(file_list):
        with open(os.path.join(root, 'json', one), 'r', encoding = 'utf-8') as f:
            poems = json.load(f)
            content_5 = [''.join(poem["paragraphs"]) for poem in poems if len(re.split('[，。]',''.join(poem["paragraphs"]))[0]) == 5]
            content_7 = [''.join(poem["paragraphs"]) for poem in poems if len(re.split('[，。]',''.join(poem["paragraphs"]))[0]) == 7]
            to_txt(w_5_path, content_5)
            to_txt(w_7_path, content_7)
    w_5_simp_path = './data/5_simplified.txt'
    w_7_simp_path = './data/7_simplified.txt'
    Traditional_file2Simplified_file(w_5_path, w_5_simp_path)
    Traditional_file2Simplified_file(w_7_path, w_7_simp_path)

def to_txt(path, content):
    
    with open(path, 'a', encoding = 'utf-8') as f:
        for line in content:
            f.write(line + '\n')

def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence

def Traditional_file2Simplified_file(trad_path, simple_path):
    print('Traditional_file2Simplified_file...')
    with open(trad_path, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
    with open(simple_path, 'w', encoding = 'utf-8') as f:
        for line in lines:
            f.write(Traditional2Simplified(line))

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
    
def get_mini_corpus(size):
    with open('./data/7_simplified.txt', 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
    random.shuffle(lines)
    with open('./data/mini_7_simplified.txt', 'w', encoding = 'utf-8') as f:
        i = 0
        for line in lines:
            if i <= size:
                if len(''.join(re.split('[，。]', line.strip()))) % 7 == 0:
                    f.write(line)
                    i += 1
                else:
                    pass
            else:
                break

def generate_train_pair(path):
    with open(path, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
    data = ''.join(lines)
    content_7 = [i.strip() for i in re.split('[，。]',data) if len(i.strip()) == 7]
    open('./data/singleline.txt','w',encoding = 'utf-8').write('\n'.join(content_7))

if __name__ == '__main__': 
    # merge_json()          
    # get_mini_corpus(50000)
    generate_train_pair('./data/mini_7_simplified.txt')