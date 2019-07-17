'''
@Author: feiyun
@Github: https://github.com/feiyunamy
@Blog: https://blog.feiyunamy.cn
@Date: 2019-07-11 15:18:39
@LastEditors: feiyun
@LastEditTime: 2019-07-16 21:14:55
'''
import pickle
from collections import Counter
import numpy as np
import tensorflow.keras as keras
from model import ModelConfig
import os

def read_corpus(corpus_path):
    with open(corpus_path, 'r' , encoding = 'utf-8') as f:
        lines = f.readlines()
    return lines

def get_data(data):
    first_words = []
    next_content = []
    for line in data:
        first_words.append(line.strip()[:-1])
        next_content.append(line.strip()[1:])
    return first_words, next_content
    

def process_file(filename, word_to_id, max_length=6):
    """将文件转换为id表示"""
    data = read_corpus(filename)
    first_words, next_content = get_data(data)
    assert len(first_words) == len(next_content), 'DATA ERROR!'
    first_id, next_id = [], []
    for i in range(len(first_words)):
        first_id.append([word_to_id[x] for x in first_words[i] if x in word_to_id])
        next_id.append([word_to_id[x] for x in next_content[i] if x in word_to_id])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = keras.preprocessing.sequence.pad_sequences(first_id, max_length)
    y_pad = keras.preprocessing.sequence.pad_sequences(next_id, max_length)
    return x_pad, y_pad 
    
def vocab_build(vocab_path, corpus_path, min_count):
    """
    :param vocab_path:
    :param corpus_path:
    :param min_count:
    :return:
    """
    data = read_corpus(corpus_path)
    word_list = [word for content in data for word in content]
    counter = Counter(word_list)
    words = sorted(counter.keys(), key=lambda x: counter[x], reverse=True)
    words = ['<PAD>','<GO>','<EOS>','<UKN>'] + [word for word in words if counter[word] > min_count and word.strip() not in ['','，','。']]
    open('./data/vocab.txt', mode='w', encoding = 'utf-8').write('\n'.join(words))
    msg = 'Vocab size:{:>6}'
    print(msg.format(len(words)))
    word2id = dict(zip(words, range(len(words))))
    with open(vocab_path, 'wb') as fw:
        pickle.dump(word2id, fw)

def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

if __name__ == '__main__':
    # vocab_build('./data/word2id.pkl','./data/mini_7_simplified.txt', 5)
    config = ModelConfig()
    base_dir = 'data/'
    train_dir = os.path.join(base_dir, 'singleline.txt')
    vocab_dir = os.path.join(base_dir, 'word2id.pkl')
    with open(vocab_dir, 'rb') as f:
        word2id = pickle.load(f)
    x_train, y_train = process_file(train_dir, word2id, 6)
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            print(1)