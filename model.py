'''
@Author: feiyun
@Github: https://github.com/feiyunamy
@Blog: https://blog.feiyunamy.cn
@Date: 2019-07-11 16:44:58
@LastEditors: feiyun
@LastEditTime: 2019-07-16 21:31:52
'''
import tensorflow as tf
class ModelConfig(object):
    # 模型参数
    embedding_dim = 200      # 词向量维度
    seq_length = 6          # 序列长度
    vocab_size = 5000       # 词汇表达小
    units = 1024            # 隐层单元

    num_layers= 2           # 隐藏层层数
    hidden_dim = 128        # 隐藏层神经元
    rnn = 'gru'             # lstm 或 gru
    decoder_rnn = 'gru'     # lstm 或 gru
    decoder_num_layers= 2   # 隐藏层层数
    decoder_hidden_dim = 128# 隐藏层神经元

    dropout_keep_prob = 0.8 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 256         # 每批训练大小 
    num_epochs = 50          # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard

    target_length = 6
    mode = 'train'           # 加载模式 对应decoder的过程

class Model(tf.keras.Model):
    def __init__(self, config):
        super(Model, self).__init__()
        #超参数设置
        self.config = config

        # 构建模型
        with tf.device('/cpu:0'):
            self.embedding = tf.keras.layers.Embedding(self.config.vocab_size, self.config.embedding_dim)

        with tf.name_scope("rnn"):
        
            self.gru = tf.keras.layers.GRU(self.config.units,
                                            return_sequences=True,
                                            recurrent_activation='sigmoid',
                                            recurrent_initializer='glorot_uniform',
                                            stateful=True)
        with tf.name_scope("fc"):
            self.fc = tf.keras.layers.Dense(self.config.vocab_size)
        

    def call(self, x):
        embedding = self.embedding(x)

        # output at every time step
        # output shape == (batch_size, seq_length, hidden_size)
        output = self.gru(embedding)

        # The dense layer will output predictions for every time_steps(seq_length)
        # output shape after the dense layer == (seq_length * batch_size, vocab_size)
        prediction = self.fc(output)

        # states will be used to pass at every step to the model while training
        return prediction