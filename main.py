'''
@Author: feiyun
@Github: https://github.com/feiyunamy
@Blog: https://blog.feiyunamy.cn
@Date: 2019-07-11 21:16:51
@LastEditors: feiyun
@LastEditTime: 2019-07-16 21:39:01
'''
from utils import get_time_dif
import os
import pickle
from model import Model, ModelConfig
from data_loader import process_file, batch_iter
import tensorflow as tf
tf.enable_eager_execution()
import time
import sys
import numpy as np

base_dir = 'data/'
train_dir = os.path.join(base_dir, '7_simplified.txt')
vocab_dir = os.path.join(base_dir, 'word2id.pkl')

# Using adam optimizer with default arguments
optimizer = tf.train.AdamOptimizer()

# Using sparse_softmax_cross_entropy so that we don't have to create one-hot vectors
def loss_function(real, preds):
    return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)
    
def train():
    model.build(tf.TensorShape([config.batch_size, config.seq_length]))
    model.summary()
    x_train, y_train = process_file(train_dir, word2id, 6)
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        start = time.time()
        hidden = model.reset_states()
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for (batch, (x_batch, y_batch)) in enumerate(batch_train):
            if batch < 601:
                with tf.GradientTape() as tape:
                    # feeding the hidden state back into the model
                    # This is the interesting step
                    x_batch = tf.convert_to_tensor(x_batch)
                    y_batch = tf.convert_to_tensor(y_batch)
                    predictions = model(x_batch)
                    loss = loss_function(y_batch, predictions)

                grads = tape.gradient(loss, model.variables)
                optimizer.apply_gradients(zip(grads, model.variables))

                if batch % 100 == 0:
                    print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch+1,
                                                                    batch,
                                                                    loss))
        # saving (checkpoint) the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.save_weights(checkpoint_prefix)
        print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
def test():
    model = Model(config)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.build(tf.TensorShape([1, None]))
    # Evaluation step (generating text using the learned model)

    # You can change the start string to experiment
    start = input()
    poetry = []
    for start_string in start:
        # Number of characters to generate
        num_generate = 6

        # Converting our start string to numbers (vectorizing)
        input_eval = [word2id[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        # Empty string to store our results
        text_generated = start_string

        # Low temperatures results in more predictable text.
        # Higher temperatures results in more surprising text.
        # Experiment to find the best setting.
        temperature = 5
        # Here batch size == 1
        # model.reset_states()
        for i in range(num_generate):
            predictions = model(input_eval)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)

            # using a multinomial distribution to predict the word returned by the model
            predictions = predictions / temperature
            predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()

            # We pass the predicted word as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated += idx2char[predicted_id]
        poetry.append(text_generated)
    pretty_print(poetry)

def pretty_print(poetry):
    print('=======================')
    # print('{:>7}'.format('无题'))
    line_num = 0
    for line in poetry:
        line_num += 1
        if line_num % 2 ==0:
            print(line + '。')
        else:
            print(line + '，')
    print('=======================')

        


if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) != 2 or sys.argv[1] not in ['train','test']:
        raise ValueError("""usage: python run_rnn.py [train / test]""")
    print('Configuring RNN model...')
    config = ModelConfig()
    # if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
    #     build_vocab(train_dir, vocab_dir, config.vocab_size)
    # categories, cat_to_id = read_category()
    # words, word_to_id = read_vocab(vocab_dir)
    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    with open(vocab_dir, 'rb') as f:
        word2id = pickle.load(f)
    idx2char = list(word2id.keys())
    config.vocab_size = len(word2id.keys())
    config.units = 1024
    config.embedding_dim = 200
    model = Model(config)
    if sys.argv[1] == 'train':
        train()
    else:
        test()
    