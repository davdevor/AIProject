import time
import random
import numpy as np
import tensorflow as tf


def build_rnn(in_size, lstm_size, num_layers, out_size, learning_rate=0.001):
    lstm_laste_state = np.zeros(num_layers * 2 * lstm_size)
    with tf.variable_scope("rnn"):
        xinput = tf.placeholder(tf.float32, shape=(None, None, in_size), name='xinput')
        lstm_init_value = tf.placeholder(tf.float32, shape=(None, num_layers * 2 * lstm_size), name="lstm-init_value")
        lstm_cells = []
        for i in range(num_layers):
            lstm_cells.append(tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=False))
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells, state_is_tuple=False)
        outputs, lstm_new_state = tf.nn.dynamic_rnn(lstm, xinput, initial_state=lstm_init_value, dtype=tf.float32)
        rnn_out_W = tf.Variable(tf.random_normal((lstm_size, out_size), stddev=.01))
        rnn_out_B = tf.Variable(tf.random_normal((out_size,), stddev=.01))
        outputs_reshaped = tf.reshape(outputs, [-1, lstm_size])
        network_output = tf.matmul(outputs_reshaped, rnn_out_W) + rnn_out_B
        batch_time_shape = tf.shape(outputs)
        final_outputs = tf.reshape(tf.nn.softmax(network_output), (batch_time_shape[0], batch_time_shape[1], out_size))
        y_batch = tf.placeholder(tf.float32, (None, None, out_size))
        y_batch_long = tf.reshape(y_batch, [-1, out_size])
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=network_output, labels=y_batch_long))
        train_op = tf.train.RMSPropOptimizer(learning_rate, .9).minimize(cost)

        return cost, train_op, xinput, y_batch, lstm_init_value, final_outputs, lstm_laste_state, lstm_new_state


def train_batch(session, cost, train_op, xinput, y_batch, lstm_init_value, xbatch, ybatch, num_layers, lstm_size):
    init_value = np.zeros((xbatch.shape[0], num_layers * 2 * lstm_size))
    newcost, _ = session.run([cost, train_op], feed_dict={xinput: xbatch, y_batch: ybatch, lstm_init_value: init_value})
    return newcost

def embed_to_vocab(data_, vocab):
    """
    Embed string to character-arrays -- it generates an array len(data)
    x len(vocab).

    Vocab is a list of elements.
    """
    l1 = len(data_)
    l2 = len(vocab)
    data = np.zeros((l1, l2))
    cnt = 0
    for s in data_:
        v = [0.0] * len(vocab)
        v[vocab.index(s)] = 1.0
        data[cnt, :] = v
        cnt += 1
    return data

def decode_embed(array, vocab):
    return vocab[array.index(1)]

def load_data(input):
    # Load the data
    data_ = ""
    with open(input, 'r', encoding='UTF-8') as f:
        data_ += f.read()
    data_ = data_.lower()
    # Convert to 1-hot coding
    vocab = sorted(list(set(data_)))
    data = embed_to_vocab(data_, vocab)
    return data, vocab


# Input: X is a single element, not a list!
def run_step(x, num_layers, lstm_size,lstm_last_state,final_outputs, lstm_new_state, xinput, lstm_init_value,session, init_zero_state=True):
    # Reset the initial state of the network.
    if init_zero_state:
        init_value = np.zeros((num_layers * 2 * lstm_size,))
    else:
        init_value = lstm_last_state
    out, next_lstm_state = session.run([final_outputs, lstm_new_state],
        feed_dict={
            xinput: [x],
            lstm_init_value: [init_value]
        }
    )
    lstm_last_state = next_lstm_state[0]
    return out[0][0], lstm_last_state


def main():
    TEST_PREFIX = 'The'
    ckpt_file = "saved/model.ckpt"
    data, vocab = load_data('poetry.txt')
    train = False
    in_size = out_size = len(vocab)
    lstm_size = 256
    num_layers = 2
    batch_size = 64
    time_steps = 100
    NUM_TRAIN_BATCHES = 20000
    # Number of test characters of text to generate after training the network
    LEN_TEST_TEXT = 500

    # Initialize the network
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    cost, train_op, xinput, y_batch, lstm_init_value, final_outputs, lstm_last_state, lstm_new_state = build_rnn(in_size, lstm_size, num_layers, out_size, .001)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    if train:
        last_time = time.time()
        batch = np.zeros((batch_size, time_steps, in_size))
        batch_y = np.zeros((batch_size, time_steps, in_size))
        possible_batch_ids = range(data.shape[0] - time_steps - 1)

        for i in range(NUM_TRAIN_BATCHES):
            # Sample time_steps consecutive samples from the dataset text file
            batch_id = random.sample(possible_batch_ids, batch_size)

            for j in range(time_steps):
                ind1 = [k + j for k in batch_id]
                ind2 = [k + j + 1 for k in batch_id]

                batch[:, j, :] = data[ind1, :]
                batch_y[:, j, :] = data[ind2, :]

            cst = train_batch(sess, cost, train_op, xinput, y_batch, lstm_init_value, batch, batch_y, num_layers, lstm_size)

            if (i % 100) == 0:
                new_time = time.time()
                diff = new_time - last_time
                last_time = new_time
                print("batch: {}  loss: {}  speed: {} batches / s".format(i, cst, 100 / diff))
                saver.save(sess, ckpt_file)
    else:
        # 2) GENERATE LEN_TEST_TEXT CHARACTERS USING THE TRAINED NETWORK
        saver.restore(sess, ckpt_file)
        TEST_PREFIX = TEST_PREFIX.lower()
        for i in range(len(TEST_PREFIX)):
            out, lstm_last_state = run_step(embed_to_vocab(TEST_PREFIX[i], vocab), num_layers,lstm_size,lstm_last_state,final_outputs,lstm_new_state,xinput,lstm_init_value,sess,i == 0)

        print("Sentence:")
        gen_str = TEST_PREFIX
        for i in range(LEN_TEST_TEXT):
            # Sample character from the network according to the generated
            # output probabilities.
            element = np.random.choice(range(len(vocab)), p=out)
            gen_str += vocab[element]
            out, lstm_last_state = run_step(embed_to_vocab(vocab[element], vocab), num_layers,lstm_size,lstm_last_state,final_outputs,lstm_new_state,xinput,lstm_init_value,sess, False)

        print(gen_str)

main()
