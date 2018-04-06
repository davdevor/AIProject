import time
import random
import numpy as np
import tensorflow as tf


def build_rnn(in_size, lstm_size, num_layers, out_size, learning_rate=0.001):

    # initializes last state with all zeros
    lstm_last_state = np.zeros(num_layers * 2 * lstm_size)

    with tf.variable_scope("rnn"):
        # placeholder for the xinput with size in_size
        xinput = tf.placeholder(tf.float32, shape=(None, None, in_size), name='xinput')

        # initial state for lstm
        lstm_init_value = tf.placeholder(tf.float32, shape=(None, num_layers * 2 * lstm_size), name="lstm-init_value")

        # the list of the lstm_cells each item in it is one layer
        lstm_cells = []
        for i in range(num_layers):
            # creates a lstm cell and appends to lstm_cells the cell holds a number of cells equal to lstm_size
            lstm_cells.append(tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=False))

        # create a RNN cell composed sequentially of a number of RNNCells
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells, state_is_tuple=False)

        # outputs is a tensor of shape [batch_size, max_time, cell_state_size]
        # lstm_is a N-tuple where N is the number of LSTMCells containing a
        # tf.contrib.rnn.LSTMStateTuple for each cell
        outputs, lstm_new_state = tf.nn.dynamic_rnn(lstm, xinput, initial_state=lstm_init_value, dtype=tf.float32)

        # the weights for the network
        rnn_out_W = tf.Variable(tf.random_normal((lstm_size, out_size), stddev=.01))
        # the bias for the network
        rnn_out_B = tf.Variable(tf.random_normal((out_size,), stddev=.01))

        outputs_reshaped = tf.reshape(outputs, [-1, lstm_size])

        # the output of the network is the actual outputs mat mul with the weights then you add the biases to it
        network_output = tf.matmul(outputs_reshaped, rnn_out_W) + rnn_out_B
        batch_time_shape = tf.shape(outputs)
        # softmax squashes a K-dimensional vector z of arbitrary real values to a K-dimensional vector of real values in
        # in the range (0,1) that add up to 1
        final_outputs = tf.reshape(tf.nn.softmax(network_output), (batch_time_shape[0], batch_time_shape[1], out_size))
        y_batch = tf.placeholder(tf.float32, (None, None, out_size))
        y_batch_long = tf.reshape(y_batch, [-1, out_size])

        # computes softmax_cross_entropy between network_output and y_batch_long
        # it measures the probability error in discrete classification tasks in which the classes are mutually exclusive
        # reduce_mean computes the mean of elements across dimensions of a tensor
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=network_output, labels=y_batch_long))
        # Optimizer that implements the RMSProp algorithm
        # minimize takes care of computing the gradients and applying them to the variables in cost
        train_op = tf.train.RMSPropOptimizer(learning_rate, .9).minimize(cost)

        return cost, train_op, xinput, y_batch, lstm_init_value, final_outputs, lstm_last_state, lstm_new_state


def train_batch(session, cost, train_op, xinput, y_batch, lstm_init_value, xbatch, ybatch, num_layers, lstm_size):
    init_value = np.zeros((xbatch.shape[0], num_layers * 2 * lstm_size))
    newcost, _ = session.run([cost, train_op], feed_dict={xinput: xbatch, y_batch: ybatch, lstm_init_value: init_value})
    return newcost


def embed_to_vocab(data_, vocab):

    # Embed string to character-arrays -- it generates an array len(data) x len(vocab).
    # Vocab is a list of elements.
    l1 = len(data_)
    l2 = len(vocab)
    data = np.zeros((l1, l2))
    cnt = 0
    for s in data_:
        v = [0.0] * len(vocab)
        # get the position of the character stored in s in vocab and put a 1 in that smae postion in v
        v[vocab.index(s)] = 1.0
        # store v in the right place in data
        data[cnt, :] = v
        cnt += 1
    # data contains the onehot encoding
    return data

def load_data(input):
    # Load the data
    data_ = ""
    with open(input, 'r', encoding='UTF-8') as f:
        data_ += f.read()
    data_ = data_.lower()
    # Convert to 1-hot coding
    # vocab is a set of all the characters from the file
    vocab = sorted(list(set(data_)))
    data = embed_to_vocab(data_, vocab)
    return data, vocab


# Input: X is a single element, not a list!
def run_step(x, num_layers, lstm_size,lstm_last_state,final_outputs, lstm_new_state, xinput, lstm_init_value,session, init_zero_state=True):
    # Reset the initial state of the network if generating new string
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
    # update last state to use in next run of nn
    lstm_last_state = next_lstm_state[0]
    return out[0][0], lstm_last_state


def main():
    # the first word of every test
    test_prefix = 'the'
    # the save file
    ckpt_file = "saved/model.ckpt"
    data, vocab = load_data('poetry.txt')
    train = False
    # input and output sizes
    # vocab is a list of unique characters
    in_size = out_size = len(vocab)
    lstm_size = 256
    num_layers = 2
    batch_size = 64
    time_steps = 100
    training_batches = 20000
    # Number of test characters of text to generate after training the network
    test_length = 500

    # Initialize the network
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    # build rnn
    cost, train_op, xinput, y_batch, lstm_init_value, final_outputs, lstm_last_state, lstm_new_state = build_rnn(in_size, lstm_size, num_layers, out_size, .001)
    # initialize session and saver
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    if train:
        last_time = time.time()
        # initialize batch to zeros
        batch = np.zeros((batch_size, time_steps, in_size))
        batch_y = np.zeros((batch_size, time_steps, in_size))
        possible_batch_ids = range(data.shape[0] - time_steps - 1)

        for i in range(training_batches):
            # Sample time_steps consecutive samples from the data set text file
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
        # restore session with save tensorflow variables
        saver.restore(sess, ckpt_file)

        for x in range(3):
            test_prefix = test_prefix.lower()
            # run each character of the the test_prefix through the nn
            # this is done to start the generation, you need to give it a starting point
            for i in range(len(test_prefix)):
                out, lstm_last_state = run_step(embed_to_vocab(test_prefix[i], vocab), num_layers,lstm_size,lstm_last_state,final_outputs,lstm_new_state,xinput,lstm_init_value,sess,i == 0)

            print("poem:")
            gen_str = test_prefix
            for i in range(test_length):
                # Sample character from the network according to the generated
                # p is the list of probabilities that correspond to each position in vocab
                # p is set to out because out is the probabilities from the nn
                element = np.random.choice(range(len(vocab)), p=out)
                # add the choice to the string
                # the choice is one character from the vocab
                gen_str += vocab[element]
                # give the choice to the nn to run again
                out, lstm_last_state = run_step(embed_to_vocab(vocab[element], vocab), num_layers,lstm_size,lstm_last_state,final_outputs,lstm_new_state,xinput,lstm_init_value,sess, False)

            # the final output string
            print(gen_str)


main()

