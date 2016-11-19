from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import tensorflow as tf
import time
import pickle

FLAGS = tf.app.flags.FLAGS
NUM_ARTICLES = 40

# Model Parameters
tf.app.flags.DEFINE_integer(
    'num_steps', 1, 'Number of unrolled steps before backprop.')
tf.app.flags.DEFINE_integer('embedding_size', 50, 'Size of the Embeddings.')
tf.app.flags.DEFINE_integer('hidden_size', 256, 'Size of the LSTM Layer.')

# Training Parameters
tf.app.flags.DEFINE_integer('num_epochs', 5, 'Number of Training Epochs.')
tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'Size of a batch (for training).')  # TODO: HACK
tf.app.flags.DEFINE_float('learning_rate', 1e-4,
                          'Learning rate for Adam Optimizer.')
tf.app.flags.DEFINE_float(
    'dropout_prob', 0.5, 'Keep probability, for dropout.')
tf.app.flags.DEFINE_integer(
    'eval_every', 10000, 'Print statistics every eval_every words.')

class BiRNN():
    def __init__(self, vocab_size, embedding_size, num_steps, hidden_size,
                 batch_size, learning_rate):

        self.vocab_size, self.embedding_size = vocab_size, embedding_size
        self.hidden, self.num_steps = hidden_size, num_steps
        self.bsz, self.learning_rate = batch_size, learning_rate

        # Setup Placeholders
        self.X = tf.placeholder(tf.int32, shape=[None, self.num_steps])
        self.Y = tf.placeholder(tf.int32, shape=[None, self.num_steps])
        self.keep_prob = tf.placeholder(tf.float32)

        # Instantiate Network Weights
        self.instantiate_weights()

        # Build the Inference Graph
        self.logits, self.final_states = self.inference()

        # Build the Loss Computation
        self.loss_val = self.loss()

        # Build the Training Operation
        self.train_op = self.train()

    def instantiate_weights(self):
        # Embedding Matrix
        self.E = self.weight_variable(
            [self.vocab_size, self.embedding_size], 'Embedding')

        # Basic LSTM Cell
        self.fw_cell = rnn_cell.BasicLSTMCell(self.hidden)
        self.bw_cell = rnn_cell.BasicLSTMCell(self.hidden)

        self.initial_state_fw = self.fw_cell.zero_state(self.bsz, tf.float32)
        self.initial_state_bw = self.bw_cell.zero_state(self.bsz, tf.float32)

        # Softmax Output
        self.softmax_w = self.weight_variable(
            [self.hidden, self.vocab_size], 'Softmax_Weight')
        self.softmax_b = self.weight_variable(
            [self.vocab_size], 'Softmax_Bias')

    def inference(self):
        """
        Build the inference computation graph for the model, going
        from the input to the output logits (before final softmax
        activation).

        :return Tuple of 2D Logits Tensor [bsz * steps, vocab],
        and Final State [num_layers]
        """
        # Feed input through the Embedding Layer, Dropout.
        # Shape [bsz, steps, hidden]
        emb = tf.nn.embedding_lookup(self.E, self.X)
        drop_emb = tf.nn.dropout(emb, self.keep_prob)

        # TODO: ask TA
        sequence_len = np.ones(self.bsz)
        # Feed input through dynamic_rnn
        # Shape [bsz, steps, hidden]
        outs, f_states = rnn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell, drop_emb, 
                                              sequence_length = sequence_len,
                                              initial_state_fw = self.initial_state_fw, 
                                              initial_state_bw = self.initial_state_bw)
        # Reshape the outputs into a single 2D Tensor
        # Shape [bsz * steps, hidden]
        # TODO: check
        out = (outs[0] + outs[1]) / 2
        outputs = tf.reshape(out, [-1, self.hidden])

        # Feed through final layer, compute logits
        logits = tf.matmul(outputs, self.softmax_w) + \
            self.softmax_b   # Shape [bsz * steps, vocab]
        return logits, f_states

    def loss(self):
        seq_loss = tf.nn.seq2seq.sequence_loss_by_example(
            [self.logits],
            [tf.reshape(self.Y, [-1])],
            [tf.ones([self.bsz * self.num_steps])])
        loss = tf.reduce_sum(seq_loss) / self.bsz
        return loss

    def train(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        return optimizer.minimize(self.loss_val)

    @staticmethod
    def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    @staticmethod
    def bias_variable(shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

def read(i):
    x = np.array(data[i]['text_v'][:-1], dtype=int)
    y = np.array(data[i]['text_v'][1:], dtype=int)
    choices = data[i]['choices_v']
    keys = data[i]['keys_v']
    return x, y, choices, keys

# Main Training Block
if __name__ == "__main__":
    with open('data', 'rb') as f:
        data = pickle.load(f)
    with open('vocab', 'rb') as f:
        vocab = pickle.load(f)

    # Launch Tensorflow Session
    print('Launching Tensorflow Session')
    with tf.Session() as sess:
        # Instantiate Model
        birnn = BiRNN(len(vocab), FLAGS.embedding_size, FLAGS.num_steps,
                            FLAGS.hidden_size, FLAGS.batch_size,
                            FLAGS.learning_rate)

        # Initialize all Variables
        sess.run(tf.initialize_all_variables())

        # Start Training
        ex_bsz, bsz, steps = FLAGS.batch_size * \
            FLAGS.num_steps, FLAGS.batch_size, FLAGS.num_steps
        for epoch in range(FLAGS.num_epochs):
            for i in range(NUM_ARTICLES):
                # Preprocess and vectorize the data
                x, y, _, keys = read(i)
                state_fw, state_bw = sess.run([birnn.initial_state_fw, birnn.initial_state_bw])
                loss, iters, start_time = 0., 0, time.time()
                # TODO: HACK
                keys_i = 0
                for start, end in zip(range(0, len(x) - ex_bsz, ex_bsz),
                                      range(ex_bsz, len(x), ex_bsz)):
                    # TODO: HACK
                    if y[start:end] == [vocab['BLANK']]:
                        y[start:end] = [keys[keys_i]]
                        keys_i += 1

                    # Build the Feed Dictionary, with inputs, outputs, dropout
                    # probability, and states.
                    feed_dict = {birnn.X: x[start:end].reshape(bsz, steps),
                                 birnn.Y: y[start:end].reshape(bsz, steps),
                                 birnn.keep_prob: FLAGS.dropout_prob,
                                 birnn.initial_state_fw[0]: state_fw[0],
                                 birnn.initial_state_fw[1]: state_fw[1],
                                 birnn.initial_state_bw[0]: state_bw[0],
                                 birnn.initial_state_bw[1]: state_bw[1]}

                    # Run the training operation with the Feed Dictionary,
                    # fetch loss and update state.
                    curr_loss, _, (state_fw, state_bw) = sess.run([
                        birnn.loss_val, birnn.train_op,
                        birnn.final_states], feed_dict=feed_dict)
                    # Update counters
                    loss, iters = loss + curr_loss, iters + steps

                    # Print Evaluation Statistics
                    if start % FLAGS.eval_every == 0:
                        print('Epoch {} Words {}>{} Perplexity: {}. {} seconds'
                              .format(epoch, start, end, np.exp(loss / iters),
                                      time.time() - start_time))
                        loss, iters = 0.0, 0
        # Evaluate Test Perplexity
        test_loss, test_iters, total_correct, total_blanks = 0., 0, 0., 0
        for i in range(NUM_ARTICLES):
            x, y, choices, keys = read(i)
            state_fw, state_bw = sess.run([birnn.initial_state_fw, birnn.initial_state_bw])
            blank_i = 0
            for s, e in zip(range(0, len(x - ex_bsz), ex_bsz),
                            range(ex_bsz, len(x), ex_bsz)):
                # TODO: BLANK
                # Build the Feed Dictionary, with inputs, outputs, dropout
                # probability, and states.
                feed_dict = {birnn.X: x[s:e].reshape(bsz, steps),
                             birnn.Y: y[s:e].reshape(bsz, steps),
                             birnn.keep_prob: FLAGS.dropout_prob,
                             birnn.initial_state_fw[0]: state_fw[0],
                             birnn.initial_state_fw[1]: state_fw[1],
                             birnn.initial_state_bw[0]: state_bw[0],
                             birnn.initial_state_bw[1]: state_bw[1]}

                # Fetch the loss, and final state
                logits, curr_loss, (state_fw, state_bw) = sess.run([
                    birnn.logits, birnn.loss_val, birnn.final_states],
                    feed_dict=feed_dict)
                if y[s:e][0] == vocab['BLANK']:
                    choices_d = {j: logits[0][j]
                                 for j in range(len(logits[0]))
                                 if j in choices[blank_i]}
                    if choices_d[keys[blank_i]] == max(choices_d.values()):
                        total_correct += 1
                    total_blanks += 1
                    blank_i += 1

                # Update counters
                test_loss += curr_loss
                test_iters += steps

        # Print Final Output
        print('Test Perplexity: {}'.format(np.exp(test_loss / test_iters)))
        print('Blank Accuracy: {}'.format(total_correct / total_blanks))
