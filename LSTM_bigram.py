from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import tensorflow as tf
import time
import pickle

FLAGS = tf.app.flags.FLAGS
NUM_CLOZES = 128

# Model Parameters
tf.app.flags.DEFINE_integer(
    'num_steps', 1, 'Number of unrolled steps before backprop.')
tf.app.flags.DEFINE_integer('embedding_size', 50, 'Size of the Embeddings.')
tf.app.flags.DEFINE_integer('hidden_size', 256, 'Size of the LSTM Layer.')

# Training Parameters
tf.app.flags.DEFINE_integer('num_epochs', 2, 'Number of Training Epochs.')
tf.app.flags.DEFINE_integer(
    'batch_size', 20, 'Size of a batch (for training).')
tf.app.flags.DEFINE_float('learning_rate', 1e-4,
                          'Learning rate for Adam Optimizer.')
tf.app.flags.DEFINE_float(
    'dropout_prob', 0.5, 'Keep probability, for dropout.')
tf.app.flags.DEFINE_integer(
    'eval_every', 10000, 'Print statistics every eval_every words.')


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary('stddev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)


class BiRNN():
    def __init__(self, vocab_size, embedding_size, num_steps, hidden_size,
                 batch_size, learning_rate):
        """
        Instantiate an RNNLangmod Model, with the necessary hyperparameters.

        :param vocab_size: Size of the vocabulary.
        :param num_steps: Number of words to feed into LSTM before performing
        a gradient update.
        :param hidden_size: Size of the LSTM Layer.
        :param num_layers: Number of stacked LSTM Layers in the model.
        :param batch_size: Batch size (for training).
        :param learning_rate: Learning rate for Adam Optimizer
        """
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

        # Summaries for Tensorboard
        self.summaries = tf.merge_all_summaries()

    def instantiate_weights(self):
        # Embedding Matrix
        self.E = self.weight_variable(
            [self.vocab_size, self.embedding_size], 'Embedding')

        # LSTM Cells
        self.fw_cell = rnn_cell.LSTMCell(self.hidden)
        self.bw_cell = rnn_cell.LSTMCell(self.hidden)

        self.initial_state_fw = self.fw_cell.zero_state(self.bsz, tf.float32)
        self.initial_state_bw = self.bw_cell.zero_state(self.bsz, tf.float32)

        # Softmax Output
        self.softmax_w = self.weight_variable(
            [self.hidden * 2, self.vocab_size], 'Softmax_Weight')
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
        outs, f_states = rnn.bidirectional_dynamic_rnn(
            self.fw_cell, self.bw_cell, drop_emb,
            sequence_length=sequence_len,
            initial_state_fw=self.initial_state_fw,
            initial_state_bw=self.initial_state_bw)
        # Reshape the outputs into a single 2D Tensor
        # Shape [bsz * steps, 2 * hidden]
        outputs = tf.reshape(tf.concat(0, outs), [-1, 2 * self.hidden])

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
        tf.scalar_summary('loss', loss)
        return loss

    def train(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        return optimizer.minimize(self.loss_val)

    @staticmethod
    def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        var = tf.Variable(initial, name=name)
        variable_summaries(var, name)
        return var

    @staticmethod
    def bias_variable(shape, name):
        initial = tf.constant(0.1, shape=shape)
        var = tf.Variable(initial, name=name)
        variable_summaries(var, name)
        return var


def read_cloze(i):
    x = np.array(clozes_data[i]['text_v'][:-1], dtype=int)
    y = np.array(clozes_data[i]['text_v'][1:], dtype=int)
    choices = clozes_data[i]['choices_v']
    keys = clozes_data[i]['keys_v']
    return x, y, choices, keys


def read_training():
    with open('books_training', 'rb') as f:
        books_training = pickle.load(f)
    x = np.array(books_training[:-1])
    y = np.array(books_training[1:])
    return x, y


test_step_counter = 0


def test(birnn, sess, saved_trace=False):
    global test_step_counter
    # Evaluate Test Perplexity
    test_loss, test_iters, total_correct, total_blanks = 0., 0, 0., 0
    d = {}  # error test
    for i in range(NUM_CLOZES):
        x, y, choices, keys = read_cloze(i)
        state_fw, state_bw = sess.run(
            [birnn.initial_state_fw, birnn.initial_state_bw])
        blank_i = 0
        for s, e in zip(range(0, len(x - ex_bsz), ex_bsz),
                        range(ex_bsz, len(x), ex_bsz)):
            # Build the Feed Dictionary, with inputs, outputs, dropout
            # probability, and states.
            feed_dict = {birnn.X: x[s:e].reshape(bsz, steps),
                         birnn.Y: y[s:e].reshape(bsz, steps),
                         birnn.keep_prob: FLAGS.dropout_prob,
                         birnn.initial_state_fw[0]: state_fw[0],
                         birnn.initial_state_fw[1]: state_fw[1],
                         birnn.initial_state_bw[0]: state_bw[0],
                         birnn.initial_state_bw[1]: state_bw[1]}
            # Fetch the logits, loss, and final state
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            logits, curr_loss, (state_fw, state_bw), summaries = sess.run([
                birnn.logits, birnn.loss_val, birnn.final_states,
                birnn.summaries],
                feed_dict=feed_dict,
                options=run_options,
                run_metadata=run_metadata)
            test_writer.add_summary(summaries)

            for batch in range(bsz):
                if y[s:e][batch] == vocab['BLANK']:
                    choices_d = {j: logits[batch][j]
                                 for j in range(len(logits[batch]))
                                 if j in choices[blank_i]}
                    if saved_trace:
                        d[(i, blank_i)] = {"logits": logits[batch],
                                           "choices": choices_d,
                                           "key": keys[blank_i],
                                           "correct": False}
                    if choices_d[keys[blank_i]] == max(choices_d.values()):
                        total_correct += 1
                        if saved_trace:
                            d[(i, blank_i)]["correct"] = True
                    total_blanks += 1
                    blank_i += 1

            # Update counters
            test_iters += steps
            test_loss += curr_loss

    if saved_trace:
        with open('error_analysis', 'wb') as f:
            pickle.dump(d, f)
    # Print Final Output
    print('Test Perplexity: {}'.format(np.exp(test_loss / test_iters)))
    print('Blank Accuracy: {}'.format(total_correct / total_blanks))


# Global variables
ex_bsz, bsz, steps = FLAGS.batch_size * \
    FLAGS.num_steps, FLAGS.batch_size, FLAGS.num_steps
with open('clozes', 'rb') as f:
    clozes_data = pickle.load(f)
with open('vocab', 'rb') as f:
    vocab = pickle.load(f)

# Main Training Block
if __name__ == "__main__":
    # Launch Tensorflow Session
    print('Launching Tensorflow Session')
    with tf.Session() as sess:
        # Instantiate Model
        birnn = BiRNN(len(vocab), FLAGS.embedding_size, FLAGS.num_steps,
                      FLAGS.hidden_size, FLAGS.batch_size,
                      FLAGS.learning_rate)

        # Tensorboard writers
        train_writer = tf.train.SummaryWriter('./tensorboard/train',
                                              sess.graph)
        test_writer = tf.train.SummaryWriter('./tensorboard/test', sess.graph)

        # Initialize all Variables
        sess.run(tf.initialize_all_variables())

        # Start Training
        x, y = read_training()
        for epoch in range(FLAGS.num_epochs):
            # Preprocess and vectorize the data
            state_fw, state_bw = sess.run(
                [birnn.initial_state_fw, birnn.initial_state_bw])
            loss, iters, start_time = 0., 0, time.time()

            for start, end in zip(range(0, len(x) - ex_bsz, ex_bsz),
                                  range(ex_bsz, len(x), ex_bsz)):

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
                if start % 200 == 0:
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    curr_loss, _, (state_fw, state_bw), summaries = sess.run([
                        birnn.loss_val, birnn.train_op,
                        birnn.final_states, birnn.summaries],
                        feed_dict=feed_dict,
                        options=run_options,
                        run_metadata=run_metadata)

                    train_writer.add_summary(summaries)
                else:
                    curr_loss, _, (state_fw, state_bw) = sess.run([
                        birnn.loss_val, birnn.train_op,
                        birnn.final_states],
                        feed_dict=feed_dict)
                # Update counters
                loss, iters = loss + curr_loss, iters + steps

                # Print Evaluation Statistics
                if start % FLAGS.eval_every == 0:
                    print('Epoch {} Words {}>{} Perplexity: {}. {} seconds'
                          .format(epoch, start, end, np.exp(loss / iters),
                                  time.time() - start_time))
                    loss, iters = 0.0, 0
                    test(birnn, sess)
        test(birnn, sess, saved_trace=True)

        train_writer.close()
        test_writer.close()
