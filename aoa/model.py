import os
import time
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.ops import sparse_ops
from util import softmax, orthogonal_initializer

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('vocab_size', 14061, 'Vocabulary size')
flags.DEFINE_integer('embedding_size', 384, 'Embedding dimension')
flags.DEFINE_integer('hidden_size', 256, 'Hidden units')
flags.DEFINE_integer('batch_size', 20, 'Batch size')
flags.DEFINE_integer('epochs', 2, 'Number of epochs to train/test')
flags.DEFINE_boolean('training', True, 'Training or testing a model')
flags.DEFINE_string(
    'name', 'aoa', 'Model name (used for statistics and model path')
flags.DEFINE_float('dropout_keep_prob', 0.7, 'Keep prob for embedding dropout')
flags.DEFINE_float('l2_reg', 0.0001, 'l2 regularization for embeddings')

model_path = 'models/' + FLAGS.name

if not os.path.exists(model_path):
    os.makedirs(model_path)


def read_records(index=0):
    train_queue = tf.train.string_input_producer(
        ['clozes.tfrecords'], num_epochs=FLAGS.epochs)
    validation_queue = tf.train.string_input_producer(
        ['validation.tfrecords'], num_epochs=FLAGS.epochs)
    test_queue = tf.train.string_input_producer(
        ['test.tfrecords'], num_epochs=FLAGS.epochs)

    queue = tf.QueueBase.from_list(
        index, [train_queue, validation_queue, test_queue])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'document': tf.VarLenFeature(tf.int64),
            'query': tf.VarLenFeature(tf.int64),
            'answer': tf.FixedLenFeature([], tf.int64),
            'choices': tf.FixedLenFeature([4], tf.int64)
        })

    document = sparse_ops.serialize_sparse(features['document'])
    query = sparse_ops.serialize_sparse(features['query'])
    answer = features['answer']
    choices = features['choices']

    document_batch_serialized, query_batch_serialized, answer_batch, choices_batch = tf.train.shuffle_batch(
        [document, query, answer, choices],
        batch_size=FLAGS.batch_size,
        capacity=2000,
        min_after_dequeue=1000)

    sparse_document_batch = sparse_ops.deserialize_many_sparse(
        document_batch_serialized, dtype=tf.int64)
    sparse_query_batch = sparse_ops.deserialize_many_sparse(
        query_batch_serialized, dtype=tf.int64)

    document_batch = tf.sparse_tensor_to_dense(sparse_document_batch)
    document_weights = tf.sparse_to_dense(
        sparse_document_batch.indices, sparse_document_batch.shape, 1)

    query_batch = tf.sparse_tensor_to_dense(sparse_query_batch)
    query_weights = tf.sparse_to_dense(
        sparse_query_batch.indices, sparse_query_batch.shape, 1)

    return document_batch, document_weights, query_batch, query_weights, answer_batch, choices_batch


def inference(documents, doc_mask, query, query_mask):
    embedding = tf.get_variable('embedding',
                                [FLAGS.vocab_size, FLAGS.embedding_size],
                                initializer=tf.random_uniform_initializer(minval=-0.05, maxval=0.05))

    regularizer = tf.nn.l2_loss(embedding)

    doc_emb = tf.nn.dropout(tf.nn.embedding_lookup(
        embedding, documents), FLAGS.dropout_keep_prob)
    doc_emb.set_shape([None, None, FLAGS.embedding_size])

    query_emb = tf.nn.dropout(tf.nn.embedding_lookup(
        embedding, query), FLAGS.dropout_keep_prob)
    query_emb.set_shape([None, None, FLAGS.embedding_size])

    with tf.variable_scope('document', initializer=orthogonal_initializer()):
        fwd_cell = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size)
        back_cell = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size)

        doc_len = tf.reduce_sum(doc_mask, reduction_indices=1)
        h, _ = tf.nn.bidirectional_dynamic_rnn(
            fwd_cell, back_cell, doc_emb,
            sequence_length=tf.to_int64(doc_len),
            dtype=tf.float32)
        h_doc = tf.concat(2, h)

    with tf.variable_scope('query', initializer=orthogonal_initializer()):
        fwd_cell = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size)
        back_cell = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size)

        query_len = tf.reduce_sum(query_mask, reduction_indices=1)
        h, _ = tf.nn.bidirectional_dynamic_rnn(
            fwd_cell, back_cell, query_emb,
            sequence_length=tf.to_int64(query_len),
            dtype=tf.float32)
        h_query = tf.concat(2, h)

    M = tf.batch_matmul(h_doc, h_query, adj_y=True)
    M_mask = tf.to_float(tf.batch_matmul(tf.expand_dims(
        doc_mask, -1), tf.expand_dims(query_mask, 1)))

    alpha = softmax(M, 1, M_mask)
    beta = softmax(M, 2, M_mask)

    query_importance = tf.expand_dims(tf.reduce_sum(
        beta, 1) / tf.to_float(tf.expand_dims(doc_len, -1)), -1)

    s = tf.squeeze(tf.batch_matmul(alpha, query_importance), [2])

    unpacked_s = zip(tf.unpack(s, FLAGS.batch_size),
                     tf.unpack(documents, FLAGS.batch_size))
    y_hat = tf.pack([
        tf.unsorted_segment_sum(attentions, sentence_ids, FLAGS.vocab_size)
        for (attentions, sentence_ids) in unpacked_s
    ])

    return y_hat, regularizer


def train(y_hat, regularizer, document, doc_weight, answer, choices):
    # Equivalent to: relevant = tf.gather_nd(y_hat, index)
    index_base = tf.range(0, FLAGS.batch_size) * FLAGS.vocab_size
    index = index_base + tf.to_int32(answer)
    flat = tf.reshape(y_hat, [-1])
    relevant = tf.gather(flat, index)

    flat_choices = tf.reshape(choices, [-1])
    choices_index = tf.reshape(tf.tile(tf.reshape(index_base, [-1, 1]), [1, 4]), [-1]) + tf.to_int32(flat_choices)
    relevant_c = tf.gather(flat, choices_index)
    relevant_c_reshaped = tf.reshape(relevant_c, [FLAGS.batch_size, 4])
    num_arg_max = tf.argmax(relevant_c_reshaped, 1)
    index_nam = tf.range(0, FLAGS.batch_size) * 4 + tf.to_int32(num_arg_max)
    indices = tf.gather(flat_choices, index_nam)

    # mean cause reg is independent of batch size
    loss = -tf.reduce_mean(tf.log(relevant + 1e-10)) + FLAGS.l2_reg * regularizer
    # loss = -tf.reduce_mean(tf.log(relevant + 1e-16)) + tf.reduce_mean(tf.log(relevant_c + 1e-16)) / 4 + FLAGS.l2_reg * regularizer

    global_step = tf.Variable(0, name="global_step", trainable=False)

    accuracy = tf.reduce_mean(tf.to_float(
        tf.equal(indices, answer)))

    optimizer = tf.train.AdamOptimizer()
    grads_and_vars = optimizer.compute_gradients(loss)
    capped_grads_and_vars = [(tf.clip_by_value(grad, -5, 5), var)
                             for (grad, var) in grads_and_vars]
    train_op = optimizer.apply_gradients(
        capped_grads_and_vars, global_step=global_step)

    tf.scalar_summary('loss', loss)
    tf.scalar_summary('accuracy', accuracy)
    return loss, train_op, global_step, accuracy


def main():
    dataset = tf.placeholder_with_default(0, [])
    document_batch, document_weights, query_batch, query_weights, answer_batch, choices_batch = read_records(
        dataset)

    y_hat, reg = inference(
        document_batch, document_weights, query_batch, query_weights)
    loss, train_op, global_step, accuracy = train(
        y_hat, reg, document_batch, document_weights, answer_batch, choices_batch)
    summary_op = tf.merge_all_summaries()

    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter(model_path, sess.graph)
        # Visualize embeddings
        saver = tf.train.Saver()
        config = projector.ProjectorConfig()
        # You can add multiple embeddings. Here we add only one.
        embedding = config.embeddings.add()
        embedding.tensor_name = 'embedding'
        embedding.metadata_path = './vocab.csv'

        # Saves a configuration file that TensorBoard will read during startup.
        projector.visualize_embeddings(summary_writer, config)

        saver_variables = tf.all_variables()
        if not FLAGS.training:
            not_included = ['input_producer/limit_epochs/epochs:0',
                            'smooth_acc:0', 'avg_acc:0']
            saver_variables = list(filter(
                lambda var: var.name not in not_included, saver_variables))
        saver = tf.train.Saver(saver_variables)

        sess.run([
            tf.initialize_all_variables(),
            tf.initialize_local_variables()])
        model = tf.train.latest_checkpoint(model_path + '/model.ckpt')
        if model:
            print('Restoring ' + model)
            saver.restore(sess, model)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        start_time = time.time()
        accumulated_accuracy = 0
        try:
            if FLAGS.training:
                while not coord.should_stop():
                    loss_t, _, step, acc = sess.run(
                        [loss, train_op, global_step, accuracy],
                        feed_dict={dataset: 0})
                    elapsed_time, start_time = time.time() - start_time, time.time()
                    print(step, loss_t, acc, elapsed_time)
                    if step % 3 == 0:
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, step)
                    if step % 50 == 1:
                        saver.save(sess, model_path + '/model.ckpt', global_step=step)
            else:
                step = 0
                while not coord.should_stop():
                    acc = sess.run(accuracy, feed_dict={dataset: 2})
                    step += 1
                    accumulated_accuracy += (acc - accumulated_accuracy) / step
                    elapsed_time, start_time = time.time() - start_time, time.time()
                    print(accumulated_accuracy, acc, elapsed_time)
        except tf.errors.OutOfRangeError:
            print('Done!')
        finally:
            coord.request_stop()
        coord.join(threads)
        saver.save(sess, model_path + '/model.ckpt', global_step=step)
        summary_writer.close()


if __name__ == "__main__":
    main()
