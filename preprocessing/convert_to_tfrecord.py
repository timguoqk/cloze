import pickle
import tensorflow as tf


def convert_clozes():
    def _list64(l):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=l))
    BLANK = vocab['BLANK']
    STOP = vocab['BLANK']

    with open('clozes', 'rb') as f:
        clozes = pickle.load(f)
    writer = tf.python_io.TFRecordWriter('./clozes.tfrecords')
    for article in clozes:
        blank_loc = [i for i, x in enumerate(article['text_v']) if x == BLANK]
        for num_b, b in enumerate(blank_loc):
            stop_prev = next(i for i in range(b - 1, -2, -1)
                             if i == -1 or article['text_v'][i] == STOP)
            stop_next = next(i for i in range(b + 1, len(article['text_v']) + 1)
                             if i == len(article['text_v']) or article['text_v'][i] == STOP)
            example = tf.train.Example(features=tf.train.Features(feature={
                'document': _list64(article['text_v']),
                'query': _list64(article['text_v'][stop_prev:stop_next]),
                'answer': _list64([article['keys_v'][num_b]]),
                'choices': _list64(article['choices_v'][num_b])
            }))
            writer.write(example.SerializeToString())
    writer.close()


def convert_books():
    # filenames = [f for f in os.listdir(BOOKS_DIR)
    #              if os.path.isfile(BOOKS_DIR + f)]
    # res = ''
    # for fname in filenames:
    #     with open(BOOKS_DIR + fname) as f:
    #         res += proces_text(f.read()) + ' STOP '
    # vec_res = [vocab.get(word, vocab['UNK']) for word in res.split()]
    # with open('books_training', 'wb') as f:
    #     pickle.dump(vec_res, f, 2)
    pass


if __name__ == '__main__':
    with open('vocab', 'rb') as f:
        vocab = pickle.load(f)

    convert_clozes()
    # convert_books()
