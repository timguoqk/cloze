import os
import re
import pickle

CLOZES_DIR = 'cloze_data/'
BOOKS_DIR = 'books/'
PREFIXES = list(map(str, range(1, 128)))
# PREFIXES = ['24']

RE0 = re.compile(r'\[\w+\]')  # num
RE1 = re.compile(r'\n\n+')
RE2 = re.compile(r'a\(n\)')
RE3 = re.compile(r' \d+ ')
RE4 = re.compile(r'\([^a-zA-Z]+\)')
RE5 = re.compile(r'!')
RE6 = re.compile(r'\?')
RE7 = re.compile('[\(.,:;\)“…”，\'\n]')
RE8 = re.compile(r'  +')
RE9 = re.compile(r'(STOP ){2,}')
RE10 = re.compile(r'\d+')


def proces_text(text):
    text = text.lower().replace('’', '')
    text = RE0.sub('NUM', text)
    text = RE1.sub(' STOP ', text)
    text = RE2.sub('a', text)
    text = RE3.sub(' BLANK ', text)
    text = text.replace('-', ' ')
    text = RE10.sub('NUM', text)
    text = RE4.sub(' CHINESE', text)
    text = RE5.sub(' EXCLAMATION STOP ', text)
    text = RE6.sub(' QUESTION STOP ', text)
    text = RE7.sub(' STOP ', text)
    text = RE8.sub(' ', text)
    text = RE9.sub('STOP ', text)
    return text


def preprocess_clozes():
    result = []
    RE_CHOICES = re.compile(r'[A-D]\. ((?:[\w\-\'’]+ )+)')
    for p in PREFIXES:
        p = CLOZES_DIR + p
        entry = {
            'text': '',
            'choices': [],
            'keys': []
        }
        with open(p + '_a') as f:
            entry['text'] = proces_text(f.read())
            assert entry['text'].count('BLANK') == 20

        with open(p + '_b') as f:
            for line in f.readlines():
                line = line[line.index('A'):].strip() + ' '
                splitted = RE_CHOICES.split(line)
                entry['choices'].append(
                    [x.lower().strip() for i, x in enumerate(splitted)
                     if i % 2 == 1])
                assert len(entry['choices'][-1]) == 4
            assert len(entry['choices']) == 20

        with open(p + '_c') as f:
            keys = list(f.read().strip())
            entry['keys'] = [entry['choices'][i][ord(k) - ord('A')]
                             for i, k in enumerate(keys)]
            assert len(entry['keys']) == 20
        result.append(entry)

    for entry in result:
        entry['text_v'] = [vocab.get(word, vocab['UNK'])
                           for word in entry['text'].split()]
        entry['choices_v'] = [
            [vocab.get(word.split()[0], vocab['UNK']) for word in choices]
            for choices in entry['choices']
        ]
        entry['keys_v'] = [vocab.get(word.split()[0], vocab['UNK'])
                           for word in entry['keys']]
    with open('clozes', 'wb') as f:
        pickle.dump(result, f, 2)


def preprocess_books():
    filenames = [f for f in os.listdir(BOOKS_DIR)
                 if os.path.isfile(BOOKS_DIR + f)]
    res = ''
    for fname in filenames:
        with open(BOOKS_DIR + fname) as f:
            res += proces_text(f.read()) + ' STOP '
    vec_res = [vocab.get(word, vocab['UNK']) for word in res.split()]
    with open('books_training', 'wb') as f:
        pickle.dump(vec_res, f, 2)


if __name__ == '__main__':
    with open('vocab', 'rb') as f:
        vocab = pickle.load(f)

    preprocess_clozes()
    preprocess_books()
