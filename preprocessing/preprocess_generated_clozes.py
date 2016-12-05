import pickle

CLOZE_DIR = '../generated_clozes/'
PREFIXES = list(map(str, range(1, 16)))


result = []

with open('../vocab', 'rb') as f:
    vocab = pickle.load(f)

for p in PREFIXES:
    p = CLOZE_DIR + p
    entry = {}

    with open(p + '_a', 'rb') as f:
        print(p)
        body = pickle.load(f)
        body += ['STOP'] * 20 # Padding to match batch_size

        entry['text_v'] = list(map(lambda x: vocab.get(x, vocab['UNK']), body))

    with open(p + '_b', 'rb') as f:
        choices = pickle.load(f) # list of list of strings
        entry['choices_v'] = list(map(lambda x: list(map(lambda c: vocab.get(c, vocab['UNK']), x)), choices))

    with open(p + '_c', 'rb') as f:
        answers = pickle.load(f)

        entry['keys_v'] = list(map(lambda x: vocab.get(x, vocab['UNK']), answers))

    num_of_blanks = entry['text_v'].count(vocab.get('BLANK'))
    print(num_of_blanks)
    print(len(entry['choices_v']))

    assert num_of_blanks == len(entry['choices_v']) # number of blanks == number of questions
    assert len(entry['choices_v']) == len(entry['keys_v']) # number of answers == number of questions

    result.append(entry)

print(len(result))
assert len(result) == 15


with open('../more_generated_clozes', 'wb') as f:
    pickle.dump(result, f, 2)