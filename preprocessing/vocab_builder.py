import re
import pickle
import csv

DIR = 'books/vocab/'
FILE_NAMES = ['NewConcepts', 'T1', 'T2', 'CST4']

words = set()
for filename in FILE_NAMES:
    with open(DIR + filename) as f:
        raw = f.read()
        text = re.sub(r'[^a-zA-Z ]', ' ', raw).lower()
        words |= set(text.split())

vocab = dict(map(lambda t: t[::-1], enumerate(words)))
KEYWORDS = ['STOP', 'NUM', 'BLANK', 'CHINESE', 'EXCLAMATION', 'QUESTION',
            'UNK']
for kw in KEYWORDS:
    vocab[kw] = len(vocab)

with open('vocab', 'wb') as f:
    pickle.dump(vocab, f, 2)

with open('vocab.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Word', 'Key']) # Header
    for key, value in sorted(vocab.items(), key=lambda t: t[1]):
        writer.writerow([key, value])
