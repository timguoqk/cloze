import re
import pickle

FILE_NAMES = ['NewConcepts', 'T1', 'T2']
vocab = {}

words = set()
for filename in FILE_NAMES:
    with open(filename) as f:
        raw = f.read()
        text = re.sub(r'[^a-zA-Z ]', ' ', raw).lower()
        words |= set(text.split())

d = dict(map(lambda t: t[::-1], enumerate(words)))
KEYWORDS = ['STOP', 'NUM', 'BLANK', 'CHINESE', 'EXCLAMATION', 'QUESTION']
for kw in KEYWORDS:
    d[kw] = len(d)


with open('vocab', 'wb') as f:
    pickle.dump(d, f, 2)
