import os
import re
from random import randrange

BOOKS_DIR = '../books/'

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



MIN_INTERVAL = 10

BLANK = '=====BLANK====='
KEYWORDS = ['STOP', 'NUM', 'CHINESE', 'EXCLAMATION', 'QUESTION', 'UNK']


def isTxt(name):
    withExtension = name.split('.')
    return len(withExtension) == 2 and withExtension[1] == 'txt'


filenames = [BOOKS_DIR + f for f in os.listdir(BOOKS_DIR)
             if os.path.isfile(BOOKS_DIR + f) and isTxt(f)]

for filename in filenames:
    with open(filename) as f:
        index = MIN_INTERVAL
        answers = []

        body = proces_text(f.read()).split()
        length = len(body)

        # print(body[:100])

        repeat = 0
        while index < len(body):
            random_index = randrange(index - MIN_INTERVAL, index)
            if body[random_index] not in KEYWORDS:
                answers.append(body[random_index])
                body[random_index] = BLANK
                index += MIN_INTERVAL
            elif repeat < 3:
                repeat += 1
                continue

            index += MIN_INTERVAL
            repeat = 0

        # print('\n')
        print(body[:100])
        print(answers[:10])

    print('\n\n\n')
