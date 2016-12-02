import pickle
from PyDictionary import PyDictionary

with open('../vocab', 'rb') as f:
    load_dict = pickle.load(f)
    vocab = list(load_dict.keys())

    dictionary = PyDictionary(vocab)
    sym = dictionary.getSynonyms()
    ant = dictionary.getAntonyms()

    symMap = {}
    antMap = {}

    for d in sym:
        if d is not None:
            symMap.update(d)

    for d in ant:
        if d is not None:
            antMap.update(d)

    with open('vocab_sym', 'wb') as sym_f:
        pickle.dump(sym, sym_f, 2)

    with open('vocab_ant', 'wb') as ant_f:
        pickle.dump(ant, ant_f, 2)
