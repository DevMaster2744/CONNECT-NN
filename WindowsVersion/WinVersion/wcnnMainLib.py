import unidecode
import numpy as np
import json
import ctypes
from math import log
from random import randint
import hashlib

def cross_entropy(true: np.float64, pred: np.float64):
    redundancy = 1e-15

    pred += redundancy

    true = true if true > 0 else redundancy
    pred = pred if pred > 0 else redundancy
    '''
    true1 = (1 - true) if (1 - true) > 0 else (1 - redundancy)
    pred1 = (1 - pred) if (1 - pred) > 0 else (1 - redundancy)
    '''
    #return np.add(np.multiply(true, np.log(pred)), np.multiply(true1, np.log(pred1)))
    return -np.multiply(true, np.log(pred))

def bp_algorithm(true: np.longdouble, out: float):
    return cross_entropy(true, out)

def ff_algorithm(phrase: str):
    decoded = decode_str(phrase)
    return decoded

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

def prevent_shutdown():
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

def allow_shutdown():
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

def read_train_data():
    with open("train_list.json") as f:
        data = json.load(f)
    return data

train_data = read_train_data()

def select_random_train_data(_train_data):
    phrases_len = len(_train_data["phrases"])

    rand_int = randint(0, phrases_len - 1)

    data = _train_data["phrases"][rand_int]
    return unidecode.unidecode(data["phrase"]), data["isBad"], rand_int

def mdi(x):
    for i in range(x):
        div = i + 1
        if x % div == 0:
            return div

def decode_str(str_: str):
    '''out = np.long(0)
    for wrd in str_:
        chr_ord = str(ord(wrd))
        out *= 10 * (len(chr_ord) + 1)
        out += int(chr_ord)

    out_mdi = np.long(mdi(out))

    out /= out_mdi'''
    return np.longlong(hashlib.sha256(unidecode.unidecode(str_.replace(' ', '')).encode()).hexdigest())