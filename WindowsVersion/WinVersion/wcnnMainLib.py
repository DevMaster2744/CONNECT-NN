import unidecode
import numpy as np
import json
import ctypes
from math import log
from random import randint
import hashlib
from ast import literal_eval

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

def getRspDataTables():
    with open("train_list.json") as f:
        data_json = json.load(f)["phrases"] # Get the phrases from the data JSON

    raw_phrases = []
    raw_cond = []
    for data in data_json:
        raw_phrases.append(data["phrase"])
        raw_cond.append(int(data["isBad"]))

    return raw_phrases, raw_cond

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
    base_str = unidecode.unidecode_expect_ascii(str_).lower()
    print(base_str)

    out = np.array([0 for _ in range(50)])

    for i, let in enumerate(base_str):
        out[i] = ord(let)

    return out