import unidecode
import numpy as np
import json
import ctypes
from random import randint

def bp_algorithm(cond: bool, out: float):
    return (0.501 if cond else 0.5) - out

def ff_algorithm(phrase: str):
    decoded = decode_str(phrase)
    return [np.longdouble(decoded / ((len(str(decoded)) * 10) - 1))]

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

def decode_str(str_: str):
    out = np.longdouble(0)
    for wrd in str_:
        for chr in wrd:
            chr_ord = ord(chr)
            out = (out * (10 ** len(str(chr_ord))) + ord(chr))
    return out