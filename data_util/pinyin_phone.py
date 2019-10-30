import pandas as pd
import numpy as np
import pickle
import os


cdir = os.path.dirname(__file__)
pkl_file = open(os.path.join(cdir, 'pinyin-phoneme.pkl'), 'rb')
pp_dict = pickle.load(pkl_file)


def get_phoneme(pinyin):

    if pinyin in pp_dict:
        phns = []
        ph = pp_dict[pinyin]
        for p in ph:
            phns.append(p)
        return phns
    else:
        return [pinyin]


def Is_pinyin(pinyin):
    if pinyin in pp_dict:
        return True
    return False


# 不发声,元音,辅音
def get_phn_class(phn):
    if phn.strip() in ['pau', 'sli', 'br', 'sp', '#']:
        return 0
    elif phn in ['k', 'p', 's', 'h', 't', 'j', 'c', 'b', 'z', 'm', 'g', 'l',
                 'd', 'ch', 'zh', 'x', 'q', 'sh', 'f', 'n', 'r', 'pl']:
        return 1
    else:
        return 2


def get_all_phon():
    all_phon = ['none', 'pau', 'br', 'sil']
    for k, v in pp_dict.items():
        for phn in v:
            if phn not in all_phon:
                all_phon.append(phn)

    return all_phon


if __name__ == '__main__':
    print(get_phoneme('shei'))


