import librosa
import pyworld as pw
import numpy as np
from data_util.pinyin_phone import get_phn_class, get_all_phon
import soundfile as sf

import data_util.textgrid as textgrid
from data_util.sp_code import code_harmonic, decode_harmonic
from config import *

all_phn = get_all_phon()


def process_wav(wav_path):
    y, osr = sf.read(wav_path)

    if len(y.shape) > 1:
        y = np.ascontiguousarray(y[:, 0])

    sr = sample_rate
    if osr != sr:
        y = librosa.resample(y, osr, sr)

    sf.write(wav_path, y, sample_rate)

    # 使用dio算法计算音频的基频F0
    _f0, t = pw.dio(y, sr, f0_floor=f0_min, f0_ceil=f0_max,
                        frame_period=pw.default_frame_period)
    _f0 = pw.stonemask(y, _f0, t, sr)
    _f0[_f0 > f0_max] = f0_max

    print(_f0.shape)

    # 使用CheapTrick算法计算音频的频谱包络
    _sp = pw.cheaptrick(y, _f0, t, sr)

    code_sp = code_harmonic(_sp, 60)
    print(_sp.shape, code_sp.shape)
    # 计算aperiodic参数
    _ap = pw.d4c(y, _f0, t, sr)

    code_ap = pw.code_aperiodicity(_ap, sr)
    print(_ap.shape, code_ap.shape)

    return _f0, _sp, code_sp, _ap, code_ap


def process_phon_label(label_path):

    py_grid = textgrid.TextGrid.fromFile(label_path)

    source_tier = None
    for tier in py_grid.tiers:
        if tier.name == 'phoneme':  # 人工标注的
            source_tier = tier
            break
        elif tier.name == 'phones':  # Montreal-Forced-Aligner 标注
            source_tier = tier
            break

    assert source_tier != None

    time_phon_list = []
    phon_list = []
    for i, interval in enumerate(source_tier):
        phn = interval.mark.strip()

        tup = (int(round(interval.minTime*sample_rate/hop)),
               int(round(interval.maxTime*sample_rate/hop)), phn)

        assert (phn in all_phn)

        time_phon_list.append(tup)
        if phn not in phon_list:
            phon_list.append(phn)

    return time_phon_list, phon_list


def make_timbre_model_condition(time_phon_list, f0):
    all_phon = get_all_phon()


    f0_mel = 1127*np.log(1+f0/700)
    f0_mel_min = 1127*np.log(1 + f0_min/700)
    f0_mel_max = 1127*np.log(1 + f0_max/700)

    # f0_mel[f0_mel == 0] = 0
    # 大于0的分为256个箱
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * \
        (f0_bin-2)/(f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel < 0] = 1
    f0_mel[f0_mel > f0_bin-1] = f0_bin-1

    f0_coarse = np.rint(f0_mel).astype(np.int)
    print('Max f0: ', np.max(f0_coarse), ' ||Min f0: ', np.min(f0_coarse))
    assert(np.max(f0_coarse) <= 256 and np.min(f0_coarse) >= 0)

    if time_phon_list[-1][1] != len(f0):
        tup = time_phon_list[-1]
        tup_list = list(tup)
        tup_list[1] = len(f0)
        time_phon_list[-1] = tuple(tup_list)
    assert(time_phon_list[-1][1] == len(f0))

    label_list = []
    oh_list = []
    for i in range(len(f0)):
        pre_phn, cur_phn, next_phn, pos_in_phon = (0, 0, 0, 0)
        for j in range(len(time_phon_list)):
            if time_phon_list[j][0] <= i < time_phon_list[j][1]:
                cur_phn = all_phon.index(time_phon_list[j][2])
                if j == 0:
                    pre_phn = all_phon.index('none')
                else:
                    pre_phn = all_phon.index(time_phon_list[j - 1][2])

                if j == len(time_phon_list) - 1:
                    next_phn = all_phon.index('none')
                else:
                    next_phn = all_phon.index(time_phon_list[j + 1][2])

                begin = time_phon_list[j][0]
                end = time_phon_list[j][1]
                width = end - begin  # [begin, end)

                # 正常语速1分钟200个字
                if width < 100:
                    fpos = int(width / 2)
                    spos = fpos
                else:
                    fpos = 50
                    spos = width-50

                pos = i-begin
                if pos < fpos:
                    pos_in_phon = pos+1
                elif fpos <= pos < spos:
                    pos_in_phon = 0
                else:
                    pos_in_phon = pos-spos+(100-(width-spos)+1)

        #log(begin, end, width, pos,pos_in_phon)
        label_list.append([pre_phn, cur_phn, next_phn,
                           pos_in_phon, f0_coarse[i]])

        # onehot
        pre_phn_oh = np.zeros(len(all_phon))
        cur_phn_oh = np.zeros(len(all_phon))
        next_phn_oh = np.zeros(len(all_phon))
        pos_in_phon_oh = np.zeros(101)
        f0_coarse_oh = np.zeros(f0_bin)

        pre_phn_oh[pre_phn] = 1
        cur_phn_oh[cur_phn] = 1
        next_phn_oh[next_phn] = 1
        pos_in_phon_oh[pos_in_phon] = 1
        f0_coarse_oh[f0_coarse[i]] = 1

        oh_list.append(
            np.concatenate((pre_phn_oh, cur_phn_oh, next_phn_oh, pos_in_phon_oh, f0_coarse_oh)).astype(np.int8))
        if i == len(f0) - 1:
            print('timbre condition:', len(oh_list[-1]), ' ', np.sum(oh_list[-1]))

    return oh_list


def make_f0_condition(note_list, time_phon_list, frame_num=None):
    all_phon = get_all_phon()

    def bin_note_dur(odur):
        margin = 2 * sample_rate / hop
        # dur 分箱 0算一类 2秒以上算一类, 其余的分16类(2*8分音符)
        bin_dur = 0
        if margin > odur > 0:
            bin_dur = int(round((odur / margin) * 16))
        elif odur > margin:
            bin_dur = 17
        return bin_dur

    if 'none' not in all_phon:
        all_phon.append('none')

    if not frame_num:
        frame_num = note_list[-1][1]
    # 临时方法处理一下精度造成的问题
    else:
        last_phn = list(time_phon_list[-1])
        if last_phn[1] != frame_num:
            print('last_phn[1] != frame_num: ', last_phn[1], '!=', frame_num)
            last_phn[1] = frame_num
        time_phon_list[-1] = tuple(last_phn)

        last_note = list(note_list[-1])
        if last_note[1] != frame_num:
            print('last_note[1] != frame_num: ', last_note[1], '!=', frame_num)
            last_note[1] = frame_num
        note_list[-1] = tuple(last_note)

    assert(note_list[-1][1] == time_phon_list[-1][1])

    label_list = []
    oh_list = []
    for i in range(frame_num):
        # find current phn
        pre_phn, cur_phn, next_phn, pos_in_phon = (0, 0, 0, 0)
        for j in range(len(time_phon_list)):
            if time_phon_list[j][0] <= i < time_phon_list[j][1]:
                cur_phn = all_phon.index(time_phon_list[j][2])
                if j == 0:
                    pre_phn = all_phon.index('none')
                else:
                    pre_phn = all_phon.index(time_phon_list[j - 1][2])

                if j == len(time_phon_list) - 1:
                    next_phn = all_phon.index('none')
                else:
                    next_phn = all_phon.index(time_phon_list[j + 1][2])

                begin = time_phon_list[j][0]
                end = time_phon_list[j][1]
                width = end - begin  # [begin, end)

                # 正常语速1分钟200个字
                # 这里是吧一个phn的宽度 分成三分,其中两个切分点分别是fpos和spos
                if width < 100:
                    fpos = int(width / 2)
                    spos = fpos
                else:
                    fpos = 50
                    spos = width - 50

                pos = i - begin
                if pos < fpos:
                    pos_in_phon = 0
                elif fpos <= pos < spos:
                    pos_in_phon = 1
                else:
                    pos_in_phon = 2

        pre_note, cur_note, next_note, pre_dur, cur_dur, next_dur, pos_in_note = (0, 0, 0, 0, 0, 0, 0)
        # find current note
        for k in range(len(note_list)):
            if note_list[k][0] <= i < note_list[k][1]:
                cur_note = note_list[k][2]
                cur_dur = bin_note_dur(note_list[k][3])

                if k == 0:
                    pre_note = 0
                    pre_dur = 0
                else:
                    pre_note = note_list[k-1][2]
                    pre_dur = bin_note_dur(note_list[k-1][3])

                if k == len(note_list) - 1:
                    next_note = 0
                    next_dur = 0
                else:
                    next_note = note_list[k+1][2]
                    next_dur = bin_note_dur(note_list[k+1][3])

                begin = note_list[k][0]
                end = note_list[k][1]
                width = end - begin  # [begin, end)

                if width < 500:
                    fpos = int(width / 2)
                    spos = fpos
                else:
                    fpos = 250
                    spos = width - 250

                pos = i - begin
                if pos < fpos:
                    pos_in_note = 0
                elif fpos <= pos < spos:
                    pos_in_note = 1
                else:
                    pos_in_note = 2

        label_list.append([pre_phn, cur_phn, next_phn, pos_in_phon,
                           pre_note, cur_note, next_note, pre_dur, cur_dur, next_dur, pos_in_note])

        #print(label_list[-1])

        # onehot
        pre_phn_oh = np.zeros(len(all_phon))
        cur_phn_oh = np.zeros(len(all_phon))
        next_phn_oh = np.zeros(len(all_phon))
        pos_in_phon_oh = np.zeros(3)

        pre_note_oh = np.zeros(128)
        cur_note_oh = np.zeros(128)
        next_note_oh = np.zeros(128)
        pre_dur_oh = np.zeros(18)
        cur_dur_oh = np.zeros(18)
        next_dur_oh = np.zeros(18)
        pos_in_note_oh = np.zeros(3)

        pre_phn_oh[pre_phn] = 1
        cur_phn_oh[cur_phn] = 1
        next_phn_oh[next_phn] = 1
        pos_in_phon_oh[pos_in_phon] = 1

        pre_note_oh[pre_note] = 1
        cur_note_oh[cur_note] = 1
        next_note_oh[next_note] = 1
        pre_dur_oh[pre_dur] = 1
        cur_dur_oh[cur_dur] = 1
        next_dur_oh[next_dur] = 1
        pos_in_note_oh[pos_in_note] = 1

        oh_list.append(
            np.concatenate((pre_phn_oh, cur_phn_oh, next_phn_oh, pos_in_phon_oh,
                            pre_note_oh, cur_note_oh, next_note_oh, pre_dur_oh, cur_dur_oh, next_dur_oh,
                            pos_in_note_oh)).astype(np.int8))

        if i == frame_num - 1:
            print('f0 condition:', len(oh_list[-1]), ' ', np.sum(oh_list[-1]))
    return oh_list
