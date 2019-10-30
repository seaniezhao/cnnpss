from config import *
import fnmatch
import sys
from tqdm import tqdm
import numpy as np
from data_util.pinyin_phone import get_all_phon
from data_util.data_tools import process_wav, process_phon_label, make_timbre_model_condition, make_f0_condition
from data_util.midi_util import get_midi_notes
import random

def prepare_directory():
    to_prepares = [TRAIN_SP_PATH, TRAIN_AP_PATH, TRAIN_VUV_PATH, TRAIN_CONDITION_PATH, TRAIN_F0_PATH,
                   TEST_SP_PATH, TEST_AP_PATH, TEST_VUV_PATH, TEST_CONDITION_PATH, TEST_F0_PATH,
                   TRAIN_F0_CONDITION_PATH, TEST_F0_CONDITION_PATH]

    for p in to_prepares:
        if not os.path.exists(p):
            os.makedirs(p)


def main():
    sp_min, sp_max = sys.maxsize, (-sys.maxsize - 1)
    ap_min, ap_max = sys.maxsize, (-sys.maxsize - 1)

    supported_extensions = '*.wav'
    wav_files = fnmatch.filter(os.listdir(RAW_DATA_PATH), supported_extensions)

    # 为了获取sp ap最大值最小值, 先暂存然后在处理成条件
    data_to_save = []

    for file in wav_files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        print('processing: ', file_name)
        wav_path = os.path.join(RAW_DATA_PATH,  file)
        midi_path = os.path.join(RAW_DATA_PATH, file_name + '.mid')
        # 默认是 phn.TextGrid
        txt_path = os.path.join(RAW_DATA_PATH,   file_name + '-phn.TextGrid')

        if not os.path.isfile(txt_path):
            txt_path = os.path.join(RAW_DATA_PATH,   file_name + '.TextGrid')

        if not os.path.isfile(txt_path):
            txt_path = os.path.join(RAW_DATA_PATH,   file_name + '_pinyin.TextGrid')

        if not os.path.isfile(txt_path):
            print("[Warning]   no found the TextGrid of " + wav_path)
            continue

        time_phon_list, _ = process_phon_label(txt_path)
        note_list = get_midi_notes(midi_path)
        f0, _sp, code_sp, _ap, code_ap = process_wav(wav_path)
        v_uv = f0 > 0

        data_to_save.append(
            (file_name, time_phon_list, note_list, f0, code_sp, code_ap, v_uv))

        _sp_min = np.min(code_sp)
        _sp_max = np.max(code_sp)

        sp_min = _sp_min if _sp_min < sp_min else sp_min
        sp_max = _sp_max if _sp_max > sp_max else sp_max

        _ap_min = np.min(code_ap)
        _ap_max = np.max(code_ap)

        ap_min = _ap_min if _ap_min < ap_min else ap_min
        ap_max = _ap_max if _ap_max > ap_max else ap_max

        #break

    np.save(os.path.join(DATA_ROOT_PATH, 'min_max_record.npy'),
            [sp_min, sp_max, ap_min, ap_max])

    # 为了更好的优化模型, 还是手动控制测试集比较好
    # test_names = ['45shexi', '20xinqiang', '10jindalaisimida']

    total_count = 0
    test_count = 0
    for (file_name, time_phon_list, note_list, f0, code_sp, code_ap, v_uv) in tqdm(data_to_save):
        total_count += 1
        timbre_condi = make_timbre_model_condition(time_phon_list, f0)
        f0_condi = make_f0_condition(note_list, time_phon_list, len(f0))

        code_sp = (code_sp - sp_min) / (sp_max - sp_min) - 0.5
        code_ap = (code_ap - ap_min) / (ap_max - ap_min) - 0.5

        f0 = f0 / f0_max
        test = random.random() > 0.8

        if test:
            test_count += 1
            np.save(TEST_CONDITION_PATH + '/' + file_name + '_condi.npy', timbre_condi)
            np.save(TEST_SP_PATH + '/' + file_name + '_sp.npy', code_sp)
            np.save(TEST_AP_PATH + '/' + file_name + '_ap.npy', code_ap)
            np.save(TEST_VUV_PATH + '/' + file_name + '_vuv.npy', v_uv)
            np.save(TEST_F0_PATH + '/' + file_name + '_f0.npy', f0)
            np.save(TEST_F0_CONDITION_PATH + '/' + file_name + '_f0_condi.npy', f0_condi)
        else:
            np.save(TRAIN_CONDITION_PATH + '/' + file_name + '_condi.npy', timbre_condi)
            np.save(TRAIN_SP_PATH + '/' + file_name + '_sp.npy', code_sp)
            np.save(TRAIN_AP_PATH + '/' + file_name + '_ap.npy', code_ap)
            np.save(TRAIN_VUV_PATH + '/' + file_name + '_vuv.npy', v_uv)
            np.save(TRAIN_F0_PATH + '/' + file_name + '_f0.npy', f0)
            np.save(TRAIN_F0_CONDITION_PATH + '/' + file_name + '_f0_condi.npy', f0_condi)

    print('total count: ', total_count, ', test count: ', test_count)


if __name__ == '__main__':
   prepare_directory()
   main()

