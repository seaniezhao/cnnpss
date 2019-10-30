import pretty_midi
import pyworld as pw
import soundfile as sf
import data_util.textgrid as textgrid
import os
import fnmatch
import pickle
from collections import Counter

pkl_file = open('Biaobei_pinyin-phoneme.pkl', 'rb')
pp_dict = pickle.load(pkl_file)
all_pinyin = []
sheng_mu = []
pp_dict_reverse = {}
for pinyin in pp_dict:
    pp_dict_reverse[str(pp_dict[pinyin])] = pinyin
    if pinyin not in all_pinyin:
        all_pinyin.append(pinyin)
    if len(pp_dict[pinyin]) > 1:
        sheng_mu.append(pp_dict[pinyin][0])

sheng_mu = set(sheng_mu)



hop = 160
sample_rate = 32000

def get_f0(path):
    y, sr = sf.read(path)

    _f0, t = pw.dio(y, sr, f0_floor=50, f0_ceil=1100, frame_period=pw.default_frame_period)
    #_f0 = pw.stonemask(y, _f0, t, sr)


    return _f0


def process_phon_label(text_grid_path):
    py_grid = textgrid.TextGrid.fromFile(label_path)

    source_tier = None
    for tier in py_grid.tiers:
        if tier.name == 'phoneme':
            source_tier = tier
    assert source_tier != None

    time_phon_list = []
    phon_list = []
    for i, interval in enumerate(source_tier):
        phn = interval.mark

        tup = (int(round(interval.minTime*sample_rate/hop)), int(round(interval.maxTime*sample_rate/hop)), phn)
        #print(tup)
        time_phon_list.append(tup)
        if phn not in phon_list:
            phon_list.append(phn)

    last = list(time_phon_list[-1])
    last[1] += 1
    time_phon_list[-1] = tuple(last)

    return time_phon_list

# 名字长吧哈哈哈
def time_phon_list_to_time_pinyin_list(time_phon_list):
    time_pinyin_list = []
    temp_list = []
    temp_start = 0
    for time_phon in time_phon_list:
        start = time_phon[0]
        end = time_phon[1]
        phn = time_phon[2].strip()

        if phn in sheng_mu:
            temp_list.append(phn)
            temp_start = start
            continue
        elif len(temp_list) > 0:
            temp_list.append(phn)
            pinyin = pp_dict_reverse[str(temp_list)]
            temp_list.clear()
            time_pinyin_list.append((temp_start, end, pinyin))
        else:
            if str([phn]) in pp_dict_reverse:
                pinyin = pp_dict_reverse[str([phn])]
            else:
                pinyin = phn
                if pinyin not in all_pinyin:
                    all_pinyin.append(pinyin)
            time_pinyin_list.append((start, end, pinyin))

    for i in time_pinyin_list:
        print(i)
    return time_pinyin_list, all_pinyin


def create_midi(f0, note_time, file_path):

    sing = pretty_midi.PrettyMIDI()
    paino = pretty_midi.Instrument(program=1)

    for note in note_time:
        start_time = note[0]/sample_rate*hop
        end_time = note[1]/sample_rate*hop
        phn = note[2]
        if phn == 'pau' or phn == 'br' or phn.strip() == '#' or not phn:
            note = pretty_midi.Note(velocity=100, pitch=0, start=start_time, end=end_time)
            paino.notes.append(note)
            continue

        f0_start = note[0]
        f0_end = note[1]
        pitch_collection = []
        sample_num = f0_end - f0_start + 1

        for i in range(sample_num):
            indx = i + f0_start

            if indx > len(f0) - 1:
                continue

            current_hertz = f0[indx]
            if current_hertz > 0:
                cpitchf = pretty_midi.hz_to_note_number(current_hertz)
                pitch_collection.append(int(round(cpitchf)))
                #print(cpitchf)


        print('--------------------------------------------------------------')
        pitch = 0
        if len(pitch_collection)>0:
            pitch_count = Counter(pitch_collection)
            pitch_mc = pitch_count.most_common(1)
            pitch = pitch_mc[0][0]
        note = pretty_midi.Note(velocity=100, pitch=pitch, start=start_time, end=end_time)
        print(note.start, note.end, note.pitch)
        # Add it to our cello instrument
        paino.notes.append(note)

    sing.instruments.append(paino)
    # Write out the MIDI data
    sing.write(file_path)
    print("write midi:", file_path)



if __name__ == '__main__':
    # 使用 world提取f0 根据标注 最终得到wav的对应midi文件

    raw_folder = 'create_midi'
    supportedExtensions = '*.wav'
    for dirpath, dirs, files in os.walk(raw_folder):
        for file in fnmatch.filter(files, supportedExtensions):
            print('!!!!!file:', file)
            file_name = file.replace('.wav', '')
            wav_path = os.path.join(dirpath, file)
            label_path = wav_path.replace('.wav', '-phn.TextGrid')
            f0 = get_f0(wav_path)
            time_phon_list = process_phon_label(label_path)
            note_times,_ =  time_phon_list_to_time_pinyin_list(time_phon_list)
            file_path =  wav_path.replace('.wav', '.mid')
            create_midi(f0, note_times, file_path)

