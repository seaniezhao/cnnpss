import os
import fnmatch
import soundfile as sf
import pretty_midi

import data_util.textgrid as textgrid
from data_util.pinyin_phone import get_all_phon

all_phn = get_all_phon()


def cut_txt(label_path, dist_folder):

    file_name = label_path.split('/')[-1].split('.')[0] . replace('-phn', '')

    py_grid = textgrid.TextGrid.fromFile(label_path)

    source_tier = None
    for tier in py_grid.tiers:
        if tier.name == 'phoneme':  # 人工标注的
            source_tier = tier
            break
        elif tier.name == 'phones':  # Montreal-Forced-Aligner 标注
            source_tier = tier
            break

    assert source_tier

    segments = []
    seg_tier = None
    seg_start = -1
    for i, interval in enumerate(source_tier):
        start = interval.minTime
        end = interval.maxTime
        phn = interval.mark.strip()

        if seg_start < 0:
            seg_start = start
            seg_tier = textgrid.IntervalTier('phoneme')

        seg_tier.add(start, end, phn)

        assert (phn in all_phn)

        if phn not in ['br', 'pau']:
            pass
        else:
            if end - seg_start > 5:
                segments.append((seg_start, end, seg_tier))

                seg_start = end
                seg_tier = textgrid.IntervalTier('phoneme')

        #最后一片的处理
        if i == len(source_tier) - 1:
            if end - seg_start > 5:
                segments.append((seg_start, end, seg_tier))

                seg_start = end
                seg_tier = textgrid.IntervalTier('phoneme')
            else:
                # 加到前面
                p_start, p_end, p_seg_tier = segments[-1]
                for i, itval in enumerate(seg_tier):
                    p_seg_tier.addInterval(itval)
                segments[-1] = (p_start, end, p_seg_tier)

    # 保存切开的textgrid
    for i, seg in enumerate(segments):

        start, _, seg_tier = seg
        for interval in seg_tier:
            interval.minTime -= start
            interval.maxTime -= start
        seg_grid = textgrid.TextGrid()
        seg_grid.append(seg_tier)
        write_path = os.path.join(dist_folder, file_name+str(i)+'.TextGrid')
        seg_grid.write(write_path)

    return segments


def cut_midi(midi_path, segment, fname, dist_folder):

    start, stop, seg_tier = segment

    start, stop = round(start, 2), round(stop, 2)

    # new midi
    new_midi = pretty_midi.PrettyMIDI()
    new_instrument = pretty_midi.Instrument(program=1)

    midi_data = pretty_midi.PrettyMIDI(midi_path)
    assert len(midi_data.instruments) >= 1
    # 若多于一轨只取一轨
    instrument = midi_data.instruments[0]
    # print(instrument.name)
    for note in instrument.notes:
        #print(note.start, note.end)
        if round(note.start, 2) >= start-0.02 and round(note.end, 2) <= stop+0.02:
            new_instrument.notes.append(note)

    # ----
    print('check start stop: ', start, '||', stop)
    print('mid start stop', new_instrument.notes[0].start, '||', new_instrument.notes[-1].end)
    assert(abs(start - new_instrument.notes[0].start) <= 0.02 and
           abs(stop - new_instrument.notes[-1].end) <= 0.02)
    # -----

    for note in new_instrument.notes:
        note.start -= start
        if note.start <= 0.02 or note.start < 0:
            note.start = 0
        note.end -= start

    new_midi.instruments.append(new_instrument)
    # Write out the MIDI data
    write_path = os.path.join(dist_folder, file_name + str(i) + '.mid')
    new_midi.write(write_path)
    print("write midi:", write_path)


def cut_wav(wav_path, segment, fname, dist_folder):

    start, stop, seg_tier = segment

    y, sr = sf.read(wav_path)

    start = int(start*sr)
    stop = int(stop*sr)

    new_path = os.path.join(dist_folder, fname+'.wav')
    y_dist = y[start:stop]

    # 将pau 和 br都静音
    for interval in seg_tier:
        if interval.mark in ['br', 'pau']:
            i_start = int(interval.minTime*sr)
            i_end = int(interval.maxTime*sr)
            y_dist[i_start:i_end] = 0

    sf.write(new_path, y_dist, sr)


if __name__ == '__main__':
    # 功能:将原始的音频以及标注切割成一句一句的音频以及标注,并且保留音频与标注的对应关系

    # 切割音频以及标注后存放位置
    dist_folder = '/home/sean/pythonProj/data/xiaolongnv_cnnpss/raw_piece'
    if not os.path.exists(dist_folder):
        os.makedirs(dist_folder)

    # 原始音频以及标注存放位置
    uncut_folder = '/home/sean/pythonProj/data/xiaolongnv_cnnpss/raw'

    supportedExtensions = '*.wav'
    for dirpath, dirs, files in os.walk(uncut_folder):
        for file in fnmatch.filter(files, supportedExtensions):

            file_name = file.replace('.wav', '')

            print('processing '+file_name)
            raw_path = os.path.join(dirpath, file)
            txt_path = raw_path.replace('.wav', '-phn.TextGrid')
            midi_path = raw_path.replace('.wav', '.mid')

            segments = cut_txt(txt_path, dist_folder)

            for i, seg in enumerate(segments):
                fname = file_name+str(i)
                cut_wav(raw_path, seg, fname, dist_folder)
                cut_midi(midi_path, seg, fname, dist_folder)
