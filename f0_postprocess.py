import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from data_util.pinyin_phone import get_phn_class
import copy
from pylab import rcParams
rcParams['figure.figsize'] = 20, 5


def tuning_postprocessing(note_list, time_phon_list, p_f0):

    # transfer f0 to semitones
    # plt.plot(predicted_f0)
    # plt.show()
    predicted_f0 = copy.copy(p_f0)
    predicted_f0[predicted_f0 > 0] = 69 + 12 * np.log2(predicted_f0[predicted_f0 > 0] / 440)

    midi_f0 = np.zeros(predicted_f0.shape)

    frame_phn = []
    for item in time_phon_list:
        pstart = item[0]
        pend = item[1]
        phn = item[2]
        plen = pend - pstart
        for i in range(plen):
            frame_phn.append(phn)

    for note in note_list:
        start = note[0]
        end = note[1]
        pitch = note[2]
        if pitch != 0:
            midi_f0[start:end] = pitch
            piece = predicted_f0[start:end]
            piece_len = len(piece)
            wes = signal.tukey(piece_len)

            if piece_len < 11:
                wds = 1
            else:
                delta_f0s = piece - signal.savgol_filter(piece, 11, 3)
                wds = []
                for df0 in delta_f0s:
                    wd = 1/np.min([1+27*np.abs(df0), 15])
                    wds.append(wd)

            wts = []
            for f0 in piece:
                abs_diff = np.abs(f0-pitch)
                if abs_diff <= 1:
                    wts.append(1)
                else:
                    wts.append(1/abs_diff)
            wps = []
            for i, f0 in enumerate(piece):
                wps.append(get_phn_class(frame_phn[start+i]))

            ws = wes * wds * wps * wts

            _f0 = sum(ws*piece)/sum(ws)

            if sum(ws) != 0:
                piece = piece+(pitch-_f0)
                predicted_f0[start:end] = piece
        else:
            predicted_f0[start:end] = 0


    # smooth
    filtered = gaussian_filter1d(predicted_f0, 1)
    filtered_f0 = np.power(2, (filtered - 69) / 12) * 440
    # plt.plot(filtered_f0)
    # plt.show()

    midi_f0 = np.power(2, (midi_f0 - 69) / 12) * 440

    return filtered_f0, midi_f0
