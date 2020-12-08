from evaluate import load_model
import numpy as np
from config import *
from data_util.pinyin_phone import get_all_phon, get_pinyins
from data_util.sp_code import decode_harmonic
from data_util.data_tools import make_f0_condition, make_timbre_model_condition
from data_util.midi_util import preprocess_midi, make_phn_from_midi
from f0_postprocess import tuning_postprocessing
import pyworld as pw
import torch
import soundfile as sf

class MachineSinger:

    def __init__(self):
        self.sp_model = load_model(0, os.path.join(SNAOSHOTS_ROOT_PATH, 'harmonic/harm_1649_2019-11-07_17-17-14'))
        self.ap_model = load_model(1, os.path.join(SNAOSHOTS_ROOT_PATH, 'aperiodic/aper_1649_2019-11-07_22-18-03'))
        self.vuv_model = load_model(2, os.path.join(SNAOSHOTS_ROOT_PATH, 'vuv/vuv_1649_2019-11-08_03-02-47'))
        self.f0_model = load_model(3, os.path.join(SNAOSHOTS_ROOT_PATH, 'f0/f0_234_2019-11-08_13-16-45'))

        [sp_min, sp_max, ap_min, ap_max] = np.load(os.path.join(DATA_ROOT_PATH, 'min_max_record.npy'))
        self.sp_min = sp_min
        self.sp_max = sp_max
        self.ap_min = ap_min
        self.ap_max = ap_max
        self.f0_max = f0_max
        self.all_phn = get_all_phon()

    def generate_sp(self, condition):

        raw_gen = self.sp_model.generate(condition, None)
        sample = (raw_gen.transpose(0, 1).cpu().numpy().astype(np.double) + 0.5) * \
                 (self.sp_max - self.sp_min) + self.sp_min
        sample = sample.copy(order='C')
        decode_sp = decode_harmonic(sample, fft_size)

        return decode_sp, raw_gen

    def generate_ap(self, condition, cat_input):

        raw_gen = self.ap_model.generate(condition, cat_input)
        sample = (raw_gen.transpose(0, 1).cpu().numpy().astype(np.double) + 0.5) * \
                 (self.ap_max - self.ap_min) + self.ap_min

        sample = np.ascontiguousarray(sample)
        decode_ap = pw.decode_aperiodicity(sample, sample_rate, fft_size)

        return decode_ap, raw_gen

    def generate_vuv(self, condition, cat_input):

        gen = self.vuv_model.generate(condition, cat_input).squeeze()
        return gen.cpu().numpy().astype(np.uint8)

    def generate_f0(self, f0_condition):

        gen = self.f0_model.generate(f0_condition, None).squeeze()
        f0 = gen.cpu().numpy() * self.f0_max

        return f0

    def sing(self, midi_path, lyrics):
        pinyins = get_pinyins(lyrics)
        note_list = preprocess_midi(midi_path)
        time_phon_list = make_phn_from_midi(note_list,pinyins)

        f0_condition = make_f0_condition(note_list, time_phon_list)
        f0_gen = self.generate_f0(torch.Tensor(f0_condition).transpose(0, 1))
        post_f0, midi_f0 = tuning_postprocessing(note_list, time_phon_list, f0_gen)

        condi = make_timbre_model_condition(time_phon_list, post_f0)
        condi_tensor = torch.Tensor(condi).transpose(0, 1)
        sp, raw_sp = self.generate_sp(condi_tensor)
        ap, raw_ap = self.generate_ap(condi_tensor, raw_sp)
        gen_cat = torch.cat((raw_ap, raw_sp), 0)
        vuv = self.generate_vuv(condi_tensor, gen_cat)

        post_f0 = post_f0.astype(np.double)
        synthesized = pw.synthesize(post_f0 * vuv, sp, ap, sample_rate, pw.default_frame_period)
        save_dir = os.path.join(GEN_PATH, 'inference')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = midi_path.split('/')[-1].split('.')[0]
        sf.write(save_dir + '/' + file_name + '.wav', synthesized, sample_rate)



if __name__ == '__main__':
    singer = MachineSinger()
    singer.sing('/home/sean/pythonProj/data/xiaolongnv_cnnpss/raw_piece/3xiaohongmao3.mid', '嘲笑誰恃美揚威沒了心如何相配')

