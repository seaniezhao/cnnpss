from evaluate import load_model
import numpy as np
from config import *
from data_util.pinyin_phone import get_all_phon
from data_util.sp_code import decode_harmonic
import pyworld as pw


class MachineSinger:

    def __init__(self):
        self.sp_model = load_model(0, '//')
        self.ap_model = load_model(0, '//')
        self.vuv_model = load_model(0, '//')
        self.f0_model = load_model(0, '//')

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

        decode_sp = decode_harmonic(sample, fft_size)

        return decode_sp, raw_gen

    def generate_ap(self, condition, cat_input):

        raw_gen = self.ap_model.generate(condition, cat_input)
        sample = (raw_gen.transpose(0, 1).cpu().numpy().astype(np.double) + 0.5) * \
                 (self.ap_max - self.ap_min) + self.ap_min

        decode_ap = pw.decode_spectral_envelope(sample, sample_rate, fft_size)

        return decode_ap, raw_gen

    def generate_vuv(self, condition, cat_input):

        gen = self.vuv_model.generate(condition, cat_input).squeeze()
        return gen.cpu().numpy().astype(np.uint8)

    def generate_f0(self, condition):

        gen = self.f0_model.generate(condition, None).squeeze()
        f0 = gen.cpu().numpy() * self.f0_max

        return f0
