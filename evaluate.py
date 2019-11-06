from model.wavenet_model import *
import hparams
import numpy as np
from config import *
from data_util.sp_code import decode_harmonic
import pyworld as pw
import matplotlib.pyplot as plt
import soundfile as sf
from data_util.data_tools import process_phon_label, make_timbre_model_condition
from data_util.midi_util import get_midi_notes
from f0_postprocess import tuning_postprocessing


def load_model(mtype, state_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if mtype == 0:
        hparam = hparams.create_harmonic_hparams()
    elif mtype == 1:
        hparam = hparams.create_aperiodic_hparams()
    elif mtype == 2:
        hparam = hparams.create_vuv_hparams()
    else:
        hparam = hparams.create_f0_hparams()

    model = WaveNetModel(hparam, device).to(device)
    states = torch.load(state_path)
    model.load_state_dict(states['state_dict'])

    return model


model_dict = {}

def load_latest_model_from(mtype, location):

    files = [location + "/" + f for f in os.listdir(location)]
    newest_file = max(files, key=os.path.getctime)

    print("load model " + newest_file)
    model = None
    if mtype not in model_dict:
        model = load_model(mtype, newest_file)
        model_dict[mtype] = model
    else:
        model = model_dict[mtype]

    return model


def load_timbre(path, m_type, mx, mn):
    load_t = np.load(path).astype(np.double)

    load_t = (load_t + 0.5) * (mx - mn) + mn
    decode_sp = decode_harmonic(load_t, fft_size)
    if m_type == 1:
        decode_sp = pw.decode_aperiodicity(load_t, 32000, fft_size)

    return decode_sp


#  type 0:harmonic, 1:aperiodic,
def generate_timbre(m_type, mx, mn, condition, cat_input=None):
    model_path = os.path.join(SNAOSHOTS_ROOT_PATH, 'harmonic')
    if m_type == 1:
        model_path = os.path.join(SNAOSHOTS_ROOT_PATH, 'aperiodic')

    model = load_latest_model_from(m_type, model_path)
    raw_gen = model.generate(condition, cat_input)
    sample = (raw_gen.transpose(0, 1).cpu().numpy().astype(np.double) + 0.5) * (mx - mn) + mn

    decode_sp = None
    if m_type == 0:
        decode_sp = decode_harmonic(sample, fft_size)
    elif m_type == 1:
        decode_sp = pw.decode_aperiodicity(sample, 32000, fft_size)

    return decode_sp, raw_gen


def generate_vuv(condition, cat_input):
    model_path = os.path.join(SNAOSHOTS_ROOT_PATH,  'vuv')

    model = load_latest_model_from(2, model_path)
    gen = model.generate(condition, cat_input).squeeze()

    return gen.cpu().numpy().astype(np.uint8)


def generate_f0(condition, mx):
    model_path = os.path.join(SNAOSHOTS_ROOT_PATH,  'f0')

    model = load_latest_model_from(3, model_path)
    gen = model.generate(condition, None).squeeze()

    f0 = gen.cpu().numpy()
    return f0.astype(np.double) * mx


def get_condition(file_name,  isTestDir=True):
    c_path = os.path.join(DATA_ROOT_PATH,   'test' if isTestDir else 'train', 'condition', file_name + '_condi.npy')
    conditon = np.load(c_path).astype(np.float)
    return torch.Tensor(conditon).transpose(0, 1)


def make_condition(file_name, f0):
    txt_path = os.path.join(RAW_DATA_PATH, file_name + '.TextGrid')
    time_phon_list, _ = process_phon_label(txt_path)
    f0 = f0 / f0_max
    condition = make_timbre_model_condition(time_phon_list, f0)
    return torch.Tensor(condition).transpose(0, 1)


def get_f0_condition(file_name,  isTestDir=True):
    c_path = os.path.join(DATA_ROOT_PATH,   'test' if isTestDir else 'train', 'f0_condition', file_name + '_f0_condi.npy')
    f0_condition = np.load(c_path).astype(np.float)
    return torch.Tensor(f0_condition).transpose(0, 1)


def f0_post_process(file_name, f0):
    txt_path = os.path.join(RAW_DATA_PATH, file_name + '.TextGrid')
    midi_path = os.path.join(RAW_DATA_PATH, file_name + '.mid')
    time_phon_list, _ = process_phon_label(txt_path)
    note_list = get_midi_notes(midi_path)
    post_f0, midi_f0 = tuning_postprocessing(note_list, time_phon_list, f0)
    return post_f0, midi_f0


# 批量合成音乐
def make_test_wav(file_name, isTestDir=True, flag="flag"):

    [sp_min, sp_max, ap_min, ap_max] = np.load(os.path.join(DATA_ROOT_PATH, 'min_max_record.npy'))

    f0_condi = get_f0_condition(file_name, isTestDir)
    f0 = generate_f0(f0_condi, f0_max)

    post_f0, midi_f0 = f0_post_process(file_name, f0)

    c_path = os.path.join(DATA_ROOT_PATH, 'test' if isTestDir else 'train', 'f0', file_name + '_f0.npy')
    origin_f0 = np.load(c_path).astype(np.double) * f0_max

    plt.title(file_name+' f0')
    # plt.plot(f0, color='red')
    # plt.plot(post_f0, color='blue')
    plt.plot(origin_f0, color='green')
    plt.plot(midi_f0)
    plt.show()
    pass
    # --------------------------------------------------------------------

    # # condi = get_condition(file_name, isTestDir)
    # condi = make_condition(file_name, f0)
    # sp, raw_sp = generate_timbre(0, sp_max, sp_min, condi, None)
    #
    # # plt.imshow(np.log(np.transpose(sp)), aspect='auto', origin='bottom', interpolation='none')
    # # plt.show()
    #
    # ap, raw_ap = generate_timbre(1, ap_max, ap_min, condi, raw_sp)
    #
    # # plt.imshow(np.log(np.transpose(ap)), aspect='auto', origin='bottom', interpolation='none')
    # # plt.show()
    #
    # gen_cat = torch.cat((raw_ap, raw_sp), 0)
    #
    # vuv = generate_vuv(condi, gen_cat)
    # # plt.plot(vuv)
    # # plt.show()
    #
    # synthesized = pw.synthesize(f0*vuv, sp, ap, sample_rate, pw.default_frame_period)
    # save_dir = os.path.join(GEN_PATH, flag)
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # sf.write(save_dir + '/' + file_name + '.wav', synthesized, sample_rate)


if __name__ == '__main__':
    condi_names = os.listdir(os.path.join(DATA_ROOT_PATH, 'test/condition'))
    file_name_list = []
    for item in condi_names:
        item = item.replace('_condi.npy', '')
        file_name_list.append(item)
    isTestDir = True

    # 用于区别不同时间生成的
    flag = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    for file_name in file_name_list:
        make_test_wav(file_name, isTestDir=isTestDir, flag=flag)