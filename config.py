import os

hop = 160  # sample_rate * 0.005 in witch world default_frame_period is 5 millisecond == 0.005 second
sample_rate = 32000
fft_size = 2048
f0_bin = 256
f0_max = 1100.0
f0_min = 50.0


ROOT_PATH = '/home/sean/pythonProj/data/xiaolongnv_cnnpss'
# ------------path config------------------
RAW_DATA_PATH = os.path.join(ROOT_PATH, 'raw_piece')
DATA_ROOT_PATH = os.path.join(ROOT_PATH, 'dataset')
GEN_PATH = os.path.join(ROOT_PATH, 'gen')


# 测试数据集的目录
TEST_ROOT_PATH = os.path.join(
    DATA_ROOT_PATH, 'test')  # 测试数据的根目录

TEST_SP_PATH = os.path.join(TEST_ROOT_PATH,  'sp')  # sp测试数据据集
TEST_AP_PATH = os.path.join(TEST_ROOT_PATH,  'ap')  # ap测试数据集
TEST_VUV_PATH = os.path.join(TEST_ROOT_PATH,  'vuv')  # vuv测试数据集
TEST_CONDITION_PATH = os.path.join(
    TEST_ROOT_PATH,  'condition')  # condition数据集
TEST_F0_PATH = os.path.join(TEST_ROOT_PATH,  'f0')  # f0数据集
TEST_F0_CONDITION_PATH = os.path.join(TEST_ROOT_PATH,  'f0_condition')  # f0条件

# 训练集的数据目录
TRAIN_ROOT_PATH = os.path.join(
    DATA_ROOT_PATH, 'train')  # 训练数据的根目录

TRAIN_SP_PATH = os.path.join(TRAIN_ROOT_PATH,  'sp')  # sp测试数据据集
TRAIN_AP_PATH = os.path.join(TRAIN_ROOT_PATH,  'ap')  # ap测试数据集
TRAIN_VUV_PATH = os.path.join(TRAIN_ROOT_PATH,  'vuv')  # vuv测试数据集
TRAIN_CONDITION_PATH = os.path.join(
    TRAIN_ROOT_PATH,  'condition')  # condition数据集
TRAIN_F0_PATH = os.path.join(TRAIN_ROOT_PATH,  'f0')  # f0数据集
TRAIN_F0_CONDITION_PATH = os.path.join(TRAIN_ROOT_PATH,  'f0_condition')  # f0条件

SNAOSHOTS_ROOT_PATH = os.path.join('.', 'snapshots')

# ------------path config------------------
