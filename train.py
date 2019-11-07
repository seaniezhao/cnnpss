import hparams
from model.wavenet_model import *
from model.model_training import *
from config import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def exit_handler():
#     trainer.save_model()
#     print("exit from keyboard")

if not os.path.isdir(DATA_ROOT_PATH):
    print(u"请先执行data/preprocess.py生成训练数据 ")


def train_harmonic():
    snapshot_path = os.path.join(SNAOSHOTS_ROOT_PATH, 'harmonic')
    model = WaveNetModel(hparams.create_harmonic_hparams(), device).to(device)
    print('model: ', model)
    print('receptive field: ', model.receptive_field)
    print('parameter count: ', model.parameter_count())

    trainer = ModelTrainer(model=model,
                           data_folder=DATA_ROOT_PATH,
                           lr=0.0005,
                           weight_decay=0.000005,
                           snapshot_path=snapshot_path,
                           snapshot_name='harm',
                           device=device)

    print('start train harmonic...')
    trainer.train(batch_size=32,
                  epochs=1650)


def train_aperiodic():
    snapshot_path = os.path.join(SNAOSHOTS_ROOT_PATH, 'aperiodic')
    model = WaveNetModel(hparams.create_aperiodic_hparams(), device).to(device)
    print('model: ', model)
    print('receptive field: ', model.receptive_field)
    print('parameter count: ', model.parameter_count())

    trainer = ModelTrainer(model=model,
                           data_folder=DATA_ROOT_PATH,
                           lr=0.0005,
                           weight_decay=0.0,
                           snapshot_path=snapshot_path,
                           snapshot_name='aper',
                           device=device)

    print('start train aperiodic...')
    trainer.train(batch_size=32,
                  epochs=1650)


def train_vuv():
    snapshot_path = os.path.join(SNAOSHOTS_ROOT_PATH, 'vuv')
    model = WaveNetModel(hparams.create_vuv_hparams(), device).to(device)
    print('model: ', model)
    print('receptive field: ', model.receptive_field)
    print('parameter count: ', model.parameter_count())

    trainer = ModelTrainer(model=model,
                           data_folder=DATA_ROOT_PATH,
                           lr=0.0005,
                           weight_decay=0,
                           snapshot_path=snapshot_path,
                           snapshot_name='vuv',
                           device=device)

    print('start train vuv...')
    trainer.train(batch_size=32,
                  epochs=1650)


def train_f0():
    snapshot_path = os.path.join(SNAOSHOTS_ROOT_PATH, 'f0')
    model = WaveNetModel(hparams.create_f0_hparams(), device).to(device)
    print('model: ', model)
    print('receptive field: ', model.receptive_field)
    print('parameter count: ', model.parameter_count())

    trainer = ModelTrainer(model=model,
                           data_folder=DATA_ROOT_PATH,
                           lr=0.001,
                           weight_decay=0,
                           snapshot_path=snapshot_path,
                           snapshot_name='f0',
                           device=device)

    print('start train f0...')
    trainer.train(batch_size=64,
                  epochs=235)


if __name__ == '__main__':
    train_harmonic()
    #train_aperiodic()
    train_vuv()
    train_f0()
