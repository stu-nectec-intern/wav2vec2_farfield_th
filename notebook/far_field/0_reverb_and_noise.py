import os
import pandas as pd
import torch
import torchaudio
import torchaudio.functional as F
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor

path_room = '/home/putsahaw/VUT_FIT_Q301/MicID01/'
path_exhibition = '/home/putsahaw/VUT_FIT_D105/MicID01/'

rir_dict_room = {
    'room_dis1':path_room + "SpkID01_20170910_T/09/RIR/IR_sweep_15s_45Hzto22kHz_FS16kHz.v00.wav",
    'room_dis2':path_room + "SpkID01_20170910_T/16/RIR/IR_sweep_15s_45Hzto22kHz_FS16kHz.v00.wav",
    'room_dis3':path_room + "SpkID05_20170915_S/30/RIR/IR_sweep_15s_45Hzto22kHz_FS16kHz.v00.wav",
    'room_dis4':path_room + "SpkID05_20170915_S/29/RIR/IR_sweep_15s_45Hzto22kHz_FS16kHz.v00.wav"
}

rir_dict_exhibition = {
    'exhibition_dis1':path_exhibition + "SpkID02_20170901_S/19/RIR/IR_sweep_15s_45Hzto22kHz_FS16kHz.v00.wav",
    'exhibition_dis2':path_exhibition + "SpkID07_20170904_T/22/RIR/IR_sweep_15s_45Hzto22kHz_FS16kHz.v00.wav",
    'exhibition_dis3':path_exhibition + "SpkID07_20170904_T/21/RIR/IR_sweep_15s_45Hzto22kHz_FS16kHz.v00.wav",
    'exhibition_dis4':path_exhibition + "SpkID02_20170901_S/22/RIR/IR_sweep_15s_45Hzto22kHz_FS16kHz.v00.wav"
}

snr_dict = {
    "snr_3db": torch.tensor([3]),
    "snr_5db": torch.tensor([5]),
    "snr_10db": torch.tensor([10])
}

class NoiseGenerator:
    def __init__(self, noise_path):
        self.noises, _ = torchaudio.load(noise_path)
        self.start = 0

    def get_slice(self, length):
        if self.start + length > self.noises.shape[1]:
            end = self.start + length - self.noises.shape[1]
            slice = torch.cat([self.noises[:, self.start:], self.noises[:, :end]], dim=1)
            self.start = end
        else:
            slice = self.noises[:, self.start:self.start+length]
            self.start += length
        return slice
    
def add_rir_noise(row, rir_path, snr_db, noise_gen, rir_name):
    waveform, sample_rate = torchaudio.load(row['path'])
    rir_raw, _ = torchaudio.load(rir_path)
    augmented = torchaudio.functional.convolve(waveform, rir_raw)
    noises = noise_gen.get_slice(augmented.shape[1])  
    bg_added = torchaudio.functional.add_noise(augmented, noises, snr_db)
    _, filename = os.path.split(row['path'])
    base, ext = os.path.splitext(filename)
    new_base = base + "_" + rir_name
    new_directory = "wavs_rir_added"

    new_path = os.path.join(new_directory, new_base + ext)

    torchaudio.save(filepath=new_path, src=bg_added, sample_rate=sample_rate)
    return new_path

def process_all(df, rir_dict, snr_db, noise_gen, env_name, data_type):
    dfs = []
    for rir_name, rir_path in rir_dict.items():
        df_copy = df.sample(frac=1/len(rir_dict))  # Adjust fraction according to number of splits
        df = df.drop(df_copy.index)
        df_copy['path'] = df_copy.apply(add_rir_noise, axis=1, rir_path=rir_path, snr_db=snr_db["snr_3db"], noise_gen=noise_gen, rir_name=rir_name)
        dfs.append(df_copy)
    df_final = pd.concat(dfs)
    os.makedirs(f'csv/{env_name}', exist_ok=True)
    df_final.to_csv(f'csv/{env_name}/{data_type}.csv', index=False)
    print(f'csv/{env_name}/{data_type}.csv')

df_train = pd.read_csv('csv/train.csv')
df_test = pd.read_csv('csv/test.csv')
df_val = pd.read_csv('csv/validation.csv')

noise_path = '/project/lt200007-tspai2/chavezpor/musan/all_noise.wav'
os.makedirs("wavs_rir_added", exist_ok=True)

noise_gen = NoiseGenerator(noise_path)

def process_train_room():
    process_all(df_train, rir_dict_room, snr_dict, noise_gen, 'room', 'train')

def process_test_room():
    process_all(df_test, rir_dict_room, snr_dict, noise_gen, 'room', 'test')

def process_val_room():
    process_all(df_val, rir_dict_room, snr_dict, noise_gen, 'room', 'validation')

def process_train_exhibition():
    process_all(df_train, rir_dict_exhibition, snr_dict, noise_gen, 'exhibition', 'train')

def process_test_exhibition():
    process_all(df_test, rir_dict_exhibition, snr_dict, noise_gen, 'exhibition', 'test')

def process_val_exhibition():
    process_all(df_val, rir_dict_exhibition, snr_dict, noise_gen, 'exhibition', 'validation')

print("start...")
with ProcessPoolExecutor() as executor:
    executor.submit(process_train_room)
    executor.submit(process_test_room)
    executor.submit(process_val_room)
    executor.submit(process_train_exhibition)
    executor.submit(process_test_exhibition)
    executor.submit(process_val_exhibition)