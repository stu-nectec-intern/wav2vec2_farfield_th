import os
import pandas as pd
import torchaudio
from tqdm import tqdm

data_paths = ["train.tsv","test.tsv","dev.tsv"]
prefix_path = "cv-corpus-13.0-2023-03-09/th/clips/"
save_path = "cv-corpus-13.0-2023-03-09/th/wavs/"
sr_16k = 16000

os.makedirs(save_path, exist_ok=True)

for data_path in data_paths:
    df = pd.read_csv(f"cv-corpus-13.0-2023-03-09/th/{data_path}", sep="\t", usecols=["path", "sentence"])
    for fname in tqdm(df["path"]):
        wav, sr = torchaudio.load(prefix_path + fname, format="mp3") 
        wav_16k = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sr_16k)(wav)
        wav_fname = fname[:-4] + ".wav" 
        torchaudio.save(filepath=save_path + wav_fname, src=wav_16k, sample_rate=sr_16k)


print(len(os.listdir('cv-corpus-13.0-2023-03-09/th/clips')))
print(len(os.listdir('cv-corpus-13.0-2023-03-09/th/wavs')))
