import seaborn as sns
import pandas as pd
import numpy as np
import torchaudio
from matplotlib import pyplot as plt
import concurrent.futures

df_train = pd.read_csv('csv/merged_train.csv', index_col=0)
df_test = pd.read_csv('csv/test.csv', index_col=0)
df_val = pd.read_csv('csv/validation.csv', index_col=0)

def get_audio_length(path):
    waveform, sample_rate = torchaudio.load(path)
    duration = waveform.shape[1] / sample_rate
    return duration

with concurrent.futures.ProcessPoolExecutor() as executor:
    train_lengths = list(executor.map(get_audio_length, df_train['path']))
    test_lengths = list(executor.map(get_audio_length, df_test['path']))
    val_lengths = list(executor.map(get_audio_length, df_val['path']))

all_length = train_lengths + test_lengths + val_lengths

# Creating a figure and axis object
fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

# plot histograms using seaborn
sns.histplot(train_lengths, bins=50, kde=True, ax=axs[0, 0])
axs[0, 0].set_title('Train Audio Lengths')
axs[0, 0].set_xlabel('Duration (s)')
axs[0, 0].set_ylabel('Count')
axs[0, 0].annotate(f"Max: {np.max(train_lengths):.2f}s\nMean: {np.mean(train_lengths):.2f}s\nMin: {np.min(train_lengths):.2f}s\nTotal: {np.sum(train_lengths)/3600:.2f}h", xy=(0.7, 0.7), xycoords='axes fraction')

sns.histplot(test_lengths, bins=50, kde=True, ax=axs[0, 1])
axs[0, 1].set_title('Test Audio Lengths')
axs[0, 1].set_xlabel('Duration (s)')
axs[0, 1].set_ylabel('Count')
axs[0, 1].annotate(f"Max: {np.max(test_lengths):.2f}s\nMean: {np.mean(test_lengths):.2f}s\nMin: {np.min(test_lengths):.2f}s\nTotal: {np.sum(test_lengths)/3600:.2f}h", xy=(0.7, 0.7), xycoords='axes fraction')

sns.histplot(val_lengths, bins=50, kde=True, ax=axs[1, 0])
axs[1, 0].set_title('Validation Audio Lengths')
axs[1, 0].set_xlabel('Duration (s)')
axs[1, 0].set_ylabel('Count')
axs[1, 0].annotate(f"Max: {np.max(val_lengths):.2f}s\nMean: {np.mean(val_lengths):.2f}s\nMin: {np.min(val_lengths):.2f}s\nTotal: {np.sum(val_lengths)/3600:.2f}h", xy=(0.7, 0.7), xycoords='axes fraction')

sns.histplot(all_length, bins=50, kde=True, ax=axs[1, 1])
axs[1, 1].set_title('All Audio Lengths')
axs[1, 1].set_xlabel('Duration (s)')
axs[1, 1].set_ylabel('Count')
axs[1, 1].annotate(f"Max: {np.max(all_length):.2f}s\nMean: {np.mean(all_length):.2f}s\nMin: {np.min(all_length):.2f}s\nTotal: {np.sum(all_length)/3600:.2f}h", xy=(0.7, 0.7), xycoords='axes fraction')

plt.tight_layout()
plt.show()

# Save the figure
fig.savefig('audio_lengths.png', dpi=300)
