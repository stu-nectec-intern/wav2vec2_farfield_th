import re
import pandas as pd
import torchaudio
import gc
from pythainlp.tokenize import word_tokenize
from datasets import load_dataset, Audio, Dataset, DatasetDict
from transformers import Wav2Vec2Processor
from datasets import disable_caching
import logging

# I used vocab and preprocessor that created before

class DatasetPreparation:
    def __init__(self, processor_path):
        self.processor = Wav2Vec2Processor.from_pretrained(processor_path)

    @staticmethod
    def process_and_resample_audio(item):
        waveform, original_sampling_rate = torchaudio.load(item["path"])
        resampler = torchaudio.transforms.Resample(orig_freq=original_sampling_rate, new_freq=16000)
        item["audio"] = {"array": resampler(waveform).numpy().squeeze(), "sampling_rate": 16000}
        return item

    @staticmethod
    def th_words_tokenize(batch):
        sentence = batch["sentence"]
        pattern = r"[^ก-๙a-zA-Z0-9\s]+"
        sentence = re.sub(pattern, '', sentence)
        tokens = word_tokenize(sentence.replace('เเ', 'แ'), engine="deepcut")
        # Deal with 'ๆ'
        processed_tokens = []
        for i, token in enumerate(tokens):
            if token == 'ๆ':
                # go back to find the last non-whitespace token
                for j in range(i-1, -1, -1):
                    if tokens[j].strip():  # this will be false for whitespace
                        processed_tokens.append(tokens[j])
                        break
            else:
                processed_tokens.append(token)
        
        batch["sentences"] = " ".join(processed_tokens).replace("  ", " ")
        return batch

    def prepare_dataset(self, batch):
        audio = batch["audio"]
        batch["input_values"] = self.processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["labels"] = self.processor.tokenizer(batch["sentences"], return_tensors="pt").input_ids[0]
        return batch

    @staticmethod
    def save_data(dataset, path):
        dataset = dataset.shuffle(seed=42)
        dataset.save_to_disk(path)

    def run(self, path_dict):
        disable_caching()
        print('Load data...')
        dfs = {split: pd.read_csv(path, index_col=0)[["path", "sentence"]] for split, path in path_dict.items()}  

        for df in dfs.values():
            df["sentence"] = df["sentence"].str.lower()

        dataset = DatasetDict()

        for key, df in dfs.items():
            dataset[key] = Dataset.from_pandas(df)

        del dfs

        for key in dataset.keys():
            print(f'tokenizing {key}...')
            dataset[key] = dataset[key].map(self.th_words_tokenize,remove_columns=["sentence"], num_proc=2)
            print(f'resampling {key}...')
            dataset[key] = dataset[key].map(self.process_and_resample_audio, num_proc=2)
            print(f'generate input feature {key}...')
            dataset[key] = dataset[key].map(self.prepare_dataset, remove_columns=dataset[key].column_names, num_proc=2)

        print('saving dataset...')

        self.save_data(dataset, "dataset_wav2vec2_far_field/dataset")


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.WARNING)
    
    path_dict = {
        'train': 'csv/merged_train.csv',
        'test': 'csv/test.csv',
        'val': 'csv/validation.csv'
    }

    # Add path to your existing processor
    existing_processor_path = "/project/lt200007-tspai2/thepeach/wav2vec2-xlsr53-TH-cmv-processor"
    prep = DatasetPreparation(existing_processor_path)

    prep.run(path_dict)
