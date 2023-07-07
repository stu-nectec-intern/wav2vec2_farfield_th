import re
import json
import pandas as pd
import torchaudio
import gc, os
from pythainlp.tokenize import word_tokenize
from datasets import load_dataset, Audio, Dataset, DatasetDict
from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
from datasets import disable_caching
import logging

'''
Class: DatasetPreparation
    __init__(self)
    Initializes the DatasetPreparation object.

    process_and_resample_audio(self, item)
    Takes in an audio item, loads it into a waveform using torchaudio.load(), and resamples it to a new frequency of 16kHz using torchaudio.transforms.Resample(). It then returns the audio item with its waveform array and sampling rate.

    th_words_tokenize(self, batch)
    Performs preprocessing on the input sentences. Converts the sentence to lowercase, removes non-Thai and non-English characters, and tokenizes the sentence into words using the Deepcut tokenizer from the PyThaiNLP library. It also handles the 'ๆ' character, duplicating the previous token if 'ๆ' is encountered. The processed tokens are then joined back into a sentence and returned.

    prepare_dataset(self, batch)
    Prepares the dataset for training by converting the audio array to input values using the processor and tokenizing the sentences into labels.

    save_data(self, dataset, path)
    Shuffles the dataset and saves it to the specified disk path.

    extract_all_chars(self, df)
    Extracts all unique characters from the sentences in the input dataframe.

    create_and_save_processor(self, vocab_list)
    Creates and saves a Wav2Vec2Processor object. This is done by creating a vocabulary dictionary from the input list, saving the vocabulary to a JSON file, initializing a Wav2Vec2CTCTokenizer and a Wav2Vec2FeatureExtractor, and finally creating the Wav2Vec2Processor with the tokenizer and feature extractor.

    run(self, path_dict)
    The main function of the class that is responsible for loading the data, creating the processor, creating and processing the dataset, and finally saving it.
'''

class DatasetPreparation:
    def __init__(self):
        self.processor = None

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

    @staticmethod
    def extract_all_chars(df):
        return list(set(" ".join(df["sentence"])))

    def create_and_save_processor(self, vocab_list):
        # Check if the processor already exists
        if os.path.exists("wav2vec2-xlsr53-TH-cmv-processor"):
            print("'wav2vec2-xlsr53-TH-cmv-processor' already exists. Skipping the processor creation.")
            self.processor = Wav2Vec2Processor.from_pretrained("wav2vec2-xlsr53-TH-cmv-processor")
            return

        # Create a new processor if it doesn't exist
        vocab_dict = {v: k for k, v in enumerate(vocab_list)}
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)

        with open('vocab.json', 'w') as vocab_file:
            json.dump(vocab_dict, vocab_file)

        tokenizer = Wav2Vec2CTCTokenizer("vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
        self.processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

        self.processor.save_pretrained("wav2vec2-xlsr53-TH-cmv-processor")

    def run(self, path_dict):
        disable_caching()
        print('Load data...')
        dfs = {split: pd.read_csv(path, index_col=0)[["path", "sentence"]] for split, path in path_dict.items()}  

        for df in dfs.values():
            df["sentence"] = df["sentence"].str.lower()

        vocab_list = self.extract_all_chars(pd.concat(dfs.values()))
        print("vocab and preprocessor")
        self.create_and_save_processor(vocab_list)
        gc.collect()

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

        self.save_data(dataset, "dataset_wav2vec2/dataset")

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.WARNING)
    
    path_dict = {
        'train': 'csv/train.csv',
        'test': 'csv/test.csv',
        'val': 'csv/validation.csv'
    }

    prep = DatasetPreparation()

    prep.run(path_dict)
