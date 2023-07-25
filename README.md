# wav2vec2_farfield_th

This repository contains the `wav2vec2_farfield_th` model, which is a pre-trained model for speech recognition in the Thai language. The original model can be found under [pytorch/fairseq/examples/wav2vec](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec#wav2vec-20).

To effectively utilize this model, follow these steps:

1. **Study Wav2Vec2 Model:**
   - [Fine-Tune Wav2Vec2 for English ASR with 🤗 Transformers](https://huggingface.co/blog/fine-tune-wav2vec2-english): Read this blog post to understand how to fine-tune Wav2Vec2 for English Automatic Speech Recognition (ASR).
   - [Overview of Wav2vec 2.0](https://jonathanbgn.com/2021/09/30/illustrated-wav2vec-2.html): This article provides an illustrated overview of the Wav2Vec 2.0 model.
   - [Huggingface documentation](https://huggingface.co/docs/transformers/model_doc/wav2vec2): For in-depth technical information and documentation about Wav2Vec2 and other models, refer to the official Huggingface documentation.
   - [Study the trainer of Wav2vec2](https://huggingface.co/docs/transformers/main_classes/trainer)

2. **Fine-tune Wav2Vec2 with the Prepared Dataset:**
   - Dataset that we use in this repository is the combination of [Common Voice 13](https://commonvoice.mozilla.org/en/datasets), [Gowajee](https://github.com/ekapolc/gowajee_corpus), and [ASR Thai elderly dataset](https://github.com/VISAI-DATAWOW/Thai-Elderly-Speech-dataset/releases/tag/v1.0.0).
   - To prepare the data for fine-tuning, follow these steps:
     - Clean special characters from the sentences.
     - Replace "เเ" with "แ".
     - Tokenize the sentences using Deepcut.
     - Handle 'ๆ' tokens by replacing them with the previous token. Here's the Python code for this preprocessing step:
     
```python
def th_words_tokenize(batch):
    import re
    from pythainlp.tokenize import word_tokenize

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
```


   - Create a processor (it's recommended to sort the vocab dictionary) and resample the audio data to 16k, following the wav2vec2 requirements.

3. **Try Various Techniques to Improve Performance:**
   - Implement the [Cosine scheduler with restart](https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup.num_training_steps) to optimize the training process.
   - Check out the [Boosting Wav2Vec2 with n-grams in 🤗 Transformers](https://huggingface.co/blog/wav2vec2-with-ngram) blog post to explore n-gram techniques for further performance improvements.

4. **Data Augmentation simulating Farfiled:**
   - Convolve with selected Room impulse response and add noise utilizing the function from torchaudio from this [Documentation](https://pytorch.org/audio/stable/tutorials/audio_data_augmentation_tutorial.html).

![Image Alt Text](./images/rir.png)

   - Source of the Noise is from [Musan](https://www.openslr.org/17/) and Room Impulse Response (RIR) from [Butspeech](https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database).

Happy ASR development with `wav2vec2_farfield_th`! 🎉



