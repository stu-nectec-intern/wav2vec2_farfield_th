from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydub import AudioSegment
import torch, torchaudio
from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC, Pipeline
from transformers.pipelines import PIPELINE_REGISTRY
import tempfile
import os, gc
from transformers import AutoProcessor
from pyctcdecode import build_ctcdecoder
import logging 

logger = logging.getLogger('my-logger')
logger.propagate = False

os.environ["TOKENIZERS_PARALLELISM"] = "false"

processor = AutoProcessor.from_pretrained("Pongsathorn/wav2vec2_cosine_48000")
vocab_dict = processor.tokenizer.get_vocab()
sorted_vocab_dict = {k.lower(): v for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}

decoder = build_ctcdecoder(
    labels=list(sorted_vocab_dict.keys()),
    kenlm_model_path="language_model/4gram_correct.bin",
)

from transformers import Wav2Vec2ProcessorWithLM

processor_with_lm = Wav2Vec2ProcessorWithLM(
    feature_extractor=processor.feature_extractor,
    tokenizer=processor.tokenizer,
    decoder=decoder
)

class MyWav2Vec2Pipeline(Pipeline):
    def __init__(self, model="Pongsathorn/wav2vec2_cosine_48000", device=0):
        self.model = Wav2Vec2ForCTC.from_pretrained(model).to("cpu")
        self.processor_with_lm = processor_with_lm

        super().__init__(model=self.model, tokenizer=self.processor_with_lm, device="cpu")

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, inputs):
        waveform, original_sampling_rate = torchaudio.load(inputs)
        waveform = waveform[0].reshape(1, -1) #handling of stereo (2 channel) 
        resampler = torchaudio.transforms.Resample(orig_freq=original_sampling_rate, new_freq=16000)
        resampled_array = resampler(waveform).numpy().flatten()
        input_values = self.processor_with_lm(resampled_array, sampling_rate=16_000, return_tensors="pt").input_values
        return {"input_values": input_values.to("cpu")}  # use CPU here

    def _forward(self, model_inputs):
        logits = self.model(model_inputs["input_values"]).logits
        return {"logits": logits}

    def postprocess(self, model_outputs):
        predicted_ids = torch.argmax(model_outputs["logits"], dim=-1)
        transcription_lm = self.processor_with_lm.batch_decode(model_outputs["logits"].detach().cpu().numpy()).text
        return {"transcription": transcription_lm}

PIPELINE_REGISTRY.register_pipeline("wav2vec2", pipeline_class=MyWav2Vec2Pipeline)
pipeline = MyWav2Vec2Pipeline()
app = FastAPI()

def transcribe_long_audio(file_path, chunk_size=10_000):  # chunk_size in milliseconds
    transcriptions = []
    audio = AudioSegment.from_wav(file_path)
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_chunk:
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size + 200]
            chunk.export(temp_chunk.name, format="wav")
            result = pipeline.preprocess(temp_chunk.name)
            result = pipeline._forward(result)
            result = pipeline.postprocess(result)
            transcription = result["transcription"][0]
            transcriptions.append(transcription)
            del result
            gc.collect()
    del audio
    gc.collect()
    return {"transcription": " ".join(transcriptions)}  # return as single string

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if file.content_type.startswith('audio/'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            data = await file.read()
            temp_file.write(data)

            if file.filename.endswith('.mp3'):
                audio = AudioSegment.from_mp3(temp_file.name)
                audio.export(temp_file.name, format="wav")

            result = pipeline.preprocess(temp_file.name)
            result = pipeline._forward(result)
            result = pipeline.postprocess(result)

            del audio
            gc.collect()

        return result
    else:
        return JSONResponse(status_code=400, content={"detail": "File type not supported."})

@app.post("/long_transcribe")
async def long_transcribe_audio(file: UploadFile = File(...)):
    if file.content_type.startswith('audio/'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            data = await file.read()
            temp_file.write(data)

            audio = None
            if file.filename.endswith('.mp3'):
                audio = AudioSegment.from_mp3(temp_file.name)
                audio.export(temp_file.name, format="wav")

            result = transcribe_long_audio(temp_file.name)

            if audio is not None:
                del audio
                gc.collect()

            return result
    else:
        return JSONResponse(status_code=400, content={"detail": "File type not supported."})

# run with: uvicorn local_cpu_inference:app --reload --port 8000
