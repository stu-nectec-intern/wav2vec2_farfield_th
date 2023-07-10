from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydub import AudioSegment
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Pipeline
from transformers.pipelines import PIPELINE_REGISTRY
import tempfile
import os


class MyWav2Vec2Pipeline(Pipeline):
    def __init__(self, model="Pongsathorn/wav2vec2_cosine_48000", processor="Pongsathorn/wav2vec2_cosine_48000", device=0):
        self.model = Wav2Vec2ForCTC.from_pretrained(model).to("cpu")
        self.processor = Wav2Vec2Processor.from_pretrained(processor)
        super().__init__(model=self.model, tokenizer=self.processor, device="cpu")

    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}

    def preprocess(self, inputs):
        waveform, original_sampling_rate = torchaudio.load(inputs)
        waveform = waveform[0].reshape(1, -1)
        resampler = torchaudio.transforms.Resample(orig_freq=original_sampling_rate, new_freq=16000)
        resampled_array = resampler(waveform).numpy().flatten()
        input_values = self.processor(resampled_array, sampling_rate=16_000, return_tensors="pt").input_values
        return {"input_values": input_values.to("cpu")}  # use CPU here

    def _forward(self, model_inputs):
        logits = self.model(model_inputs["input_values"]).logits
        return {"logits": logits}

    def postprocess(self, model_outputs):
        predicted_ids = torch.argmax(model_outputs["logits"], dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return {"transcription": transcription}

    def transcribe_long_audio(self, file_path, chunk_size=10_000):  # chunk_size in milliseconds
        audio = AudioSegment.from_wav(file_path)
        transcriptions = []
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_chunk:
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size + 200]
                chunk.export(temp_chunk.name, format="wav")
                result = self.preprocess(temp_chunk.name)
                result = self._forward(result)
                result = self.postprocess(result)
                transcriptions.append(result["transcription"])
        return {"transcription": " ".join(transcriptions)}

PIPELINE_REGISTRY.register_pipeline("wav2vec2", pipeline_class=MyWav2Vec2Pipeline)
pipeline = MyWav2Vec2Pipeline()
app = FastAPI()

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

        return result
    else:
        return JSONResponse(status_code=400, content={"detail": "File type not supported."})

@app.post("/long_transcribe")
async def long_transcribe_audio(file: UploadFile = File(...)):
    if file.content_type.startswith('audio/'):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            data = await file.read()
            temp_file.write(data)

            if file.filename.endswith('.mp3'):
                audio = AudioSegment.from_mp3(temp_file.name)
                audio.export(temp_file.name, format="wav")

            result = pipeline.transcribe_long_audio(temp_file.name)

        return result
    else:
        return JSONResponse(status_code=400, content={"detail": "File type not supported."})

# run with: uvicorn local_cpu_main:app --reload --port 8000