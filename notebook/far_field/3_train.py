import gc,json, re
import pandas as pd
import torch, torchaudio, evaluate
import numpy as np
from dataclasses import dataclass, field
from datasets import Audio, Dataset, disable_caching, load_from_disk
from transformers import TrainingArguments, Trainer, Wav2Vec2ForCTC, Wav2Vec2Processor, SchedulerType, get_cosine_with_hard_restarts_schedule_with_warmup
from typing import Any, Dict, List, Optional, Union

disable_caching()
processor = Wav2Vec2Processor.from_pretrained("wav2vec2-xlsr53-TH-cmv-processor")
train_dataset = load_from_disk('/project/lt200007-tspai2/thepeach/dataset_wav2vec2_far_field/dataset/train')
val_dataset = load_from_disk('/project/lt200007-tspai2/thepeach/dataset_wav2vec2_far_field/dataset/val')

hyper_param = {
    'num_train': 20,
    'cosine_cycles': 4,
    'learning_rate': 1e-4,
    'train_len':len(train_dataset)
}

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer_metric = evaluate.load("metric/wer.py")
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

def main():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    num_gpus = torch.cuda.device_count()

    seed_val = 42

    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    per_device_train_batch_size = 16
    num_training_steps_per_epoch = hyper_param['train_len'] / (per_device_train_batch_size * num_gpus)  # considering your num_gpus
    num_training_steps = num_training_steps_per_epoch * hyper_param['num_train']
    num_warmup_steps = 0.1 * num_training_steps  # 10% of training


    training_args = TrainingArguments(
        output_dir="wav2vec2-xlsr53-TH-cmv-ckp-farfield/",
        group_by_length=True,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        metric_for_best_model='wer',
        evaluation_strategy="steps",
        save_strategy="steps",
        num_train_epochs=hyper_param['num_train'],
        save_steps=1000,
        eval_steps=1000,
        logging_steps=100,
        learning_rate=hyper_param['learning_rate'],
        save_total_limit=3,
        fp16=True,
        greater_is_better=False,
        eval_accumulation_steps=100,
        dataloader_drop_last=True,
    )

    model = Wav2Vec2ForCTC.from_pretrained(
        "model",
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    model.freeze_feature_extractor()
    model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=hyper_param['learning_rate'])

    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps= num_warmup_steps,
        num_training_steps=num_training_steps, 
        num_cycles=hyper_param['cosine_cycles']
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.feature_extractor,
        optimizers = (optimizer, scheduler)
    )
    print("Start train...")

    trainer.train()

if __name__ == "__main__":
    main()

#https://huggingface.co/docs/transformers/main_classes/trainer
#https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup.num_training_steps
