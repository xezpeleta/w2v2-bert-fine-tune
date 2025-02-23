import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import Audio, load_dataset, load_metric
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2BertForCTC,
    Wav2Vec2BertProcessor,
    SeamlessM4TFeatureExtractor,
    Wav2Vec2CTCTokenizer,
)

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2BertProcessor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    output_dir: str = field(
        metadata={"help": "The output directory where the model checkpoints will be written."}
    )

@dataclass
class DataTrainingArguments:
    dataset_name: str = field(
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    language: str = field(
        metadata={"help": "Language code for the dataset"}
    )
    chars_to_remove: Optional[str] = field(
        default="[\,\?\.\!\-\;\:\"\"\%\'\"\�\'\»\«]",
        metadata={"help": "Regular expression for characters to remove from transcriptions"}
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Load datasets
    common_voice_train = load_dataset(
        data_args.dataset_name, 
        data_args.language, 
        split="train+validation",
        use_auth_token=True
    )
    common_voice_test = load_dataset(
        data_args.dataset_name,
        data_args.language,
        split="test",
        use_auth_token=True
    )

    # Remove unnecessary columns
    columns_to_remove = ["accent", "age", "client_id", "down_votes", "gender", "locale", "segment", "up_votes"]
    common_voice_train = common_voice_train.remove_columns(columns_to_remove)
    common_voice_test = common_voice_test.remove_columns(columns_to_remove)

    # Preprocess text
    chars_to_remove_regex = data_args.chars_to_remove

    def remove_special_characters(batch):
        batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
        return batch

    common_voice_train = common_voice_train.map(remove_special_characters)
    common_voice_test = common_voice_test.map(remove_special_characters)

    # Create vocabulary
    def extract_all_chars(batch):
        all_text = " ".join(batch["sentence"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocab_train = common_voice_train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_train.column_names)
    vocab_test = common_voice_test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_test.column_names)

    vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}

    # Add special tokens
    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    # Save vocab
    os.makedirs(model_args.output_dir, exist_ok=True)
    vocab_path = os.path.join(model_args.output_dir, "vocab.json")
    with open(vocab_path, 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)

    # Create tokenizer and feature extractor
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|"
    )
    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(model_args.model_name_or_path)
    processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # Resample audio
    common_voice_train = common_voice_train.cast_column("audio", Audio(sampling_rate=16_000))
    common_voice_test = common_voice_test.cast_column("audio", Audio(sampling_rate=16_000))

    # Prepare dataset
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["input_length"] = len(batch["input_features"])
        batch["labels"] = processor(text=batch["sentence"]).input_ids
        return batch

    common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=common_voice_train.column_names)
    common_voice_test = common_voice_test.map(prepare_dataset, remove_columns=common_voice_test.column_names)

    # Load model
    model = Wav2Vec2BertForCTC.from_pretrained(
        model_args.model_name_or_path,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.0,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        add_adapter=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    # Data collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # Metric
    wer_metric = load_metric("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=common_voice_train,
        eval_dataset=common_voice_test,
        tokenizer=processor.feature_extractor,
    )

    # Train
    trainer.train()

    # Save final model and processor
    trainer.save_model()
    processor.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()