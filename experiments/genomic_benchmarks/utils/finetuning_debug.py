import os
import comet_ml
import torch 
import datasets 
import numpy as np
import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict, load_metric
from huggingface_hub import notebook_login
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, EarlyStoppingCallback
from genomic_benchmarks.loc2seq import download_dataset
from genomic_benchmarks.data_check import list_datasets, is_downloaded
from utils.preprocessing import get_preprocessed_datasets
from transformers import set_seed

# todo change-able stride
# todo support multiclass datasets

def fine_tune_debug(hug_model_link, dataset_name, epochs, POSITIVE_CLASS_INDEX, NUM_OF_LABELS, seed):
    print(torch.cuda.get_device_name(0))
    print(dataset_name)
    set_seed(seed) 

    kmer_len = 6
    stride = 6
    
    tokenizer = AutoTokenizer.from_pretrained("armheb/DNA_bert_6")
    model = AutoModelForSequenceClassification.from_pretrained(hug_model_link, num_labels=NUM_OF_LABELS)
    model.to('cuda')
    
            
    """## 0) Get dataset"""
    
    encoded_samples, encoded_samples_test = get_preprocessed_datasets(
        dataset_name, 
        tokenizer, 
        kmer_len = kmer_len, 
        stride = stride,
    )
    
    ratio = 0.2
    train_dset_len = len(encoded_samples)
    val_dset_len = int(train_dset_len*ratio)
    print('Dataset', dataset_name, 'of length', train_dset_len, 'has valid size of', val_dset_len, 'that is ratio of ', ratio)
    
    """## 1) Fine-tuning"""
    
    training_args = TrainingArguments(
        output_dir='./model',
        num_train_epochs=epochs,
        per_device_train_batch_size=64, 
        per_device_eval_batch_size=64,    
        fp16=True,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        save_strategy="no",
        # learning_rate= zmenšit -- deberta finetuning default
    )

    def compute_metrics(eval_preds):
        # metric = load_metric("accuracy", "f1")
        metric = load_metric("glue", "mrpc")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset = encoded_samples[:-val_dset_len], 
        eval_dataset = encoded_samples[-val_dset_len:],
        # tokenizer=tokenizer, 
        compute_metrics=compute_metrics,
    #     early_stopping_patience - considers evaluation calls (for us, steps at the moment)
        # callbacks=[EarlyStoppingCallback(early_stopping_patience = 3, early_stopping_threshold = 0.02)],
    )

    trainer.train()
    
    predictions = trainer.predict(encoded_samples_test)
    print(predictions.metrics)
    
    del model
    del trainer
    return predictions.metrics["test_accuracy"], predictions.metrics["test_f1"]