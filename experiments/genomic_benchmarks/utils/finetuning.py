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
from transformers import set_seed, logging

def finetune(hug_model_link, dataset_name, epochs, POSITIVE_CLASS_INDEX, NUM_OF_LABELS, seed):
    logging.set_verbosity_error()
    set_seed(seed) 
    
    print(torch.cuda.get_device_name(0))
    print(dataset_name)
    
    tokenizer = AutoTokenizer.from_pretrained("armheb/DNA_bert_6")
    model = AutoModelForSequenceClassification.from_pretrained(hug_model_link, num_labels=NUM_OF_LABELS)
    model.to('cuda')
    
            
    """## 0) Get dataset"""
    
    if not is_downloaded(dataset_name):
        print("downloading dataset")
        download_dataset(dataset_name, version=0, force_download = False, use_cloud_cache = False)

    def kmers(s, k=6):
        return [s[i:i + k] for i in range(0, len(s), k) if i + k <= len(s)]

    tmp_dict = {}
    for dset in ['train', 'test']:
        list_dir = os.listdir(Path(f'/home/jovyan/.genomic_benchmarks/{dataset_name}/{dset}/'))
        for c in list_dir:
            for f in Path(f'/home/jovyan/.genomic_benchmarks/{dataset_name}/{dset}/{c}/').glob('*.txt'):
                txt = f.read_text()
                # temporal solution to fix the problem with repeated indexes in positive and negative class (index X can be found in both classes, for each index X)
                tmp_dict[f.stem + c] = (dset, int(c == list_dir[POSITIVE_CLASS_INDEX]), txt)

    df = pd.DataFrame.from_dict(tmp_dict).T.rename(columns = {0: "dset", 1: "cat", 2: "seq"})

    train_valid_index = df.query("dset == 'train'").shape[0] // 100 * 80
    train_df = df[df['dset']=='train'].iloc[:train_valid_index,:]
    valid_df = df[df['dset']=='train'].iloc[train_valid_index:,:]
    test_df = df[df['dset']=='test']

    datasets = [train_df, valid_df, test_df]
    datasets = [Dataset.from_pandas(x) for x in datasets]

    def tok_func(x): return tokenizer(" ".join(kmers(x["seq"])))
    datasets = [x.map(tok_func, batched=False).rename_columns({'cat':'labels'}) for x in datasets]

    dds = DatasetDict({
        'train': datasets[0],
        'validation': datasets[1],
        'test':  datasets[2],
    })

    """## 1) Fine-tuning"""
    
#     see https://huggingface.co/microsoft/deberta-v3-xsmall
    training_args = TrainingArguments(
        output_dir='./model',
        num_train_epochs=epochs,
        per_device_train_batch_size=64, 
        per_device_eval_batch_size=64,    
        fp16=True,
        evaluation_strategy='epoch',
        logging_strategy='epoch',
        save_strategy="no",
        warmup_steps= 1000, 
#         1.5e-5,2e-5,3e-5,4e-5
        learning_rate= 2e-5,
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
        train_dataset = dds['train'], 
        eval_dataset = dds['validation'],
        tokenizer=tokenizer, 
        compute_metrics=compute_metrics,
    #     early_stopping_patience - considers evaluation calls (steps/epochs)
        # callbacks=[EarlyStoppingCallback(early_stopping_patience = 3, early_stopping_threshold = 0.02)],
    )

    trainer.train()
    
    predictions = trainer.predict(dds['test'])
    print(predictions.metrics)

    return predictions.metrics["test_accuracy"], predictions.metrics["test_f1"]