# 🤗 Models For Genomic Sequences

## Notebooks

* [Human_DNA_small](Human_DNA_small.ipynb): DeBERTa small model trained over [Human_DNA_v0](https://huggingface.co/datasets/simecek/Human_DNA_v0) dataset (10 epochs)
* [DNA data](DNA_data.ipynb): Reshaping Human genome (DNA) into HF dataset
* [DNA data configurable](experiments/organisms/DNA_data_configurable.ipynb): Configurable script for downloading, processing, and uploading of DNA data from fasta files to HuggingFace (HF) datasets
* [Architecture pretraining](experiments/architectures/architecture_pretraining.ipynb): Script for pretraining various architectures on human DNA
* [Training_with_cDNA](Training_with_cDNA.ipynb): Current training script demonstrated on BERT architecture and cDNA dataset, not very useful
* [env_init](env_init.ipynb): Internal script for installation needed on our virtual machines (E-INFRA HUB)


## Datasets

*   [Human_DNA_v0](https://huggingface.co/datasets/simecek/Human_DNA_v0): DNA splitted into 10kb pieces
*   [Human_DNA_v0_DNABert6tokenized](https://huggingface.co/datasets/simecek/Human_DNA_v0_DNABert6tokenized): DNA tokenized and ready for language model training (tensors of 512 tokens)
*   [Human_cdna](https://huggingface.co/datasets/Vlasta/human_cdna): `Homo_sapiens.GRCh38.cdna.abinitio.fa.gz` reshaped into HF dataset 
*   [Other organisms](https://huggingface.co/davidcechak) HF datasets of other organisms can be found here (TODO: break them down here and link one by one)

## Models

* [cDNABERT_v0](https://huggingface.co/simecek/cDNABERT_v0): the output of [Training_with_cDNA](Training_with_cDNA.ipynb) script, not very useful model
* [DNA_Deberta_1epoch](https://huggingface.co/Vlasta/humandna_deberta_default_empty_stud_8442): Deberta pretrained on DNA for 1 epoch
* [DNA_Bert_1epoch](https://huggingface.co/Vlasta/humandna_bert_default_beautiful_bench_4197): Bert pretrained on DNA for 1 epoch
* [DNA_DistilBert_1epoch](https://huggingface.co/Vlasta/humandna_distillbert_random_systematic_walrus_56): Bert pretrained on DNA for 1 epoch

## Tokenizers

* [DNA_bert_6](https://huggingface.co/armheb/DNA_bert_6): we are currently using this tokenize (the sequence needs to be preprocessed before using it)

## Other(s)
