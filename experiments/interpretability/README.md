# Interpretability

- [1_g_quadruplex_tokenization.ipynb](./1_g_quadruplex_tokenization.ipynb) - used for tokenization of the G quadruplex dataset to K-mers where K=6
- [2_g_quadruplex_training.ipynb](./2_g_quadruplex_training.ipynb) - used for training [armheb/DNA_bert_6](https://huggingface.co/armheb/DNA_bert_6?text=The+goal+of+life+is+%5BMASK%5D.) model for our task
- [3_dnabert_classification_interpretation_not_trained.ipynb](./3_dnabert_classification_interpretation_not_trained.ipynb) - used to visualize the Layer Integrated Gradients of an untrained model 
- [3_dnabert_classification_interpretation_trained.ipynb](./3_dnabert_classification_interpretation_trained.ipynb) - used for visualize the Layer Integrated Gradients of model trained for 1 epoch
- [3_dnabert_classification_interpretation_trained_overfit.ipynb](./3_dnabert_classification_interpretation_trained_overfit.ipynb) - used for visualize the Layer Integrated Gradients of model trained for 25 epochs which overfits

The 3. part is divided into three parts so that we can observe the difference in visualizations of attribution scores with respect to how well the model is able to solve the task and generalize at the same time.
