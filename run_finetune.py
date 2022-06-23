import comet_ml
from utils.log_utils import log_extra 
from utils.fine_tuning import fine_tune
from genomic_benchmarks.data_check import list_datasets
import os

########
### Known Issues: ###
########
# what more params to extract to "upper" script?
# dataset se stahuje ikdyz uz stahli je, kouknout do genomic benchmarks
# 
# resursive warning during training - is there a problem?
    #  huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    
    
########
## INFO
#######
# cca 00:06:11 na jednu epochu (2/3x eval a 1/2x save)
# 

    
    
    
########
# important paramaters so far hardcoded in the fine_tuning script:
########
# early_stopping_patience = 3,
# early_stopping_threshold = 0.02
# POSITIVE_CLASS_INDEX = 1
# logging_steps=400
# save_steps=800


genomic_datasets = list_datasets()
# list_datasets() is not deterministic (resets on lib load)
genomic_datasets.sort()
# skipping multiclass dataset for now
genomic_datasets.remove("human_ocr_ensembl")
genomic_datasets.remove("dummy_mouse_enhancers_ensembl")
print(genomic_datasets)

#         config:
model_name = "DNADeberta_fine"
commet_key = 'EpKIINrla6U4B4LJhd9Sv4i0b'
hug_model_link = "simecek/DNADeberta"
epochs = 1

os.environ['COMET_API_KEY'] = commet_key

for dataset_name in genomic_datasets:
    for _ in range(1):
        comet_ml.init(project_name="DNA_finetuning", api_key=commet_key)
        
        f1_test, acc_test = fine_tune(hug_model_link, model_name, dataset_name, epochs)
        
        log_extra(model_name, dataset_name, f1_test, acc_test)
        
# TODO output csv file with results

print('ALL DONE')