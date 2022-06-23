import comet_ml
import os
import csv
import datetime
from statistics import stdev, fmean
from genomic_benchmarks.data_check import list_datasets
from utils.log_utils import log_extra 
from utils.fine_tuning import fine_tune


# logging
exp_start = "{:%Y-%b-%d_%H:%M:%S}".format(datetime.datetime.now())
csv_rows=[]

genomic_datasets = list_datasets()
# list_datasets() is not deterministic (resets on lib load)
genomic_datasets.sort()
# skipping multiclass dataset for now
genomic_datasets.remove("human_ensembl_regulatory")
# skipping dataset with sample length longer then 512 chars, for now
genomic_datasets.remove("human_ocr_ensembl")
genomic_datasets.remove("dummy_mouse_enhancers_ensembl")
# TODO remove for debug
# genomic_datasets.remove("demo_coding_vs_intergenomic_seqs")
genomic_datasets.remove("demo_human_or_worm")
genomic_datasets.remove("human_enhancers_cohn")
genomic_datasets.remove("human_enhancers_ensembl")
print(genomic_datasets)

#         config:
# comet_key = 'EpKIINrla6U4B4LJhd9Sv4i0b'
hug_model_link = "simecek/DNADeberta"
epochs = 1
dataset_iterations = 1

# skip comet for now
# os.environ['COMET_API_KEY'] = comet_key

for dataset_name in genomic_datasets:
    if(dataset_name == 'human_ensembl_regulatory'):
        NUM_OF_LABELS = 3
#         TODO what is possitive class for human_ensembl_regulatory (for F1 score)
    else:
        NUM_OF_LABELS = 3
    POSITIVE_CLASS_INDEX = 1
    
    for _ in range(dataset_iterations):
        # comet_ml.init(project_name="DNA_finetuning", api_key=comet_key)
        f1_test, acc_test = fine_tune(hug_model_link, dataset_name, epochs, POSITIVE_CLASS_INDEX, NUM_OF_LABELS)
        
        # log_extra(hug_model_link, dataset_name, f1_test, acc_test)       
        csv_rows.append([dataset_name, f1_test, acc_test])

# compute and log metrics
stand_devs_header = ["dataset", "f1 mean", "f1 sd", "acc mean", "acc sd"]
stand_devs = []
for dataset_index in range(len(genomic_datasets)):
    f_scores = [csv_rows[i][1] for i in range(dataset_iterations)]
    acc_scores = [csv_rows[i][2] for i in range(dataset_iterations)]
    stand_devs.append([
        genomic_datasets[dataset_index], 
        fmean(f_scores), stdev(f_scores), 
        fmean(acc_scores), stdev(acc_scores)
    ])

fields = ['Dataset', 'F1', 'Acc'] 
slash_index = hug_model_link.index("/")
file_name = hug_model_link[:slash_index] + '_' + hug_model_link[slash_index+1:] + '.csv'

with open(file_name, 'a+') as f:
    print("open")
    write = csv.writer(f)  
    write.row(stand_devs_header)
    write.writerows(stand_devs)
    write.writerow(fields)
    write.writerows(csv_rows)
    write.writerow([hug_model_link, exp_start])
    
print('ALL DONE') 


# resursive warning during training - is there a problem?
    #  huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    
