import comet_ml
import os
import csv
import datetime
import numpy as np
from math import sqrt
from statistics import stdev, fmean
from genomic_benchmarks.data_check import list_datasets
from utils.finetuning import fine_tune
from utils.log_utils import log_extra

# TODO omezit/snizit desatinou přesnost na F1 a acc

###       
# config:
###
hug_model_link = "simecek/DNADeberta"
epochs = 4
dataset_iterations = 4
###

# logging
csv_rows=[]
slash_index = hug_model_link.index("/")
file_name = hug_model_link[:slash_index] + '_' + hug_model_link[slash_index+1:]
exp_start = "{:%Y-%b-%d_%H:%M:%S}".format(datetime.datetime.now())


genomic_datasets = list_datasets()
# list_datasets() is not deterministic (resets on lib load)
genomic_datasets.sort()
# skipping multiclass dataset for now
# genomic_datasets.remove("human_ensembl_regulatory")
# skipping dataset with sample length longer then 512 chars, for now
genomic_datasets.remove("human_ocr_ensembl")
genomic_datasets.remove("dummy_mouse_enhancers_ensembl")
print(genomic_datasets)


comet_key = 'EpKIINrla6U4B4LJhd9Sv4i0b'
os.environ['COMET_API_KEY'] = comet_key
comet_ml.init(project_name="DNA_finetuning", api_key=comet_key)
# os.environ['COMET_MODE'] = 'DISABLED'

def debug_export(to_print):
    with open('debug.csv', 'a+') as f:
        print("open")
        write = csv.writer(f)
        write.writerow(to_print)
        print("______")
        print(to_print)
        print("______")

        
for dataset_name in genomic_datasets:
    if(dataset_name == 'human_ensembl_regulatory'):
        NUM_OF_LABELS = 3
#         TODO what is possitive class for human_ensembl_regulatory (for F1 score)
    else:
        NUM_OF_LABELS = 2
    POSITIVE_CLASS_INDEX = 1
    
    for _ in range(dataset_iterations):
        f1_test, acc_test = fine_tune(hug_model_link, dataset_name, epochs, POSITIVE_CLASS_INDEX, NUM_OF_LABELS)
        
        debug_export(to_print = [dataset_name, f1_test, acc_test])
        csv_rows.append([dataset_name, f1_test, acc_test])
        log_extra(file_name, dataset_name, f1_test, acc_test)      

# compute and log metrics
if(dataset_iterations > 1):
    stats_header = ["Dataset", "F1 mean", "F1 SEM", "Acc mean", "Acc SEM"]
    stats = []
    for dataset_index in range(len(genomic_datasets)):
        f_scores = []
        acc_scores = []
        for i in range(dataset_iterations):
            row_index = dataset_index * dataset_iterations + i
            f_scores.append(csv_rows[row_index][1])
            acc_scores.append(csv_rows[row_index][2])
        stats.append([
            genomic_datasets[dataset_index], 
            fmean(f_scores), stdev(f_scores)/sqrt(len(f_scores)), 
            fmean(acc_scores), stdev(acc_scores)/sqrt(len(acc_scores)),
        ])


fields = ['Dataset', 'F1', 'Acc'] 
exp_end = "{:%Y-%b-%d_%H:%M:%S}".format(datetime.datetime.now())

with open(file_name + '.csv', 'a+') as f:
    print("open")
    write = csv.writer(f)  
    if(dataset_iterations > 1):
        write.writerow(stats_header)
        write.writerows(stats)
    write.writerow(fields)
    write.writerows(csv_rows)
    write.writerow([hug_model_link, exp_start, exp_end])
    
print(csv_rows)
    
print('ALL DONE') 