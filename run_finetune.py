import comet_ml
from utils.log_utils import log_extra 
from utils.fine_tuning import fine_tune
from genomic_benchmarks.data_check import list_datasets

genomic_datasets = list_datasets()
genomic_datasets.sort()
genomic_datasets.remove("human_ocr_ensembl")
print(genomic_datasets)

for dataset_name in genomic_datasets:
    for _ in range(1):
        
#         config:
        model_name = "DNADeberta_fine"
        commet_key = 'EpKIINrla6U4B4LJhd9Sv4i0b'
        hug_model_link = "simecek/DNADeberta"
        epochs = 10
        
        
        os.environ['COMET_API_KEY'] = commet_key
        comet_ml.init(project_name="DNA_finetuning", api_key=commet_key)
        
        f1_test, acc_test = fine_tuning(hug_model_link, model_name, dataset_name, epochs)
        
        log_extra(model_name, dataset_name, f1_test, acc_test)
        

print('ALL DONE')