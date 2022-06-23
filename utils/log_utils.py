import comet_ml

def log_extra(model_name, dataset_name, f1_test, acc_test):
    current_experiment = comet_ml.get_global_experiment()#
    afterlog_experiment = comet_ml.ExistingExperiment(previous_experiment=current_experiment.get_key())
    exp_name = f"model={model_name}_dataset={dataset_name}_:_"
    old_name = afterlog_experiment.get_name()
    afterlog_experiment.set_name(exp_name + old_name)
    afterlog_experiment.log_parameters({
        'model_name': model_name,
        'dataset_name': dataset_name,
    })
    afterlog_experiment.log_metric("test F1", f1_test)
    afterlog_experiment.log_metric("test acc", acc_test)
    afterlog_experiment.add_tag(model_name)
    afterlog_experiment.add_tag(dataset_name)
    afterlog_experiment.end()