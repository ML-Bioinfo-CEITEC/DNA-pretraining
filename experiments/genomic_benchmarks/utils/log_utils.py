import comet_ml


# TODO investigate the "COMET ERROR: Unknown exception happened in Experiment.log_metric; ignoring"
def log_extra(model_name, dataset_name, f1_test, acc_test):
    current_experiment = comet_ml.get_global_experiment()#
    afterlog_experiment = comet_ml.ExistingExperiment(previous_experiment=current_experiment.get_key())
    exp_name = f"{model_name}_{dataset_name}_"
    old_name = afterlog_experiment.get_name()
    afterlog_experiment.set_name(exp_name + old_name)
    afterlog_experiment.log_parameters({
        'model_name': model_name,
        'dataset_name': dataset_name,
    })
    afterlog_experiment.log_metric("test_F1", f1_test)
    afterlog_experiment.log_metric("test_Acc", acc_test)
    afterlog_experiment.add_tag(model_name)
    afterlog_experiment.add_tag(dataset_name)
    afterlog_experiment.end()