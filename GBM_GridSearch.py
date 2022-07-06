# Enables inline-plot rendering

import database2dataframe
import plotly.io as pio
import subprocess
from h2o.grid.grid_search import H2OGridSearch
import datetime
import pandas as pd
pd.set_option('display.max_columns', None)
pio.renderers.default = "browser"

# Analysis with df
df = database2dataframe.db_to_df().copy()
print("Rows:", len(df))

# H20 DRF - Distributed Random Forest
import h2o
from h2o.estimators import H2OGradientBoostingEstimator

h2o.init(nthreads=-1, min_mem_size_GB=20)
# Split the dataset into a train and valid set:
h2o_data = h2o.H2OFrame(df, destination_frame="CatNum")
train, valid, test = h2o_data.split_frame([0.7, 0.15], seed=1234)
train.frame_id = "Train"
valid.frame_id = "Valid"
test.frame_id = "Test"

grid_params = dict()
grid_params['ntrees'] = [100, 600]
grid_params['learn_rate'] = [0.01]
grid_params['sample_rate'] = [1, 0.9, 0.95]  # important
grid_params['col_sample_rate_per_tree'] = [1, 0.9]  # important
# grid_params['balance_classes'] = [True, False]
grid_params['min_rows'] = [20, 40]  # important
grid_params['nbins'] = [20, 80]
grid_params['max_depth'] = [0, 20, 40]
grid_params['seed'] = [1234]
grid_params['tweedie_power'] = [1.9]
grid_params['stopping_metric'] = ['deviance']
grid_params['stopping_rounds'] = [6, 9, 12]
# grid_params['score_each_iteration'] = ['True']  # not gridable

models_grid = H2OGridSearch(model=H2OGradientBoostingEstimator(nfolds=5),
                            hyper_params=grid_params)
print("Training")
X = df[df.columns.drop('porosity')].columns.values.tolist()
y = "porosity"
models_grid.train(x=X,
                  y=y,
                  training_frame=train,
                  validation_frame=valid)

grid_sorted = models_grid.get_grid(sort_by='r2', decreasing=True)
print("Getting best model")
best_model = grid_sorted[0]
pred_test = best_model.predict(test)
pred_valid = best_model.predict(valid)
best_model.show()
r2, r2_train = best_model.r2(valid=True), best_model.r2()
r2, r2_train = "{:.04f}".format(r2), "{:.04f}".format(r2_train)

print(f"R2: train {best_model.r2()} \n valid {best_model.r2(valid=True)} \n "
      f"diff {best_model.r2() - best_model.r2(valid=True) }")


print("R2 and mae", r2, best_model.mae(valid=True))
best_model.learning_curve_plot()
now = datetime.datetime.now().strftime("%y%m%d%H%M")
h2o.save_model(best_model, path="temp/best_GBM_model", filename=f"GBM_{r2}_{r2_train}_{now}", force=True)

# With temp_cold r2: train 0.799 e valid = 0.609
# New: 0.0624

"""
{'model_id': None,
 'training_frame': None,
 'validation_frame': None,
 'nfolds': 0,
 'keep_cross_validation_models': True,
 'keep_cross_validation_predictions': False,
 'keep_cross_validation_fold_assignment': False,
 'score_each_iteration': False,
 'score_tree_interval': 0,
 'fold_assignment': 'AUTO',
 'fold_column': None,
 'response_column': None,
 'ignored_columns': None,
 'ignore_const_cols': True,
 'offset_column': None,
 'weights_column': None,
 'balance_classes': False,
 'class_sampling_factors': None,
 'max_after_balance_size': 5.0,
 'max_confusion_matrix_size': 20,
 'ntrees': 50,
 'max_depth': 20,
 'min_rows': 1.0,
 'nbins': 20,
 'nbins_top_level': 1024,
 'nbins_cats': 1024,
 'r2_stopping': 1.7976931348623157e+308,
 'stopping_rounds': 0,
 'stopping_metric': 'AUTO',
 'stopping_tolerance': 0.001,
 'max_runtime_secs': 0.0,
 'seed': -1,
 'build_tree_one_node': False,
 'mtries': -1,
 'sample_rate': 0.632,
 'sample_rate_per_class': None,
 'binomial_double_trees': False,
 'checkpoint': None,
 'col_sample_rate_change_per_level': 1.0,
 'col_sample_rate_per_tree': 1.0,
 'min_split_improvement': 1e-05,
 'histogram_type': 'AUTO',
 'categorical_encoding': 'AUTO',
 'calibrate_model': False,
 'calibration_frame': None,
 'distribution': 'AUTO',
 'custom_metric_func': None,
 'export_checkpoints_dir': None,
 'check_constant_response': True,
 'gainslift_bins': -1,
 'auc_type': 'AUTO'}

"""
