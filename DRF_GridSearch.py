import datetime
import sys

from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators import H2ORandomForestEstimator
from functions.functions import best_model_results
import matplotlib as plt
import datetime
import time
import pickle
import pandas as pd

# Load generated df
df = pd.read_pickle('freeze_casting_df_v3.0.pkl')

import h2o
for seed in [6, 18, 25, 34, 42]:
    start = time.time()
    h2o.init(nthreads=-1, min_mem_size_GB=10)

    # Split the dataset into a train and valid set:
    h2o_data = h2o.H2OFrame(df, destination_frame="CatNum")

    train, test, valid = h2o_data.split_frame([0.7, 0.15], seed=seed)
    train.frame_id = "Train"
    valid.frame_id = "Valid"
    test.frame_id = "Test"

    grid_params = dict()
    grid_params['ntrees'] = [120]   # 120
    grid_params['max_depth'] = [20]  # 20 - 30
    grid_params['min_rows'] = [10]  # 10
    grid_params['nbins'] = [32]  # 32
    grid_params['nbins_cats'] = [100]  # important
    grid_params['seed'] = [seed]
    grid_params['sample_rate'] = [1]  # 0.99 important
    grid_params['col_sample_rate_per_tree'] = [1]  # 1 important
    grid_params['stopping_rounds'] = [20]  #
    # grid_params['stopping_tolerance'] = [0.001]

    drf_grid = H2OGridSearch(model=H2ORandomForestEstimator(),
                             hyper_params=grid_params)
    print("Training")
    X = df[df.columns.drop('porosity')].columns.values.tolist()
    y = "porosity"
    drf_grid.train(x=X,
                   y=y,
                   training_frame=train,
                   validation_frame=valid)
    print("Importance results")
    # drf_grid.show()
    grid_sorted = drf_grid.get_grid(sort_by='mean_residual_deviance', decreasing=False)
    print("Getting best model")
    best_model = grid_sorted[0]
    r2, mae, mrd = best_model_results(best_model, test)
    now = datetime.datetime.now().strftime("%y%m%d%H%M")
    h2o.save_model(best_model, path="temp/best_DRF_model", filename=f"DRF_{now}_{seed}_{r2}_{mae}_{mrd}", force=True)
    #print(best_model.actual_params)
    h2o.cluster().shutdown()
    time.sleep(10)

    # com drop top 5 , mae 0.10 validation e 0,108 train
    # com drop, 0,10 e 0.108
    # sem 0.074 e  0.54R^2  e 0.64 0.67R^2

    # Without pore_structure:
    # com freezing temp, e direction 0.1088 on valid e 0.116 train
    # com freezing temp: 0.1118 e 0.119
    # sem freezing temp 0.115 e 0.12 train
