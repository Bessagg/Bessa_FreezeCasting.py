import datetime
import sys

from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators import H2ORandomForestEstimator
from helpers.functions import best_model_results
import matplotlib.pyplot as plt
import datetime
import time
import pickle
import pandas as pd
import data_parser
import h2o

# Load generated df
DataParser = data_parser.DataParser()
df = DataParser.load_complete_data_from_pickle()
df = df[DataParser.selected_cols]
df = DataParser.preprocess_dropna(df)
opt_save = True
seeds = [42]
r2s = []
h2o.init(nthreads=-1, min_mem_size_GB=10)

col_dtypes = {'name_part1': 'enum', 'name_part2': 'enum', 'name_fluid1': 'enum', 'name_mold_mat': 'enum',
              'name_disp_1': 'enum', 'name_bind1': 'enum', 'wf_bind_1': 'numeric',
              'material_group': 'enum', 'temp_cold': 'numeric', 'cooling_rate': 'numeric', 'time_sub': 'numeric',
              'time_sinter_1': 'numeric', 'temp_sinter_1': 'numeric', 'vf_total': 'numeric', 'porosity': 'numeric'}

h2o_data = h2o.H2OFrame(df, destination_frame="CatNum", column_types=col_dtypes)


for seed in seeds:
    start = time.time()


    train, test, valid = h2o_data.split_frame([0.7, 0.15], seed=seed)
    train_valid = h2o.H2OFrame.rbind(train, valid)
    train.frame_id = "Train"
    valid.frame_id = "Valid"
    test.frame_id = "Test"

    grid_params = dict()
    grid_params['ntrees'] = [120]   # 120
    grid_params['max_depth'] = [20]  # [5, 10, 20, 40, 50]  # Best:30
    grid_params['min_rows'] = [5]  # 5, 10, 20, 40, 50  # Best:10
    #grid_params['nbins'] = [100]  # 32
    # grid_params['nbins_cats'] = [100]  # important
    grid_params['seed'] = [seed]
    grid_params['sample_rate'] = [1]  # 0.99 important
    grid_params['col_sample_rate_per_tree'] = [1]  # 1 important
    grid_params['stopping_rounds'] = [5]  #
    # grid_params['stopping_tolerance'] = [0.001]

    drf_grid = H2OGridSearch(model=H2ORandomForestEstimator(keep_cross_validation_predictions=True, nfolds=5),
                             hyper_params=grid_params)
    print("Training")
    X = df[df.columns.drop('porosity')].columns.values.tolist()
    y = "porosity"
    drf_grid.train(x=X,
                   y=y,
                   training_frame=train_valid,
                   #validation_frame=valid
                   )
    print("Importance results")
    # drf_grid.show()
    grid_sorted = drf_grid.get_grid(sort_by='mean_residual_deviance', decreasing=False)
    print("Getting best model")
    # best_model = grid_sorted[0]
    best_model = h2o.get_model(grid_sorted[0].model_id)
    best_model.keep_cross_validation_predictions = True

    r2, mae, mrd = best_model_results(best_model, test, train)
    # best_model.plot()
    plt.savefig(f'images/results/train_plot/DRF_{seed}', bbox_inches='tight')

    print(best_model.actual_params['max_depth'])
    print(best_model.actual_params['min_rows'])
    print(best_model.actual_params['sample_rate'])
    print(best_model.actual_params['col_sample_rate_per_tree'])
    print(best_model.actual_params['ntrees'])
    print(best_model.actual_params)

    now = datetime.datetime.now().strftime("%y%m%d%H%M")
    h2o.save_model(best_model, path="temp/best_DRF_model", filename=f"DRF_{now}_{seed}_{r2}_{mae}_{mrd}", force=True)
    h2o.cluster().shutdown()
    time.sleep(10)
    r2s.append(float(r2))

df_r2 = pd.DataFrame(r2s)
print(best_model.actual_params)
print('Mean all r2s', df_r2.mean())
print(df_r2)

