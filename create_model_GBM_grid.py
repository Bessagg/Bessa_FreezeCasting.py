# Enables inline-plot rendering
from h2o.grid.grid_search import H2OGridSearch
import matplotlib
matplotlib.use('TkAgg')
import functions
import helpers.functions as fun
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import data_parser
matplotlib.use('TkAgg')
pd.set_option('display.max_columns', None)

# Load generated df
DataParser = data_parser.DataParser()
df = DataParser.load_complete_data_from_pickle()
df = df[DataParser.selected_cols]
opt_save = False
seeds = [6, 25]
prep_name = 'complete'

import h2o
from h2o.estimators import H2OGradientBoostingEstimator

for seed in seeds:
    h2o.init(nthreads=-1, min_mem_size_GB=8)
    # Split the dataset into a train and valid set:
    h2o_data = h2o.H2OFrame(df, destination_frame="CatNum")
    train, valid, test = h2o_data.split_frame([0.7, 0.15], seed=seed)
    train.frame_id = "Train"
    valid.frame_id = "Valid"
    test.frame_id = "Test"
    grid_params = dict()
    grid_params['ntrees'] = [3000]
    grid_params['learn_rate'] = [0.0005]  # 0.01
    # grid_params['sample_rate'] = [0.95]  # important
    # grid_params['col_sample_rate_per_tree'] = [1]  # important
    grid_params['min_rows'] = [10]  # important
    grid_params['max_depth'] = [10]  # important 15 - 5
    grid_params['nbins'] = [10]  # important
    # grid_params['nbins_cats'] = [75]  # important
    grid_params['seed'] = [seed]
    grid_params['stopping_metric'] = ['deviance']
    grid_params['stopping_rounds'] = [7]  # 50
    # grid_params['score_each_iteration'] = ['True']  # not gridable

    models_grid = H2OGridSearch(model=H2OGradientBoostingEstimator(),
                                hyper_params=grid_params)
    print("Training")
    X = df[df.columns.drop('porosity')].columns.values.tolist()
    y = "porosity"
    models_grid.train(x=X,
                      y=y,
                      training_frame=train,
                      validation_frame=valid)

    grid_sorted = models_grid.get_grid(sort_by='mean_residual_deviance', decreasing=False)
    print("Getting best model")
    best_model = grid_sorted[0]

    r2, mae, mrd = fun.best_model_results(best_model, test, train)
    best_model.plot()
    plt.savefig(f'images/results/train_plot/GBM_{seed}', bbox_inches='tight')

    print("Max depth:", best_model.actual_params['max_depth'])
    print("min rows:", best_model.actual_params['min_rows'])

    # best_model.learning_curve_plot()
    now = datetime.datetime.now().strftime("%y%m%d%H%M")
    model_name = f"GBM_{now}_{seed}_{r2}_{mae}_{mrd}_v{prep_name}"
    if opt_save:
        h2o.save_model(best_model, path="temp/best_GBM_model", filename=model_name, force=True)
    print(best_model.actual_params)
    fun.save_varimps(best_model, model_name)
    print("R2 : ", r2, "Seed", seed)





