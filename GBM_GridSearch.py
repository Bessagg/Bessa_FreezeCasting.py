# Enables inline-plot rendering
from h2o.grid.grid_search import H2OGridSearch
from functions.functions import best_model_results
import matplotlib.pyplot as plt
import datetime
import pandas as pd

pd.set_option('display.max_columns', None)

# Load generated df
df = pd.read_pickle('freeze_casting_df_v3.0.pkl')

import h2o
from h2o.estimators import H2OGradientBoostingEstimator

for seed in [6, 18, 25, 34, 42]:
    h2o.init(nthreads=-1, min_mem_size_GB=8)
    # Split the dataset into a train and valid set:
    h2o_data = h2o.H2OFrame(df, destination_frame="CatNum")
    train, valid, test = h2o_data.split_frame([0.7, 0.15], seed=seed)
    train.frame_id = "Train"
    valid.frame_id = "Valid"
    test.frame_id = "Test"
    grid_params = dict()
    grid_params['ntrees'] = [4000]
    grid_params['learn_rate'] = [0.005]  # 0.01
    # grid_params['sample_rate'] = [0.95]  # important
    # grid_params['col_sample_rate_per_tree'] = [1]  # important
    grid_params['min_rows'] = [10]  # important
    grid_params['max_depth'] = [10]  # important 15 - 5
    grid_params['nbins'] = [10]  # important
    grid_params['nbins_cats'] = [75]  # important
    grid_params['seed'] = [seed]
    grid_params['stopping_metric'] = ['deviance']
    grid_params['stopping_rounds'] = [10]  # 50
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

    r2, mae, mrd = best_model_results(best_model, test)
    best_model.plot()
    plt.savefig(f'images/results/train_plot/DRF_{seed}', bbox_inches='tight')

    print("Mean residual deviance: ", mrd)
    print("Mean average error: ", mae)
    print("Pearson Coefficient R^2: ", r2)
    print("Difference of r^2 train - test: ",
          best_model.model_performance(test_data=train)['r2'] - best_model.model_performance(test_data=test)['r2'])
    print("Max depth:", best_model.actual_params['max_depth'])
    print("min rows:", best_model.actual_params['min_rows'])
    best_model.plot()

    # best_model.learning_curve_plot()
    now = datetime.datetime.now().strftime("%y%m%d%H%M")
    h2o.save_model(best_model, path="temp/best_GBM_model", filename=f"GBM_{now}_{seed}_{r2}_{mae}_{mrd}", force=True)
    print(best_model.actual_params)

