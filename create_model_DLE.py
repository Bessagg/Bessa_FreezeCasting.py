from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators import H2ODeepLearningEstimator
from helpers.functions import best_model_results
import matplotlib.pyplot as plt
import datetime
import time
import pandas as pd
import data_parser
import h2o

# Load generated df
DataParser = data_parser.DataParser()
df = DataParser.load_complete_data_from_pickle()
# df = df[DataParser.selected_cols_reduced]
df = df[DataParser.selected_cols]
df = DataParser.preprocess_dropna(df)
opt_save = True
ratios = [0.7, 0.15]  # training ratios [train (valid+test)/2]
seeds = [6, 18, 25, 32, 42]  # 6, 18, 25, 32, 42
r2s = []

start = time.time()
h2o.init(nthreads=-1, min_mem_size="8g")
# Split the dataset into a train and valid set:
h2o_data = h2o.H2OFrame(df, destination_frame="CatNum")


cv = True

for seed in seeds:
    train, test, valid = h2o_data.split_frame(ratios, seed=seed)
    train.frame_id = "Train"
    valid.frame_id = "Valid"
    test.frame_id = "Test"
    grid_params = dict()
    grid_params['hidden'] = [[100, 50, 25]]  # [50, 25, 10] [100, 50, 25] [400, 200, 100] [800,400,200],[128, 64, 32, 16,4]
    grid_params['epochs'] = [800]  # 1000
    grid_params['activation'] = ['Rectifier']  # 'TanhWithDropout', 'RectifierWithDropout'
    # grid_params['tweedie_power'] = [1.2]
    # grid_params['score_interval'] = [5.0, 3.0, 10.0]
    #grid_params['l1'] = [1e-6]  #, 5e-7, 1e-7]
    #grid_params['l2'] = [1e-6]  #, 1e-7, 5e-6]
    # grid_params['input_dropout_ratio'] = [0.1, 0.2]
    grid_params['rho'] = [0.99]
    grid_params['loss'] = ['Absolute']  # 'Quadratic', 'Huber'
    grid_params['reproducible'] = [False]  # False
    grid_params['seed'] = [seed]
    grid_params['stopping_rounds'] = [20]  # 20
    #grid_params['variable_importances'] = [True]

    rnn_grid = H2OGridSearch(model=H2ODeepLearningEstimator(standardize=True, seed=seed),
                             hyper_params=grid_params)
    print("Training")
    X = df[df.columns.drop('porosity')].columns.values.tolist()
    y = "porosity"
    rnn_grid.train(x=X,
                   y=y,
                   training_frame=train,
                   validation_frame=valid)
    print("Importance results")
    # drf_grid.show()
    grid_sorted = rnn_grid.get_grid(sort_by="mean_residual_deviance", decreasing=False)
    print("Getting best model")
    # best_model = grid_sorted[0]
    best_model = h2o.get_model(grid_sorted[0].model_id)
    # best_model.keep_cross_validation_predictions = True

    r2, mae, mrd = best_model_results(best_model, test, train)
    best_model.plot()
    plt.savefig(f'images/results/train_plot/DLE_{seed}', bbox_inches='tight')

    now = datetime.datetime.now().strftime("%y%m%d%H%M")
    h2o.save_model(best_model, path="temp/best_DLE_model", filename=f"DLE_{now}_{seed}_{r2}_{mae}_{mrd}", force=True)

    print("Elapsed {:.04f} minutes".format((time.time() - start)/60))
    print("hidden", best_model.actual_params['hidden'])
    print("epochs:", best_model.actual_params['epochs'])
    print("dropout:", best_model.actual_params['input_dropout_ratio'])
    print("l1:", best_model.actual_params['l1'])
    print("l2:", best_model.actual_params['l2'])
    r2s.append(float(r2))

print("Elapsed {:.04f} minutes".format((time.time() - start)/60))
df_r2 = pd.DataFrame(r2s)
print(best_model.actual_params)
print('Mean all r2s', df_r2.mean())
print(df_r2)
