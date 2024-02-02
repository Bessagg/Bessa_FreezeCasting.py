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
import h2o
from h2o.estimators import H2OGradientBoostingEstimator
matplotlib.use('TkAgg')
pd.set_option('display.max_columns', None)

# Load generated df
DataParser = data_parser.DataParser()
df = DataParser.load_complete_data_from_pickle()
df = DataParser.preprocess_drop_not_sublimated(df)

df = df[DataParser.selected_cols_v2]
df = DataParser.preprocess_dropna(df)

opt_save = True
ratios = [0.7, 0.15]  # training ratios [train (valid+test)/2]
seeds = [42]  # 6, 18, 25, 32, 42
r2s = []


for seed in seeds:
    h2o.init(nthreads=-1, min_mem_size_GB=8)
    # Split the dataset into a train and valid set:
    col_dtypes = {'name_part1': 'enum', 'name_part2': 'enum', 'name_fluid1': 'enum', 'name_mold_mat': 'enum',
                  'name_disp_1': 'enum', 'name_bind1': 'enum', 'wf_bind_1': 'numeric',
                  'material_group': 'enum', 'temp_cold': 'numeric', 'cooling_rate': 'numeric', 'time_sub': 'numeric',
                  'time_sinter_1': 'numeric', 'temp_sinter_1': 'numeric', 'vf_total': 'numeric', 'porosity': 'numeric'}

    h2o_data = h2o.H2OFrame(df, destination_frame="CatNum", column_types=col_dtypes)
    train, valid, test = h2o_data.split_frame(ratios, seed=seed)
    train_valid = h2o.H2OFrame.rbind(train, valid)
    train.frame_id = "Train"
    valid.frame_id = "Valid"
    test.frame_id = "Test"
    grid_params = dict()
    grid_params['ntrees'] = [1000]  # Best:4000
    grid_params['learn_rate'] = [0.01]  # Best:0.01
    # grid_params['sample_rate'] = [0.95]  # important
    # grid_params['col_sample_rate_per_tree'] = [1]  # important
    grid_params['min_rows'] = [5]  # important. Best:5
    grid_params['max_depth'] = [15]  # important 15 - 5. Best:5
   # grid_params['nbins'] = [100]  # important
   #grid_params['nbins_cats'] = [500]  # important
    grid_params['seed'] = [seed]
    grid_params['stopping_metric'] = ['deviance']
    grid_params['stopping_rounds'] = [5]  # 50
    grid_params['seed'] = seed
    # grid_params['score_each_iteration'] = ['True']  # not gridable
    model = H2OGradientBoostingEstimator(keep_cross_validation_predictions=True, nfolds=5)
    models_grid = H2OGridSearch(model,
                                hyper_params=grid_params)

    print("Training")
    X = df[df.columns.drop('porosity')].columns.values.tolist()
    y = "porosity"
    models_grid.train(x=X,
                      y=y,
                      training_frame=train_valid)

    grid_sorted = models_grid.get_grid(sort_by='mean_residual_deviance', decreasing=False)
    print("Getting best model")
    # best_model = grid_sorted[0]
    best_model = h2o.get_model(grid_sorted[0].model_id)
    # print("Sorted Grid", grid_sorted)

    r2, mae, mrd = fun.best_model_results(best_model, test, train)
    best_model.plot()
    plt.savefig(f'images/results/train_plot/GBM_{seed}', bbox_inches='tight')

    print("Max depth:", best_model.actual_params['max_depth'])
    print("min rows:", best_model.actual_params['min_rows'])

    # best_model.learning_curve_plot()
    now = datetime.datetime.now().strftime("%y%m%d%H%M")
    model_name = f"GBM_{now}_{seed}_{r2}_{mae}_{mrd}"
    if opt_save:
        h2o.save_model(best_model, path="temp/best_GBM_model", filename=model_name, force=True)
    print(best_model.actual_params)
    fun.save_varimps(best_model, model_name)
    print("R2 : ", r2, "Seed", seed)
    r2s.append(float(r2))

df_r2 = pd.DataFrame(r2s)
print(best_model.actual_params)
print('Mean all r2s', df_r2.mean())
print(df_r2)

