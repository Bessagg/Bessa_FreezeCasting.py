# Enables inline-plot rendering
from helpers.functions import  get_selected_model_from_each_folder
import pandas as pd
import h2o
from h2o.estimators import H2OStackedEnsembleEstimator
import data_parser
pd.set_option('display.max_columns', None)

# Load generated df
DataParser = data_parser.DataParser()
df = DataParser.load_complete_data_from_pickle()
df = df[DataParser.selected_cols_reduced]
df = DataParser.preprocess_dropna(df)
opt_save = True
seeds = [6, 18, 25, 32, 42]
selected_seed = 32
r2s = []
selected_models_path = "selected_models"


h2o.init(nthreads=-1, min_mem_size_GB=8)

list_model_paths = get_selected_model_from_each_folder()
loaded_models = []
# all models must be trained in the same training frame. Hence, use same train seed.
for path in list_model_paths:
    filename = path.split('\\')[-1]
    seed = int(filename.split('_')[2])
    if seed == selected_seed:
        model = h2o.load_model(path)
        model.seed = seed
        loaded_models.append(model)

for seed in seeds:
    h2o.init(nthreads=-1, min_mem_size_GB=8)
    # Split the dataset into a train and valid set:
    h2o_data = h2o.H2OFrame(df, destination_frame="CatNum")
    train, valid, test = h2o_data.split_frame([0.7, 0.15], seed=seed)
    train.frame_id = "Train"
    valid.frame_id = "Valid"
    test.frame_id = "Test"

    ensemble = H2OStackedEnsembleEstimator(model_id="my_ensemble",
                                           base_models=loaded_models)

    print("Training")
    X = df[df.columns.drop('porosity')].columns.values.tolist()
    y = "porosity"
    ensemble.train(x=X, y=y, training_frame=train)

    # Eval ensemble performance on the test data
    perf_stack_test = ensemble.model_performance(test)
    print("Getting best model")
    # best_model = grid_sorted[0]
    #
    # r2, mae, mrd = best_model_results(best_model, test, train)
    # best_model.plot()
    # plt.savefig(f'images/results/train_plot/GBM_{seed}', bbox_inches='tight')
    #
    # print("Max depth:", best_model.actual_params['max_depth'])
    # print("min rows:", best_model.actual_params['min_rows'])
    #
    # # best_model.learning_curve_plot()
    # now = datetime.datetime.now().strftime("%y%m%d%H%M")
    # h2o.save_model(best_model, path="temp/best_GBM_model", filename=f"GBM_{now}_{seed}_{r2}_{mae}_{mrd}", force=True)
    # print(best_model.actual_params)

