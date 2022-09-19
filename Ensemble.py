# Enables inline-plot rendering
from h2o.grid.grid_search import H2OGridSearch
from functions.functions import best_model_results
import matplotlib.pyplot as plt
import datetime
import pandas as pd

pd.set_option('display.max_columns', None)

# Load generated df
df = pd.read_pickle('freeze_casting_df_v4.0.pkl')

import h2o
from h2o.estimators import H2OStackedEnsembleEstimator
h2o.init(nthreads=-1, min_mem_size_GB=8)

gbm = h2o.load_model('selected_models/GBM/GBM_2209171444_18_0.6395_0.0789_0.0142')
drf = h2o.load_model('selected_models/DRF/DRF_2209171340_42_0.6664_0.0848_0.0124')
dle = h2o.load_model('selected_models/DLE/DLE_2209171646_6_0.6502_0.0898_0.0898')

for seed in [6, 18, 25, 34, 42]:
    h2o.init(nthreads=-1, min_mem_size_GB=8)
    # Split the dataset into a train and valid set:
    h2o_data = h2o.H2OFrame(df, destination_frame="CatNum")
    train, valid, test = h2o_data.split_frame([0.7, 0.15], seed=seed)
    train.frame_id = "Train"
    valid.frame_id = "Valid"
    test.frame_id = "Test"

    ensemble = H2OStackedEnsembleEstimator(model_id="my_ensemble_binomial",
                                           base_models=[gbm, drf, dle])

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

