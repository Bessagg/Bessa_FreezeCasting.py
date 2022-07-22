import datetime
import sys

from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators import H2ODeepLearningEstimator
import datetime
import time
import pickle
import pandas as pd

# Analysis with df
# import database2dataframe
# df = database2dataframe.db_to_df().copy()

df = pd.read_pickle('freeze_casting_df.pkl')
print("Used columns:", df.columns)

# pore_structure_filter = df['pore_structure'].value_counts().head(5).axes[0]
# df = df[df['pore_structure'].isin(pore_structure_filter)]

# H20 DRF - Distributed Random Forest
import h2o
start = time.time()
print("Rows:", len(df))
h2o.init(nthreads=-1, min_mem_size="4g")
# Split the dataset into a train and valid set:
h2o_data = h2o.H2OFrame(df, destination_frame="CatNum")
seed = 25  # 6 18 25 34 42
train, test, valid = h2o_data.split_frame([0.75, 0.125], seed=seed)
train.frame_id = "Train"
valid.frame_id = "Valid"
test.frame_id = "Test"
cv = True

grid_params = dict()
grid_params['hidden'] = [[200, 100, 50, 25, 4]]  # 0.58 e 0.50
grid_params['hidden'] = [[200, 100, 50]]  # 0.60 e 0.52
grid_params['hidden'] = [[200, 100, 100]]  # 0.66 e 0.59
grid_params['hidden'] = [[400, 200, 100]]  # 0.615 e 0.54
grid_params['hidden'] = [[100, 100, 50]]  # 0.638 e 0.53
grid_params['hidden'] = [[200, 100, 50]]  # 0.613 e 0.535
grid_params['hidden'] = [[100, 100, 100]]  # 0.649 0.55
grid_params['hidden'] = [[300, 100, 100]]  # 0.663 0.548
grid_params['hidden'] = [[300, 200, 100]]  # 0.70 e 0.608
grid_params['hidden'] = [[400, 200, 100]]  # 0.665 0.547
grid_params['hidden'] = [[300, 200, 200]]  # 0.64 0.537
grid_params['hidden'] = [[250, 250, 250]]  # 0.628 0.556
grid_params['hidden'] = [[300, 100, 50]]  # 0.64 0.528
grid_params['hidden'] = [[600, 200, 200]]  # 0.628 0.51
grid_params['hidden'] = [[300, 200, 150]]  # 0.678 0.57
grid_params['hidden'] = [[300, 300, 200]]  # 0.64 0.538
grid_params['hidden'] = [[300, 200, 100]]  # 0.70 e 0.608
grid_params['hidden'] = [[400, 200, 100]]  # 0.656 0.611 - 0.0005l1 e 1-5l2
grid_params['hidden'] = [[600, 300, 150]]  # 0.67 e 0.605
grid_params['hidden'] = [[600, 400, 300]]  # 0.67 e 0.607
grid_params['hidden'] = [[600, 400]]  # 0.69 e 0.62
grid_params['hidden'] = [[800, 600, 200]]  # 0.68 0.60
grid_params['hidden'] = [[1200, 600, 600]]  # 0.69 e 0.62
grid_params['hidden'] = [[1200, 600]]  # 0.68 e 0.623
grid_params['hidden'] = [[1200, 800, 600]]  # 0.7 e 0.63
grid_params['hidden'] = [[1200, 800, 800]]  # 0.71 e 0.64
grid_params['hidden'] = [[1600, 800, 400]]  # 0.7494 e 0.667  500 epochs  l1: 5e-06 l2: 5e-07
grid_params['hidden'] = [[1600, 800, 400, 200]]  # 0.738 e 0.06719
grid_params['hidden'] = [[2400, 1200, 800, 400]]  # 0.75 e 0.67
grid_params['hidden'] = [[2400, 1200, 800]]  # 0.759 e 0.666  5e-6 5e-7 / cv 0.612
grid_params['hidden'] = [[3000, 1500, 750]]  # cv 0.618 e max 0.656 / 860 epochs
# Dropout had no good results
# Only L1 had no good results
# Only l2
grid_params['hidden'] = [[1600, 800, 400, 200]]  # seed = 6 0.76 e 0.576
grid_params['hidden'] = [[400, 200, 100]]  # seed=6 0.77 e 0.587
# L1 & L2
grid_params['hidden'] = [[400, 200, 100]]  # seed=6 0.76 e 0.577, 0.76 e 0.59
grid_params['hidden'] = [[400, 200, 100]]  # seed=6 0.76 e 0.577, 0.76 e 0.59
grid_params['hidden'] = [[64, 32, 16, 4]]  # 0.755 e 0.6133
grid_params['hidden'] = [[128, 64, 32, 16, 4]]  # 0.77 0.562
grid_params['hidden'] = [[1600, 800, 400, 200]]  # seed = 6 0.6947 e 0.5830
grid_params['hidden'] = [[1600, 800, 400, 200]]  # seed = 18 0.7385 e 0.541
grid_params['hidden'] = [[1600, 800, 400, 200]]  # seed = 25 0.749 e 0.574
grid_params['hidden'] = [[100, 100, 50]]  # seed = 25 0.749 e 0.574





#grid_params['hidden'] = [[32, 16, 4]]  #

grid_params['epochs'] = [1000]
grid_params['activation'] = ['Rectifier']  # 'TanhWithDropout', 'RectifierWithDropout'
grid_params['tweedie_power'] = [1.2]
# grid_params['score_interval'] = [5.0, 3.0, 10.0]
grid_params['l1'] = [1e-6]  #, 5e-7, 1e-7]
grid_params['l2'] = [5e-6]  #, 1e-7, 5e-6]
#grid_params['input_dropout_ratio'] = [0.1, 0.2]
grid_params['rho'] = [0.95]
grid_params['loss'] = ['Absolute']  # 'Quadratic', 'Huber'
grid_params['reproducible'] = [False]  # False
grid_params['seed'] = [seed]
grid_params['stopping_rounds'] = 10

rnn_grid = H2OGridSearch(model=H2ODeepLearningEstimator(standardize=True, nfolds=5),
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
grid_sorted = rnn_grid.get_grid(sort_by='r2', decreasing=True)
print("Getting best model")
best_model = grid_sorted[0]

pred_test = best_model.predict(test)
pred_valid = best_model.predict(valid)
best_model.show()

r2, r2_train = best_model.r2(valid=True), best_model.r2()
results = best_model.cross_validation_metrics_summary()

r2, r2_train = "{:.04f}".format(r2), "{:.04f}".format(r2_train)

print(f"R2: train {best_model.r2()} \n valid {best_model.r2(valid=True)} \n "
      f"diff {best_model.r2() - best_model.r2(valid=True) }")
print("R2 and mae", r2, best_model.mae(valid=True))

now = datetime.datetime.now().strftime("%y%m%d%H%M")
h2o.save_model(best_model, path=f"temp/best_RNN_model/RNN_{seed}_{r2}_{r2_train}_{now}", force=True)

print("Elapsed {:.04f} minutes".format((time.time() - start)/60))
print("hidden", best_model.actual_params['hidden'])
print("epochs:", best_model.actual_params['epochs'])
print("dropout:", best_model.actual_params['input_dropout_ratio'])
print("l1:", best_model.actual_params['l1'])
print("l2:", best_model.actual_params['l2'])

