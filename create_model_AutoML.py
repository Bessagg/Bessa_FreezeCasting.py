import database2dataframe
import datetime
import time
import pandas as pd
import data_parser


# Load generated df
DataParser = data_parser.DataParser()
df = DataParser.load_complete_data_from_pickle()
df = df[DataParser.selected_cols]
opt_save = False

# Automl opts
seeds = [6, 25]
prep_name = 'complete'

max_models = 10
seed_t = 42  # [6, 18, 25, 34, 42]:

import h2o
h2o.init()
# Split the dataset into a train and valid set:
h2o_data = h2o.H2OFrame(df, destination_frame="CatNum")
train, valid, test = h2o_data.split_frame([0.7, 0.15], seed=seed_t)  # no need for validation frame as cross validation is enabled
train.frame_id = "Train"
valid.frame_id = "Valid"
test.frame_id = "Test"
from h2o.automl import H2OAutoML
import h2o.estimators
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
start = time.time()
aml = H2OAutoML(max_models=max_models, seed=seed_t, stopping_metric='AUTO')
print("Training")
X = df[df.columns.drop('porosity')].columns.values.tolist()
y = "porosity"
aml.train(x=X, y=y,
          training_frame=train,
          validation_frame=valid)
time.sleep(5)
best_model = aml.get_best_model(criterion="deviance")

df_data = pd.DataFrame()
for seed in [6, 18, 25, 34, 42]:
    data = dict()
    data['seed'] = seed
    train, test, valid = h2o_data.split_frame([0.7, 0.15], seed=seed)
    r2 = best_model.model_performance(test_data=test)['r2']
    mae = best_model.model_performance(test_data=test)['mae']
    mrd = best_model.model_performance(test_data=test)['mean_residual_deviance']
    data['r2'], data['mae'], data['mrd'] = r2, mae, mrd
    new_row = df_data.from_dict([data])
    df_data = pd.concat([df_data, new_row], ignore_index=True)
    now = datetime.datetime.now().strftime("%y%m%d%H%M")
    r2, mae, mrd = "{:.04f}".format(r2), "{:.04f}".format(mae), "{:.04f}".format(mrd)
    h2o.save_model(best_model, path="temp/AutoML_model", filename=f"AutoML_{now}_{seed}_{r2}_{mae}_{mrd}", force=True)

print(df_data)
r2, mae, mrd = df_data['r2'].mean(), df_data['mae'].mean(), df_data['mrd'].mean()
r2, mae, mrd, diff = "{:.04f}".format(r2), "{:.04f}".format(mae), "{:.04f}".format(mrd),\
                     "{:.04f}".format(best_model.r2() - r2)

print("Mean residual deviance: ", mrd)
print("Mean absolut error: ", mae)
print("Pearson Coefficient R^2: ", r2)
print("Difference of r^2 between test and train: ", diff)

now = datetime.datetime.now().strftime("%y%m%d%H%M")
# h2o.save_model(best_model, path="temp/AutoML_model", filename=f"AutoML_{now}_{seed}_{r2}_{mae}_{mrd}", force=True)
print("Elapsed {:.04f} minutes".format((time.time() - start)/60))
print(best_model.base_models)
# h2o.cluster().shutdown()

#
