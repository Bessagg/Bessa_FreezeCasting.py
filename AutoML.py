import database2dataframe
import datetime
import time
import pandas as pd
# Analysis with df
# import database2dataframe
# df = database2dataframe.db_to_df().copy()

df = pd.read_pickle('freeze_casting_df.pkl')
# print("Used columns:", df.columns)
df.head()
max_models = 50
seed = 42  # [6, 18, 25, 34, 42]:


import h2o

h2o.init()
# Split the dataset into a train and valid set:
h2o_data = h2o.H2OFrame(df, destination_frame="CatNum")
train, valid, test = h2o_data.split_frame([0.7, 0.15], seed=seed)  # no need for validation frame as cross validation is enabled
train.frame_id = "Train"
valid.frame_id = "Valid"
test.frame_id = "Test"
from h2o.automl import H2OAutoML
import h2o.estimators
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
start = time.time()
aml = H2OAutoML(max_models=max_models, seed=seed, stopping_metric='AUTO')
print("Training")
X = df[df.columns.drop('porosity')].columns.values.tolist()
y = "porosity"
aml.train(x=X, y=y,
          training_frame=train,
          validation_frame=valid)
time.sleep(5)
best_model = aml.get_best_model(criterion="deviance")

best_model.show()
r2 = best_model.model_performance(test_data=test)['r2']
mae = best_model.model_performance(test_data=test)['mae']
print("Mean residual deviance")
mrd = best_model.model_performance(test_data=test)['mean_residual_deviance']

r2, mae, mrd, diff = "{:.04f}".format(r2), "{:.04f}".format(mae), "{:.04f}".format(mrd),\
                     "{:.04f}".format(best_model.r2() - r2)

print("Mean residual deviance: ", mrd)
print("Mean average error: ", mae)
print("Pearson Coefficient R^2: ", r2)
print("Difference of r^2 between test and train: ", diff)

now = datetime.datetime.now().strftime("%y%m%d%H%M")
h2o.save_model(best_model, path="temp/AutoML_model", filename=f"AutoML_{now}_{seed}_{r2}_{mae}_{mrd}", force=True)
print("Elapsed {:.04f} minutes".format((time.time() - start)/60))
print(best_model.base_models)

#
