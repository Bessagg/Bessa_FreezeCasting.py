# Enables inline-plot rendering
# Utilized to create and work with dataframes
import sys
import time
from IPython.display import Image
import database2dataframe
import plotly.io as pio
pio.renderers.default = "browser"
# Analysis with df
df = database2dataframe.db_to_df().copy()
print("Rows:", len(df))
import h2o

h2o.init()
# Split the dataset into a train and valid set:
h2o_data = h2o.H2OFrame(df, destination_frame="CatNum")
train, valid, test = h2o_data.split_frame([0.6, 0.2], seed=1234)
train.frame_id = "Train"
valid.frame_id = "Valid"
test.frame_id = "Test"
from h2o.automl import H2OAutoML
import h2o.estimators
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator

aml = H2OAutoML(max_models=25, seed=1, nfolds=10, stopping_metric='AUTO')
print("Training")
X = df[df.columns.drop('porosity')].columns.values.tolist()
y = "porosity"
aml.train(x=X, y=y,
          training_frame=train,
          validation_frame=valid)
print("Leaderboard")
print(aml.leaderboard)


best_model = aml.get_best_model()
test_y = aml.leader.predict(test)
test_y = test_y.as_data_frame()

print("Testing best model")
print(best_model.model_performance(test))
print("Validating")
print(best_model.model_performance(valid))
print(best_model.show())
h2o.save_model(best_model, path="temp/AutoML_model", force=True)

# com filter treino: 0.084    stackedensemble
# com filter mae 0.84 no test, 0.56r^2  | train : 0.63 e 0.75 r^2
# sem dropar  0.0816 0.65 r^2 valid e 0.0637 no treino r^0.8 no treino
# sem dropar 0.0816 0.65 r^2 | 0.637 e 0.8r^2
# 50 modelos 0.814 0.648r^2 | 0.64 e 0.8r^2

# Without pore_structure:
# com freezing temp, e direction 0.084 e 0.64R^2 | 0.0699 e 0.78R^2
# com freezing temp: 0.0823 e 0.64r^2 e 0.65 e 0.8R^2
# both dropped: 0.0821 e 0.66R^2 e 0.7 e 0.775R^2
