# Enables inline-plot rendering
# Utilized to create and work with dataframes
import sys
import time

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
import os
import pandas as pd
from sklearn.model_selection import learning_curve

# Analysis with df
df = pd.read_pickle('freeze_casting_df.pkl')
X = df[df.columns.drop('porosity')]
y = df['porosity']

# Numeric and Cat separation
num_features, cat_features = X.select_dtypes(include=[float]).columns, X.select_dtypes(include=[object]).columns
df_num = df[num_features]
print("Numerical Nulls", df_num.isna().sum(), df_num.isna().sum()/df_num.count())
df_cat = df[cat_features]
print("Categorical Nulls", df_cat.isnull().sum(), df_cat.isnull().sum()/df_cat.count())

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

data_pipeline = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ]
)

processed_data = data_pipeline.fit_transform(df)


rnn = MLPRegressor()
model = Pipeline(
    steps=[("preprocessor", data_pipeline), ("clf", rnn)])


# model = make_pipeline(data_pipeline, RandomForestRegressor())
seed = 42  # 6 18 25 34 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
param_grid = dict()
param_grid['preprocessor__num__imputer__strategy'] = ["median", "mean"]
param_grid['clf__alpha'] = [5e-6]  # Strength of the L2 regularization term.
param_grid['clf__solver'] = ['adam']  # sgd, adam, lbfgs
param_grid['clf__learning_rate'] = ['constant']
param_grid['clf__activation'] = ['relu']
param_grid['clf__random_state'] = [seed]
param_grid['clf__verbose'] = [1]
param_grid['clf__max_iter'] = [500]
param_grid['clf__n_iter_no_change'] = [30]
param_grid['clf__tol'] = [1e-6]
# param_grid['clf__early_stopping'] = [True, False]

# min - 50 bad results
# min - 25 ok results
# min features = 0, seed = 42
param_grid['clf__hidden_layer_sizes'] = [(1600, 800, 400, 200)]  # 0.69 e 0.6227 # seed 42
param_grid['clf__hidden_layer_sizes'] = [(1600, 800, 400, 200)]  # 0.65 e 0.66 # seed 6
param_grid['clf__hidden_layer_sizes'] = [(1600, 800, 400, 200)]  # 0.77 e 0.58 # seed 18
param_grid['clf__hidden_layer_sizes'] = [(1600, 800, 400, 200)]  # 0.76 e 0.6085 # seed 25
param_grid['clf__hidden_layer_sizes'] = [(1600, 800, 400, 200)]  # 0.7935 e 0.5865 # seed 34
param_grid['clf__hidden_layer_sizes'] = [(1600, 800, 400, 200)]  # 0.75 e 0.60 # seed 34
param_grid['clf__hidden_layer_sizes'] = [(256, 128, 64, 32)]  # 0.7926 e 0.6485 seed 42
param_grid['clf__hidden_layer_sizes'] = [(256, 128, 64, 32)]  # 0.7971 e 0.6810 seed 6
param_grid['clf__hidden_layer_sizes'] = [(256, 128, 64, 32)]  # 0.8031 e 0.5998 seed 18
param_grid['clf__hidden_layer_sizes'] = [(256, 128, 64, 32)]  # 0.7977 e 0.6073 seed 25
param_grid['clf__hidden_layer_sizes'] = [(256, 128, 64, 32)]  # 0.8004 e 0.5865 seed 34
param_grid['clf__hidden_layer_sizes'] = [(256, 128, 64, 32)]  # 0.7926 e 0.6485 seed 34

gs = GridSearchCV(model, param_grid)
print("\n Grid Searching")
start = time.time()
gs.fit(X_train, y_train)
print("GS Score")
print('Test set score: ' + str(gs.score(X_test, y_test)))
print("mins elapsed:", (time.time() - start)/60)

print("Best parameters: \n", gs.best_params_)

model.set_params(**gs.best_params_)
model.fit(X_train, y_train)
r2_train = model.score(X_train, y_train)
r2 = model.score(X_test, y_test)
r2_train, r2 = "{:.04f}".format(r2_train), "{:.04f}".format(r2)

print("Train r^2 score: ", r2_train)
print("Valid r^2 score: ", r2)

import pickle
import datetime
now = datetime.datetime.now().strftime("%y%m%d%H%M")
path = f"temp/best_RNN_SCIT_model/RNN_{seed}_{r2}_{r2_train}_{now}" + ".pkl"
with open(path, 'wb') as file:
    pickle.dump(model, file)

