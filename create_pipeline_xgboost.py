import xgboost as xgb
import pandas as pd
import os
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, median_absolute_error, max_error, \
    r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import time
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

"""Parameters"""
# Load generated df
pickle_version = 4
pickle_version = "{:02d}".format(pickle_version)
df = pd.read_pickle(f'freeze_casting_df_v{pickle_version}.pkl')
target = 'porosity'
used_cols = df.columns
current_dir = os.curdir
avg_porosity = df[target].mean()
seed = 6
start = time.time()

"""Train/Val frames"""
train_valid, test = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True)
train, valid = train_test_split(train_valid, test_size=0.2, random_state=seed, shuffle=True)

train_X, train_y = train.drop([target], axis=1), train[target]
valid_X, valid_y = valid.drop([target], axis=1), valid[target]
test_X, test_y = test.drop([target], axis=1), test[target]

"""Preprocessor"""
# Get columns
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
skl_num_cols = df.select_dtypes(include=numerics).columns.to_list()  # order matters
skl_cat_cols = df.drop(skl_num_cols, axis=1).columns.to_list()

encoder = OneHotEncoder(handle_unknown='ignore',  # handle_unkown: 'ignore' or , ‘infrequent_if_exist’
                        min_frequency=0.01,
                        # min_frequency: if value is int categories with a smaller cardinality will be considered infrequent. If float categories with a smaller cardinality than min_frequency * n_samples will be considered infrequent.
                        sparse=False)

# Define custom transformer to leave numerical columns unchanged

# column transformer to combine the numerical and categorical pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('ignore', 'passthrough', skl_num_cols),  # numerical columns
        ('cat', encoder, skl_cat_cols)  # categorical columns
    ],
    remainder='passthrough'
)

# preprocessor.fit(df)
# cat_cols_out = (preprocessor.transformers_[1][1].get_feature_names_out()).tolist()  # pickle_version.get_feature_names_out().tolist()
#
# # Save All and Used columns
# all_columns = skl_num_cols + cat_cols_out
# used_columns = skl_num_cols + cat_cols_out


estimator = xgb.XGBRegressor(n_estimators=400,  # 400 default
                             # subsample=0.5, colsample_bytree=0.5, min_child_weight=5, learning_rate=0.1,
                             # reg_alpha=5,  # 5
                             # reg_lambda=3,
                             # early_stopping_rounds=100,
                             n_jobs=-1,
                             # objective='mae',
                             # eval_metric='deviance',
                             # cv=3, scoring=1)
                             )

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', xgb)
])

param_grid = {
    'model__n_estimators': [400, 1000],
}

grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='rmse')

"""Fit model"""
print("\n Fitting ...")
grid.fit(train_X, train_y, eval_set=[(valid_X, valid_y)], verbose=0)
print("Elapsed: to fit", (time.time() - start) / 60, "mins")

"""Get results"""
results = estimator.evals_result()
for X, y in [[train_X, train_y], [valid_X, valid_y], [test_X, test_y]]:
    preds = estimator.predict(X)
    mape = "{:.02f}".format(mean_absolute_percentage_error(y, estimator.predict(X)))
    rmse = "{:.02f}".format(mean_squared_error(y, estimator.predict(X)))
    mdae = "{:.02f}".format(median_absolute_error(y, estimator.predict(X)))
    maxe = "{:.02f}".format(max_error(y, estimator.predict(X)))
    r2 = "{:.02f}".format(r2_score(y, estimator.predict(X)))
    rmse2 = "{:.02f}".format(r2_score(y, estimator.predict(X)))
    print(r2)

import shap

explainer = shap.TreeExplainer(estimator)
X = train_X.copy()
shap_values = explainer.shap_values(X)
plt_shap = shap.summary_plot(shap_values, features=X, feature_names=X.columns, plot_size=(18, 8),
                             max_display=X.shape[1])
