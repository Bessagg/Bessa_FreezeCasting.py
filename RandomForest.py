# Enables inline-plot rendering
# Utilized to create and work with dataframes
import sys
import time

from database2dataframe import db_to_df
from sklearn.preprocessing import StandardScaler
import plotly.io as pio
pio.renderers.default = "browser"
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Analysis with df
df = db_to_df().copy()
min_n_rows = 500

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

rf = RandomForestRegressor()
model = Pipeline(
    steps=[("preprocessor", data_pipeline), ("clf", rf)])


# model = make_pipeline(data_pipeline, RandomForestRegressor())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

param_grid = dict()
param_grid["preprocessor__num__imputer__strategy"] = ["mean", "median"]
# param_grid['clf__max_leaf_nodes'] = ["", None]
param_grid['clf__n_estimators'] = [25, 50, 100, 200]
param_grid['clf__max_depth'] = [5, 10, 20, 60, None]
param_grid['clf__min_samples_leaf'] = [2, 4, 6]
param_grid['clf__min_samples_split'] = [2, 4, 8, 32]
param_grid['clf__max_features'] = [None, "sqrt"]
param_grid['clf__random_state'] = [None, 42]

# gs.get_params() to know which parameters to look
from sklearn.model_selection import RandomizedSearchCV
# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
gs = GridSearchCV(model, param_grid, cv=5, return_train_score=True)
print("\n Grid Searching")
start = time.time()
gs.fit(X_train, y_train)
print("GS Score")
print('Training set score: ' + str(gs.score(X_train, y_train)))
print('Test set score: ' + str(gs.score(X_test, y_test)))
print("mins elapsed:", (time.time() - start)/60)

print("Best parameters: \n", gs.best_params_)

model.set_params(**gs.best_params_)
model.fit(X_train, y_train)
print('Training set score: ' + str(model.score(X_train, y_train)))  # 0.724
print('Test set score: ' + str(model.score(X_test, y_test)))  # 0.58
# com drop 0.7 e 0.58
from sklearn.metrics import r2_score
print("Train r^2 score: ", r2_score(y_train, model.predict(X_train)))
print("Train r^2 score: ", r2_score(y_test, model.predict(X_test)))

import pickle
import datetime
now = datetime.datetime.now().strftime("%y%m%d%H%M")

path=f"temp/best_RNN_model/RNN_{r2}_{r2_train}_{now}"
pkl_filename = "Scit_RF_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)


#
# grid_search = GridSearchCV(model, param_grid, cv=10)
# grid_search.fit(X_train, y_train)
# print(f"Internal CV score: {grid_search.best_score_:.3f}")
# cv_results = pd.DataFrame(grid_search.cv_results_)
# cv_results = cv_results.sort_values("mean_test_score", ascending=False)
# cv_results[
#     [
#         "mean_test_score",
#         "std_test_score",
#         "param_preprocessor__num__imputer__strategy",
#         "param_classifier__C",
#     ]].head(5)
#
#
#
# # Splitting data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=3)
#
# # Linear Regression Pipeline
# lr_pipe = Pipeline([('scl', StandardScaler()),
#                     ('clf', LinearRegression())])
#
# lr_pipe.fit(X_train, y_train)
# lr_pipe.fit(X_train, y_train)
#
#
# def validation_curve(clf):
#     test_score = np.empty(len(clf.estimators_))
#     train_score = np.empty(len(clf.estimators_))
#
#     for i, pred in enumerate(clf.staged_predict_proba(X_test)):
#         test_score[i] = 1 - roc_auc_score(y_test, pred[:, 1])
#
#     for i, pred in enumerate(clf.staged_predict_proba(X_train)):
#         train_score[i] = 1 - roc_auc_score(y_train, pred[:, 1])
#
#     best_iter = np.argmin(test_score)
#     test_line = plt.plot(test_score, label='test')
#
#     colour = test_line[-1].get_color()
#     plt.plot(train_score, '--', color=colour, label='train')
#
#     plt.xlabel("Number of boosting iterations")
#     plt.ylabel("1 - area under ROC")
#     plt.legend(loc='best')
#     plt.axvline(x=best_iter, color=colour)
#
#
# validation_curve(model)
#
#
# # Early stopping
# from sklearn.base import ClassifierMixin, clone
# from functools import partial
#
#
# def one_minus_roc(X, y, est):
#     pred = est.predict_proba(X)[:, 1]
#     return 1 - roc_auc_score(y, pred)
#
#
# class EarlyStopping(ClassifierMixin):
#     def __init__(self, estimator, max_n_estimators, scorer,
#                  n_min_iterations=50, scale=1.02):
#         self.estimator = estimator
#         self.max_n_estimators = max_n_estimators
#         self.scorer = scorer
#         self.scale = scale
#         self.n_min_iterations = n_min_iterations
#
#     def _make_estimator(self, append=True):
#         """Make and configure a copy of the `estimator` attribute.
#
#         Any estimator that has a `warm_start` option will work.
#         """
#         estimator = clone(self.estimator)
#         estimator.n_estimators = 1
#         estimator.warm_start = True
#         return estimator
#
#     def fit(self, X, y):
#         """Fit `estimator` using X and y as training set.
#
#         Fits up to `max_n_estimators` iterations and measures the performance
#         on a separate dataset using `scorer`
#         """
#         est = self._make_estimator()
#         self.scores_ = []
#
#         for n_est in range(1, self.max_n_estimators + 1):
#             est.n_estimators = n_est
#             est.fit(X, y)
#
#             score = self.scorer(est)
#             self.estimator_ = est
#             self.scores_.append(score)
#
#             if (n_est > self.n_min_iterations and
#                     score > self.scale * np.min(self.scores_)):
#                 return self
#
#         return self
#
#
# def stop_early(classifier, **kwargs):
#     n_iterations = classifier.n_estimators
#     early = EarlyStopping(classifier,
#                           max_n_estimators=n_iterations,
#                           # fix the dataset used for testing by currying
#                           scorer=partial(one_minus_roc, X_test, y_test),
#                           **kwargs)
#     early.fit(X_train, y_train)
#     plt.plot(np.arange(1, len(early.scores_) + 1),
#              early.scores_)
#     plt.xlabel("number of estimators")
#     plt.ylabel("1 - area under ROC")
#
#
# stop_early(model, n_min_iterations=100)


