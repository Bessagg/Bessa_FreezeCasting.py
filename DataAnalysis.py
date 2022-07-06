# Enables inline-plot rendering
# Utilized to create and work with dataframes
import sys
import time

from database2dataframe import db_to_df
import pandas as pd
import numpy as np
import math as m
# MATPLOTLIB
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import warnings
import matplotlib.mlab as mlab
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib import rc
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from pylab import rcParams
from matplotlib.ticker import AutoMinorLocator
import scipy.stats as stats
import statsmodels.api as sm
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.ticker as ticker
# For point density plots:
from scipy.stats import gaussian_kde
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import norm
# command-line arguments
import sys
import prince
import plotly.io as pio
pio.renderers.default = "browser"

"""
Possíveis tentativas:
Testar dropando todos os nulos e treinando junto (306 linhas)
Treinando separadamente (num = 464 linhas, str = 1073 linhas, str_filtered = 270 linhas)
Matrix de arvore nao tem problema com isso
Talvez usar gradientboost para aumentar os dados

"""

# Analysis with df
df = db_to_df().copy()
print(f"Count of Null values out of {len(df)} rows \n", df.isnull().sum())
# # Drop columns with less than min_n_rows as not null values
# for col in df.columns:
#     print(f'{col}_rows: {df[col].count()}')
#     if df[col].count() < min_n_rows:
#         df.drop(columns=col, axis=1, inplace=True)
#         print(f"*dropped {col}, less than {min_n_rows} rows")
#
# """technique and direction only have 610 rows and temp_cold 1643 :c """
# print(f'Selected columns with more than {min_n_rows}: \n{df.columns}')
# print("Rows:", len(df))

# # #### Pore_strucutre plots
# # Only select top 5 most common pore_structures
# pore_structure_filter = df['pore_structure'].value_counts().head(5).axes[0]
# df = df[df['pore_structure'].isin(pore_structure_filter)]
# pd.value_counts(df['pore_structure']).plot.bar()
# plt.figure(figsize=(16, 6))

# Correlation heatmap
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 18}, pad=12)
print("Correlation matrix \n")
df.corr()['porosity']


# #################################### Categorical Analysis
# Plot porosity against string columns
str_cols = df.select_dtypes(include=[object]).columns
df_str = df[str_cols].dropna()
count_filter_n = 50
rank_filter_n = 5

plt.close("all")
for col in str_cols:
    filtered_df = df[df[col].notnull()]  # Remove null in column
    rank_filter_n = 5
    rank_filter = df_str[col].value_counts().head(rank_filter_n).axes[0]  # Filter top 5 in column
    count_filter = df.groupby(col).filter(lambda x: len(x) > count_filter_n)[col].unique()
    selected_filter = count_filter
    filtered_df = filtered_df[filtered_df[col].isin(selected_filter)]
    g = sns.FacetGrid(filtered_df, row=col,
                      height=1.6, aspect=4)
    g.map(sns.kdeplot, 'porosity')
    # rank_filter = df_str[col].value_counts().head(5)  # list to filter by rank

plt.show()
plt.close("all")


mca = prince.MCA()
X = df[str_cols].dropna()
X = filtered_df[str_cols].dropna()
fig, ax = plt.subplots()
mc = prince.MCA(n_components=10, n_iter=10, copy=True, check_input=True, engine='auto', random_state=42).fit(X)
mc.plot_coordinates(
     X=X,
     ax=None,
     figsize=(6, 6),
     show_row_points=True,
     row_points_size=10,
     show_row_labels=False,
     show_column_points=True,
     column_points_size=30,
     show_column_labels=False,
     legend_n_cols=1
)
print("MC eigen values", mc.eigenvalues_)


encoder = OneHotEncoder(handle_unknown='ignore')

# # #################################### Numerical Analysis

"""
Before you perform factor analysis, you need to evaluate the “factorability” of our dataset. 
Factorability means "can we found the factors in the dataset?". T
here are two methods to check the factorability or sampling adequacy:

Bartlett’s Test
Kaiser-Meyer-Olkin Test
"""
num_cols = df.select_dtypes(include=[float]).columns
df_num = df[num_cols].dropna()
count_filter_n = 50

"""
Bartlett's test
In this Bartlett ’s test, the p-value is 0. 
The test was statistically significant, indicating that the observed correlation matrix is not an identity matrix.
"""
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value, p_value = calculate_bartlett_sphericity(df_num)
print(chi_square_value, p_value)

"""
Kaiser-Meyer-Olkin (KMO) Test measures the suitability of data for factor analysis. 
It determines the adequacy for each observed variable and for the complete model. 
KMO estimates the proportion of variance among all the observed variable. 
Lower proportion id more suitable for factor analysis. KMO values range between 0 and 1. 
Value of KMO less than 0.6 is considered inadequate.
"""
from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all, kmo_model = calculate_kmo(df_num)
print(kmo_model)


# ########### PCA Analysis Principal component Analysis
n_components = 5
pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=n_components))])
pca = PCA(n_components=n_components)
# X = df_num[df_num.columns.drop('porosity')]
X = df_num[df_num.columns]
y = df_num['porosity']

# components = pca.fit_transform(df_num)
# components = pipeline.fit_transform(df_num)
X_scaled = pd.DataFrame(preprocessing.scale(X), columns=X.columns)  # normalize data
components = pca.fit_transform(X_scaled)
# print(pd.DataFrame(pca.components_, columns=X_scaled.columns, index=['PC-1', 'PC-2', 'PC-3', 'PC-4', 'PC-5']))
total_var = pca.explained_variance_ratio_.sum() * 100
labels = {str(i): f"PC {i+1}" for i in range(n_components)}
labels['color'] = 'porosity'
fig = px.scatter_matrix(
    components,
    color=y,
    dimensions=range(n_components),
    labels=labels,
    title=f'Total Explained Variance: {total_var:.2f}%',
)
fig.update_traces(diagonal_visible=False)
fig.show()

# 3D plot Variance
fig = px.scatter_3d(
    components, x=0, y=1, z=2, color=df_num['porosity'], title=f'Total Explained Variance: {total_var:.2f}%', labels=labels
)
fig.show()
exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
px.area(x=range(1, exp_var_cumul.shape[0] + 1), y=exp_var_cumul, labels={"x": "# Components", "y": "Explained Variance"})

