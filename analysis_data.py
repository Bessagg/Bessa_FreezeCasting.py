# Enables inline-plot rendering
# Utilized to create and work with dataframes
import sys
import time
import pandas as pd
import numpy as np
import math as m
# MATPLOTLIB
import matplotlib.pyplot as plt
# matplotlib parameters for Latex
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import seaborn as sns
import prince
import plotly.io as pio

pio.renderers.default = "browser"

# Load generated df
# import database2dataframe
df = pd.read_pickle('freeze_casting_df_v04.pkl')

# Rename Columns and values - translate pt
# df.rename(columns={'name_fluid1': 'Nome do Fluido', 'material': 'Nome do Sólido', 'material_group': 'Tipo da Amostra',
#                    'temp_cold': 'Temp. Congelamento', 'cooling_rate': 'Taxa de Congel.', 'time_sub': 'Tempo de Sublim.',
#                    'time_sinter_1': 'Tempo de Sinter.', 'temp_sinter_1': 'Temp. de Sinter.',
#                    'porosity': 'Porosidade', 'vf_solid': 'Fração de Vol. Sól.'}, inplace=True)
#
# df['Nome do Fluido'].replace({'water': 'Água', 'camphene': 'Canfeno',
#                               'acetic acid': 'Ácido acético', 'naphthalene': 'Naftaleno'}, inplace=True)
#
# df['Tipo da Amostra'].replace({'Ceramic': 'Cerâmico', 'Polymer': 'Polímero',
#                               'Ceramic/Polymer': 'Cerâmica/Polímero', 'Metal/Ceramic': 'Metal/Cerâmico'}, inplace=True)
print(df.head())
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

# #################################### Correlation heatmap
plt.figure(figsize=(18, 12))
# plt.tight_layout()
plt.show()
plt.subplots_adjust(left=0.21, right=1.05, top=0.95, bottom=0.3)
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG', fmt=".2%", annot_kws={"fontsize": 18})
heatmap.set_xticklabels(heatmap.get_xmajorticklabels(), fontsize=18)
heatmap.set_yticklabels(heatmap.get_ymajorticklabels(), fontsize=18)
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, horizontalalignment='right')
# heatmap.set_title('Matriz de Correlação', fontdict={'fontsize': 18}, pad=12)
print("Correlation matrix \n")
plt.savefig(f"images/Correlation.png")
# df.corr()['porosity']


# #################################### Categorical Analysis
# Plot porosidade against string columns
str_cols = df.select_dtypes(include=[object]).columns
df_str = df[str_cols].dropna()
count_filter_n = 50
rank_filter_n = 5

# Count categorical data
for col in str_cols:
    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.4)
    top_n = 5
    top_samples = df.groupby(col)[col].count().sort_values(ascending=False)[0:top_n]
    top_samples_columns = top_samples.axes[0].values
    # top_10_samples_columns = ['Al2O3', 'HAP', 'YSZ', 'Mullite', 'PZT', 'Bioglass', 'Si3N4', 'Al2O3/ZrO2', 'TiO2', 'SiO2']
    ax = top_samples.iloc[0:top_n].plot(kind="bar", fontsize=20)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    ax.bar_label(ax.containers[0], label_type='center', fontsize=24)
    ax.axes.get_yaxis().set_visible(False)
    ax.xaxis.set_label_text("")
    plt.savefig(f"images/Contagem de {col}.png", bbox_inches='tight')
    plt.show()

# plt.close("all")
for col in str_cols:
    sns.set(font_scale=1.25)
    filtered_df = df[df[col].notnull()]  # Remove null in column
    rank_filter_n = 3
    rank_filter = df_str[col].value_counts().head(rank_filter_n).axes[0]  # Filter top 5 in column
    count_filter = df.groupby(col).filter(lambda x: len(x) > count_filter_n)[col].unique()
    selected_filter = rank_filter  # change here for count or rank filtering
    filtered_df = filtered_df[filtered_df[col].isin(selected_filter)]
    g = sns.FacetGrid(filtered_df, row=col,
                      height=1.6, aspect=4)
    g.map(sns.kdeplot, 'Porosidade')
    g.set_ylabels('Densidade')
    plt.savefig(f"images/Distribuição de {col}.png", bbox_inches='tight')
    # rank_filter = df_str[col].value_counts().head(5)  # list to filter by rank

plt.show()
# plt.close("all")


mca = prince.MCA()
X = df[str_cols].dropna()
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
Factorability means "can we found the factors in the dataset?". 
here are two methods to check the factorability or sampling adequacy:

Bartlett’s Test
Kaiser-Meyer-Olkin Test
"""
num_cols = df.select_dtypes(include=[float]).columns
df_num = df[num_cols].dropna()  #
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
n_components = 3
pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=n_components))])
pca = PCA(n_components=n_components)
# X = df_num[df_num.columns.drop('Porosidade')]
X = df_num[df_num.columns]
y = df_num['Porosidade']

# components = pca.fit_transform(df_num)
# components = pipeline.fit_transform(df_num)
X_scaled = pd.DataFrame(preprocessing.scale(X), columns=X.columns)  # normalize data
components = pca.fit_transform(X_scaled)
# print(pd.DataFrame(pca.components_, columns=X_scaled.columns, index=['PC-1', 'PC-2', 'PC-3', 'PC-4', 'PC-5']))
total_var = pca.explained_variance_ratio_.sum() * 100
labels = {str(i): f"PC {i + 1}" for i in range(n_components)}
labels['color'] = 'Porosidade'
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
    components, x=0, y=1, z=2, color=df_num['Porosidade'], title=f'Total Explained Variance: {total_var:.2f}%',
    labels=labels
)
fig.show()
exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
px.area(x=range(1, exp_var_cumul.shape[0] + 1), y=exp_var_cumul,
        labels={"x": "# Components", "y": "Explained Variance"})
