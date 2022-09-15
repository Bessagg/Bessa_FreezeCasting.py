import os
import glob
import h2o
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
current_dir = os.getcwd()
temp_dir = os.path.join(current_dir, "selected_models")
h2o.init()
folders = os.listdir(temp_dir)
models = dict()
for folder in folders:
    files = os.listdir(os.path.join(temp_dir, folder))
    date = folder.split()
    files.sort(key=lambda x: x.split("_")[-2])  # sort by mrd
    files.sort(key=lambda x: x.split("_")[-1])  # sort by diff
    for file in files:
        models[file] = h2o.load_model(os.path.join(temp_dir, folder, file))


df_r = pd.DataFrame()
df_var_importance = pd.DataFrame()
for model in models:
    data = dict()
    data['model'] = model
    data['model_type'] = model.split("_")[0]
    data['r2'] = models[model].r2()
    data['r2_t'] = model.split("_")[3]
    data['mae'] = models[model].mae()
    data['mae_t'] = model.split("_")[4]
    data['mrd'] = models[model].mean_residual_deviance()
    data['mrd_t'] = model.split("_")[5]
    new_row = df_r.from_dict([data])
    df_r = pd.concat([df_r, new_row], ignore_index=True)
    if models[model].varimp() is not None:
        var_importance = models[model].varimp(True)
        var_importance['model'] = model
        var_importance['model_type'] = model.split("_")[0]
        if model.split("_")[0] == "RNN":
            continue
        df_var_importance = pd.concat([df_var_importance, var_importance], ignore_index=True)

# Scaled importance Group
df_r[['r2', 'r2_t', 'mae', 'mae_t', 'mrd', 'mrd_t']] = df_r[['r2', 'r2_t', 'mae', 'mae_t', 'mrd', 'mrd_t']].astype("float")
df_grouped = df_var_importance.groupby('variable').mean()
top_10_variables = df_var_importance.groupby('variable')['scaled_importance'].nlargest(2)
sns.set(font_scale=1.2)
g = sns.catplot(data=df_var_importance, kind="bar", x="variable", y="scaled_importance", hue="model_type",
                palette=sns.color_palette("hls", 3))
g.fig.set_size_inches(18, 5)
sns.move_legend(g, "upper right")

# Model Group Results
df_by_model_type = df_r.groupby('model_type')['r2_t', 'mae_t', 'mrd_t'].mean()
df_by_model_type_std = df_r.groupby('model_type')['r2_t', 'mae_t', 'mrd_t'].std()

# Best model mrd for each type
df_best_model = df_r.groupby(['model_type'], as_index=False)['mrd_t'].min()

df = pd.read_pickle('freeze_casting_df.pkl')
h2o_data = h2o.H2OFrame(df, destination_frame="CatNum")

# Test best models for seed 42 and compare
seed = 42
train, test, valid = h2o_data.split_frame([0.7, 0.15], seed=seed)
df_true = test[['porosity', 'material_group', 'name_fluid1']].as_data_frame()

for model in ["AutoML_2209031853_42_0.6204_0.0887_0.0149_0.1532",
              "DRF_2209132130_42_0.6519_0.0864_0.0129",
              "RNN_2207271258_42_0.6489_0.0878_0.0878_0.1094",
              "GBMv_2209111729_42_0.6311_0.0828_0.0145"]:
    # seed = int(model.split("_")[2])
    model_type = model.split("_")[0]
    predicted = models[model].predict(test).as_data_frame()

    # Model Results
    df_mr = predicted
    df_mr['model_type'] = model_type
    df_mr['error'] = df_mr['predict'] - test['porosity'].as_data_frame().squeeze()
    df_mr.rename(columns={'predict': model_type, 'error': f'error_{model_type}'}, inplace=True)

    df_true = pd.concat([df_true, df_mr], axis=1)
    # df_test.append(df_mr)
    sns.scatterplot(x='porosity', y='predict', data=df_mr, palette=sns.color_palette("hls", 2))

import plotly.express as px
import plotly.graph_objects as go

fig = px.scatter(df, x='total_bill', y='tip', color='sex', opacity=0.65)
fig.add_traces(go.Scatter(x=df_true[''], y=y_uni, name='Weights: Uniform'))
fig.add_traces(go.Scatter(x=x_range, y=y_dist, name='Weights: Distance'))
fig.show()

