import os
import h2o
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import data_parser
from helpers import functions as fun
import numpy as np

# Load generated df
DataParser = data_parser.DataParser()
df = DataParser.load_complete_data_from_pickle()
df = df[DataParser.selected_cols]
df = DataParser.preprocess_dropna(df)
opt_save = True
selected_models_seed = 32
r2s = []

warnings.filterwarnings("ignore")
current_dir = os.getcwd()
models_dir = os.path.join(current_dir, "selected_models")
pallete = "summer"
# pd.options.display.precision = 4    # format 4 significant figures


h2o.init()
folders = os.listdir(models_dir)
models = dict()
for folder in folders:
    files = os.listdir(os.path.join(models_dir, folder))
    date = folder.split()
    files.sort(key=lambda x: x.split("_")[-2])  # sort by mrd
    files.sort(key=lambda x: x.split("_")[-1])  # sort by diff
    for file in files:
        models[file] = h2o.load_model(os.path.join(models_dir, folder, file))

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
        if model.split("_")[0] == "DLE":
            continue
        df_var_importance = pd.concat([df_var_importance, var_importance], ignore_index=True)

# Test best models for selected seed and compare
# Load DF
h2o_data = h2o.H2OFrame(df, destination_frame="CatNum")
train, test, valid = h2o_data.split_frame([0.7, 0.15], seed=selected_models_seed)
df_true = test[['porosity', 'material_group', 'name_fluid1']].as_data_frame()

# Selected Seed and models analysis:
selected_models_path = fun.get_selected_model_from_each_folder(selected_models_path="selected_models")
selected_seed_models = []
for path in selected_models_path:
    filename = path.split('\\')[-1]
    seed = int(filename.split('_')[2])
    if seed == selected_models_seed:
        selected_seed_models.append(filename)

for model_path in selected_seed_models:
    print(selected_seed_models)
    # seed = int(model.split("_")[2])
    model_name = model_path.split('\\')[-1]
    model_type = model_path.split("_")[0]
    if model_type in ['GBM', "DRF"]:
        models[model_name].download_mojo('mojos')  # download mojos  # Sometimes has to manual download: Run this line in console rather than script
    predicted = models[model_name].predict(test).as_data_frame()

    # Model Results
    df_model_results = predicted
    df_model_results['model_type'] = model_type
    df_model_results['error'] = df_model_results['predict'] - test['porosity'].as_data_frame().squeeze()
    df_model_results['error_abs'] = df_model_results['error'].abs()
    df_model_results['MAPE%'] = df_model_results['error_abs'] / test['porosity'].as_data_frame().squeeze()
    df_model_results.rename(columns=
                            {'predict': model_type,
                             'error': f'error_{model_type}',
                             'error_abs': f'error_abs_{model_type}',
                             'MAPE%': f'MAPE%{model_type}'
                             }, inplace=True)

    df_true = pd.concat([df_true, df_model_results], axis=1)
    df_true = df_true.loc[:, ~df_true.columns.duplicated()]  # remove duplicate columns
    # Model performance plot
    plt.figure(figsize=(8, 8))
    ax = sns.scatterplot(x='porosity', y=model_type, data=df_true, hue=f'error_abs_{model_type}', palette=pallete)
    norm = plt.Normalize(0, 0.4)  # set min and max for color_bar
    sm = plt.cm.ScalarMappable(cmap=pallete, norm=norm)
    sm.set_array([])
    ax.set_xlabel("True Porosity", fontsize=20)
    ax.set_ylabel(f"Predicted Porosity - {model_type} - seed 42", fontsize=20)
    ax.get_legend().remove()
    ax.figure.colorbar(sm)
    ax.set(ylim=(0.01, 1.01))
    ax.set(xlim=(0.01, 1.01))
    ax.tick_params(labelsize=20)
    # sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
    plt.show()
    plt.savefig(f'images/results/{model_type}_perf', bbox_inches='tight')

    # Error distribution plot
    plt.figure(figsize=(8, 8))
    top3_material_group = df_true.material_group.value_counts().iloc[:3].index.to_list()
    df_true_mg = df_true[(df_true['material_group'].isin(top3_material_group))]
    bx = sns.histplot(data=df_true_mg, x=f"error_{model_type}", hue="material_group", bins=20)
    bx.set_xlabel("Error", fontsize=20)
    bx.set_ylabel(f"Sample count - {model_type} - seed 42", fontsize=20)
    bx.tick_params(labelsize=20)
    bx.set(xlim=(-0.4, 0.4))
    x_axis = [round(num, 2) for num in np.linspace(-0.4, 0.4, 7)]
    plt.show()
    plt.savefig(f'images/results/{model_type}_error', bbox_inches='tight')

# # Generate Mojos for Decision Tress
# models["DRF_2209171340_42_0.6664_0.0848_0.0124"].download_mojo('mojos')
# models["GBM_2209171448_42_0.6315_0.0834_0.0145"].download_mojo('mojos')


# ################################## Generate Results and Tables
# Scaled importance Group
df_r[['r2', 'r2_t', 'mae', 'mae_t', 'mrd', 'mrd_t']] = df_r[['r2', 'r2_t', 'mae', 'mae_t', 'mrd', 'mrd_t']].astype(
    "float")

# df_grouped = df_var_importance.groupby('variable').mean()
top_10_variables = df_var_importance.groupby('variable')['scaled_importance'].mean()  # .nlargest(2)
sns.set(font_scale=1.2)

g = sns.catplot(data=df_var_importance, kind="bar", x="variable", y="scaled_importance", hue="model_type",
                palette=sns.color_palette("hls", 3))
sns.set(rc={'figure.figsize': (10, 10)})
plt.tight_layout()
g.set_axis_labels("Parameters", "Scaled Importance")
g.set_xticklabels(rotation=40, ha="right")
sns.move_legend(g, "upper right")
plt.subplots_adjust(bottom=0.4)
plt.show()
plt.savefig('images/results/scaled_importance.png', bbox_inches='tight')

# Model Group Results, prints and tables
pd.set_option('display.float_format', lambda x: '%.3g' % x)
df_r['Δr2'] = df_r['r2'] - df_r['r2_t']
df_r['seed'] = df_r['model'].str.split("_", expand=True)[2]
df_r2 = df_r.copy()
df_r.drop('model', axis=1, inplace=True)

print(df_r)
df_by_model_type = df_r.groupby('model_type')[['r2_t', 'mae_t', 'Δr2']].mean()
print('Model Mean Results by Type\n', df_by_model_type)
df_by_model_type_std = df_r.groupby('model_type')[['r2_t', 'mae_t', 'mrd_t']].std()
print('Model STD of Results by Type\n', df_by_model_type_std)

# Best model mrd for each type
df_best_model = df_r.groupby(['model_type'], as_index=False)['mrd_t'].min()
