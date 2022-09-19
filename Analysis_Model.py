import os
import h2o
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
current_dir = os.getcwd()
models_dir = os.path.join(current_dir, "selected_models")
df = pd.read_pickle('freeze_casting_df_v4.0.pkl')
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

# Test best models for seed 42 and compare
seed = 42
# Load DF
h2o_data = h2o.H2OFrame(df, destination_frame="CatNum")
train, test, valid = h2o_data.split_frame([0.7, 0.15], seed=seed)
df_true = test[['porosity', 'material_group', 'name_fluid1']].as_data_frame()

for model in ["AutoML_2209180117_42_0.6270_0.0881_0.0146",
              "DRF_2209171340_42_0.6664_0.0848_0.0124",
              "DLE_2209171651_42_0.6476_0.0886_0.0886",
              "GBM_2209171448_42_0.6315_0.0834_0.0145"]:
    # seed = int(model.split("_")[2])
    model_type = model.split("_")[0]
    predicted = models[model].predict(test).as_data_frame()

    # Model Results
    df_model_results = predicted
    df_model_results['model_type'] = model_type
    df_model_results['erro'] = df_model_results['predict'] - test['porosity'].as_data_frame().squeeze()
    df_model_results['erro_abs'] = df_model_results['erro'].abs()
    df_model_results['MAPE%'] = df_model_results['erro_abs']/test['porosity']
    df_model_results.rename(columns={'predict': model_type, 'erro': f'erro_{model_type}', 'erro_abs': f'erro_abs_{model_type}'}
                            , inplace=True)

    df_true = pd.concat([df_true, df_model_results], axis=1)

    # Model performance plot
    plt.figure(figsize=(8, 8))
    ax = sns.scatterplot(x='porosity', y=model_type, data=df_true, hue=f'erro_abs_{model_type}', palette=pallete)
    norm = plt.Normalize(0, 0.4)  # set min and max for color_bar
    sm = plt.cm.ScalarMappable(cmap=pallete, norm=norm)
    sm.set_array([])
    ax.set_xlabel("Porosidade Verdadeira", fontsize=20)
    ax.set_ylabel(f"Porosidade Prevista - {model_type} - seed 42", fontsize=20)
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
    bx = sns.histplot(data=df_true_mg, x=f"erro_{model_type}", hue="material_group", bins=20)
    bx.set_xlabel("Erro", fontsize=20)
    bx.set_ylabel(f"Contagem de Amostras - {model_type} - seed 42", fontsize=20)
    bx.tick_params(labelsize=20)
    bx.set(xlim=(-0.4, 0.4))
    plt.show()
    plt.savefig(f'images/results/{model_type}_erro', bbox_inches='tight')

# Generate Mojos for Decision Tress
# models["DRF_2209132130_42_0.6519_0.0864_0.0129"].download_mojo('mojos')
# models["GBM_2209111729_42_0.6311_0.0828_0.0145"].download_mojo('mojos')


# ################################## Generate Results and Tables
# Scaled importance Group
df_r[['r2', 'r2_t', 'mae', 'mae_t', 'mrd', 'mrd_t']] = df_r[['r2', 'r2_t', 'mae', 'mae_t', 'mrd', 'mrd_t']].astype(
    "float")

df_grouped = df_var_importance.groupby('variable').mean()
top_10_variables = df_var_importance.groupby('variable')['scaled_importance'].nlargest(2)
sns.set(font_scale=1.2)
df_var_importance['variable'].replace(
    {'name_fluid1': 'Nome do Fluido', 'material': 'Nome do Sólido', 'material_group': 'Tipo da Amostra',
     'temp_cold': 'Temp. Congelamento', 'cooling_rate': 'Taxa de Congel.', 'time_sub': 'Tempo de Sublim.',
     'time_sinter_1': 'Tempo de Sinter.', 'temp_sinter_1': 'Temp. de Sinter.',
     'porosity': 'Porosidade', 'vf_solid': 'Fração de Vol. Sól.', 'vf_part_1': "Fração de Vol.Sólido",
     'vf_fluid_1': "Fração de Vol.Fluido"}, inplace=True)


g = sns.catplot(data=df_var_importance, kind="bar", x="variable", y="scaled_importance", hue="model_type",
                palette=sns.color_palette("hls", 3))
sns.set(rc={'figure.figsize': (10, 10)})
plt.tight_layout()
g.set_axis_labels("Parâmetro", "Importância Escalada")
g.set_xticklabels(rotation=40, ha="right")
sns.move_legend(g, "upper right")
plt.subplots_adjust(bottom=0.4)
plt.show()
plt.savefig('images/results/importancia_escalada.png', bbox_inches='tight')

# Model Group Results, prints and tables
pd.set_option('display.float_format', lambda x: '%.3g' % x)
df_r['Δr2'] = df_r['r2'] - df_r['r2_t']
df_r['seed'] = df_r['model'].str.split("_", expand=True)[2]
df_r2 = df_r.copy()
df_r.drop('model', axis=1, inplace=True)


print(df_r)
df_by_model_type = df_r.groupby('model_type')['r2_t', 'mae_t', 'mrd_t', 'Δr2'].mean()
print('Model Mean Results by Type\n', df_by_model_type)
df_by_model_type_std = df_r.groupby('model_type')['r2_t', 'mae_t', 'mrd_t'].std()
print('Model STD of Results by Type\n', df_by_model_type_std)

# Best model mrd for each type
df_best_model = df_r.groupby(['model_type'], as_index=False)['mrd_t'].min()