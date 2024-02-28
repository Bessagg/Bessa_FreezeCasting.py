import os
import h2o
import pandas as pd
import seaborn as sns
import warnings
import data_parser
from helpers import functions as fun
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



# Load generated df
DataParser = data_parser.DataParser()
df = DataParser.load_complete_data_from_pickle()
df = df[DataParser.selected_cols_v2]
df = DataParser.preprocess_dropna(df)
df = DataParser.rename_columns_df(df)
target = DataParser.target

opt_save = True
selected_models_seed = 42
r2s = []
warnings.filterwarnings("ignore")
current_dir = os.getcwd()
models_dir = os.path.join(current_dir, "selected_models")
pallete = "summer"
# pd.options.display.precision = 4    # format 4 significant figures


h2o.init()
h2o_data = h2o.H2OFrame(df, destination_frame="CatNum", column_types=DataParser.col_dtypes_renamed)
# train, test, valid = h2o_data.split_frame([1-test_ratio, test_ratio], seed=selected_models_seed)
# train_valid = h2o.H2OFrame.rbind(train, valid)
train, test = h2o_data.split_frame(ratios=DataParser.ratios, seed=selected_models_seed)

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
    perf = models[model].cross_validation_metrics_summary().as_data_frame()
    data['model_type'] = model.split("_")[0]
    data['r2'] =  models[model].model_performance(test_data=test)['r2']
    data['r2_t'] = models[model].model_performance(test_data=train)['r2']
    data['mae'] = models[model].model_performance(test_data=test)['mae']
    data['mae_t'] = model.split("_")[4]
    data['mrd'] = models[model].model_performance(test_data=test)['mean_residual_deviance']
    data['mrd_t'] = model.split("_")[5]
    new_row = df_r.from_dict([data])
    df_r = pd.concat([df_r, new_row], ignore_index=True)
    if models[model].varimp() is not None:
        var_importance = models[model].varimp(True)
        var_importance['model'] = model
        var_importance['model_type'] = model.split("_")[0]
        # if model.split("_")[0] == "DLE":
        #     continue
        df_var_importance = pd.concat([df_var_importance, var_importance], ignore_index=True)
    print(model, "\n", models[model].cross_validation_metrics_summary())

# Test best models for selected seed and compare
# Load DF

# df_true = test[[DataParser.target, 'material_group', 'name_fluid1']].as_data_frame()
df_true_test = test[[DataParser.target, 'Group', 'Fluid Name', 'Solid Name']].as_data_frame()

# Selected Seed and models analysis:
selected_models_path = fun.get_selected_model_from_each_folder(selected_models_path="selected_models")
selected_seed_models = []
for path in selected_models_path:
    filename = path.split('\\')[-1]
    seed = int(filename.split('_')[2])
    if seed == selected_models_seed:
        selected_seed_models.append(filename)

selected_seed_models.append('linear_reg')
# selected_seed_models.reverse()
for model_path in selected_seed_models:
    plt.close('all')
    if model_path != 'linear_reg':
        model_name = model_path.split('\\')[-1]
        model_type = model_path.split("_")[0]
        print(selected_seed_models, "\n", model_name)
        predicted = models[model_name].predict(test).as_data_frame()
        if model_type in ['GBM', "DRF"]:
            models[model_name].download_mojo(
                'mojos')  # download mojos  # Sometimes has to manual download: Run this line in console rather than script

    else:
        X_train, y_train = train['Solid Loading'].as_data_frame(), train[target].as_data_frame()
        X_test, y_test = test['Solid Loading'].as_data_frame(), test[target].as_data_frame()
        median = np.nanmedian(X_train)
        # Replace NaN values with the median
        X_train[np.isnan(X_train)] = median
        X_test[np.isnan(X_test)] = median
        X_train, y_train = np.array(X_train).reshape(-1, 1), np.array(y_train).reshape(-1, 1)  # Linear reg expects matrix, hence make it [[1], [2], [3]]
        X_test, y_test = np.array(X_test).reshape(-1, 1), np.array(y_test).reshape(-1, 1)  # Linear reg expects matrix, hence make it [[1], [2], [3]]
        model_type = 'linear_reg'
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict on training and testing sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_train_pred = [item for sublist in y_train_pred for item in sublist]
        y_train_pred = pd.DataFrame(y_train_pred, columns=['predict'])
        y_test_pred = [item for sublist in y_test_pred for item in sublist]
        y_test_pred = pd.DataFrame(y_test_pred, columns=['predict'])

        data = dict()
        data['model'] = model_type
        data['model_type'] = model_type
        data['r2'] = r2_score(y_test, y_test_pred)
        data['r2_t'] = r2_score(y_train, y_train_pred)
        data['mae'] = mean_absolute_error(y_test, y_test_pred)
        data['mae_t'] = mean_absolute_error(y_train, y_train_pred)
        data['mrd'] = mean_absolute_error(y_test, y_test_pred)
        data['mrd_t'] = mean_absolute_error(y_train, y_train_pred)
        new_row = df_r.from_dict([data])
        df_r = pd.concat([df_r, new_row], ignore_index=True)
        predicted = y_test_pred

    # Model Results
    df_model_results = predicted
    df_model_results['model_type'] = model_type
    df_model_results['error'] = df_model_results['predict'] - test[DataParser.target].as_data_frame().squeeze()
    df_model_results['error_abs'] = df_model_results['error'].abs()
    df_model_results['MAPE%'] = df_model_results['error_abs'] / test[DataParser.target].as_data_frame().squeeze()
    df_model_results.rename(columns=
                            {'predict': model_type,
                             'error': f'error_{model_type}',
                             'error_abs': f'error_abs_{model_type}',
                             'MAPE%': f'MAPE%{model_type}'
                             }, inplace=True)

    df_true_test = pd.concat([df_true_test, df_model_results], axis=1)
    df_true_test = df_true_test.loc[:, ~df_true_test.columns.duplicated()]  # remove duplicate columns
    # Model performance plot
    plt.figure(figsize=(18, 12))
    ax = sns.scatterplot(x=DataParser.target, y=model_type, data=df_true_test, hue=f'error_abs_{model_type}', palette=pallete)
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
    plt.figure(figsize=(18, 12))
    top3_material_group = df_true_test['Group'].value_counts().iloc[:3].index.to_list()
    df_true_mg = df_true_test[(df_true_test['Group'].isin(top3_material_group))]
    bx = sns.histplot(data=df_true_mg, x=f"error_{model_type}", hue="Group", bins=20,
                      palette=sns.color_palette("hls", 3))
    bx.set_xlabel("Error", fontsize=20)
    bx.set_ylabel(f"Sample count - {model_type} - seed 42", fontsize=20)
    bx.tick_params(labelsize=20)
    bx.set(xlim=(-0.4, 0.4))
    x_axis = [round(num, 2) for num in np.linspace(-0.4, 0.4, 7)]
    plt.show()
    plt.savefig(f'images/results/{model_type}_error', bbox_inches='tight')

    """Importances"""""
    df_r[['r2', 'r2_t', 'mae', 'mae_t', 'mrd', 'mrd_t']] = df_r[['r2', 'r2_t', 'mae', 'mae_t', 'mrd', 'mrd_t']].astype(
        "float")

    # df_grouped = df_var_importance.groupby('variable').mean()
    df_imp = df_var_importance[df_var_importance['model_type'] == model_type]
    top_variables = df_imp.groupby('variable')['scaled_importance'].mean().nlargest(15).index.to_list()
    top_df = df_imp.loc[df_imp['variable'].isin(top_variables)]
    if not top_df.empty:
        plt.figure()
        sns.set(font_scale=1.2)
        g = sns.catplot(data=top_df, kind="bar", x="variable", y="scaled_importance", hue="model_type",
                        palette=sns.color_palette("hls", 3))
        sns.set(rc={'figure.figsize': (18, 12)})
        plt.tight_layout()
        g.set_axis_labels("Parameters", "Scaled Importance - Decision Trees")
        g.set_xticklabels(rotation=40, ha="right")
        sns.move_legend(g, "upper right")
        plt.subplots_adjust(bottom=0.4)
        plt.show()
        plt.savefig(f'images/results/{model_type}_scaled_importance_trees.png', bbox_inches='tight')

# ################################## Generate Results and Tables
# Model Group Results, prints and tables
pd.set_option('display.float_format', lambda x: '%.3g' % x)
df_r['Δr2'] = df_r['r2'] - df_r['r2_t']
df_r['seed'] = df_r['model'].str.split("_", expand=True)[2]
df_r2 = df_r.copy()
df_r.drop('model', axis=1, inplace=True)

"""Print Results"""
print("\n", df_r)
df_by_model_type = df_r.groupby('model_type')[['r2', 'r2_t', 'mae', 'Δr2']].mean().sort_values(by='r2', ascending=False)
print('\n Model Mean Results by Type\n', df_by_model_type)
df_by_model_type_std = df_r.groupby('model_type')[['r2', 'r2_t', 'mae_t', 'mrd_t']].std().sort_values(by='r2', ascending=False)
print('\n Model STD of Results by Type\n', df_by_model_type_std)

# Best model mrd for each type
df_best_model = df_r.groupby(['model_type'], as_index=False)['mrd_t'].min()





# Define a function to calculate R^2 for each group
def calculate_r2(group, X, y):
    y = group[y]
    y_pred = group[[X]]
    r2 = r2_score(y, y_pred)
    return r2


groups_names = ['Group', 'Solid Name', 'Fluid Name']
group_dict = dict()
df_train = train.as_data_frame()
i = 0
for group_name in groups_names:
    # top_gclasses = df_true_test[group_name].value_counts().nlargest(5).index.to_list()  # top group classes in test
    top_gclasses = df_train[group_name].value_counts().nlargest(5).index.to_list()  # top classes in train

    top_gdata = df_true_test[df_true_test[group_name].isin(top_gclasses)]
    top_gtrain = df_train[df_train[group_name].isin(top_gclasses)]

    # for model_name in ['GBM', 'DRF', 'AutoML', 'linear_reg']:
    for model_name in ['GBM']:
        for group_class in top_gdata[group_name].unique().tolist():  # top in test data
            class_data = top_gdata[top_gdata[group_name].isin([group_class])]
            train_data = top_gtrain[top_gtrain[group_name].isin([group_class])]
            r2 = r2_score(class_data[target], class_data[model_name])  # true, pred
            i += 1  # group dict row number
            new_dict = {i: {'model_name': model_name, 'group_name': group_name, 'group_class': group_class, 'r2': r2,
                            'train samples': len(train_data), 'test samples': len(class_data)}}
            group_dict.update(new_dict)

group_df = pd.DataFrame(group_dict).T
group_df = group_df.sort_values(by=['group_name', 'test samples'], ascending=False)
print('Grouped performance top \n', group_df)

# r2_score(df_true_test[target], df_true_test[model_type])
# num_cols = df_true_test.select_dtypes(include=['number']).columns
# df_true_test.groupby('Group')[num_cols].mean()
# g_group = df_true_test.groupby('Group')[num_cols].mean()[[f'MAPE%{model_type}', f'error_abs_{model_type}']]
# g_fluid = df_true_test.groupby('Fluid Name')[num_cols].mean()
# g_solid = df_true_test.groupby('Solid Name')[num_cols].mean()
plt.close('all')

