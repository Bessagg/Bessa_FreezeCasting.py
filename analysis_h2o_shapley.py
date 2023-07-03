import h2o
import shap
import matplotlib.pyplot as plt
import os
import pandas as pd


"""Parameters"""
# Load generated df
preprocessor = 4
df = pd.read_pickle(f'freeze_casting_df_v{preprocessor}.pkl')
used_cols = df.columns
current_dir = os.curdir
avg_porosity = df['porosity'].mean()

"""Load Model"""
model_path = "selected_models/GBM/GBM_2304222235_6_0.6791_0.0814_0.0141_v4.0"
model_path = "selected_models/AutoML/AutoML_2304260818_6_0.7212_0.0752_0.0110"
model_path = "selected_models/DRF/DRF_2304260814_25_0.6370_0.0933_0.0153"
model_path = "selected_models/DLE/DLE_2304241215_18_0.6433_0.0887_0.0887"
seed = int(model_path.split('/')[-1].split('_')[2])

"""Get frames"""
h2o.init(nthreads=-1, min_mem_size_GB=8)
model = h2o.load_model(model_path)
h2o_data = h2o.H2OFrame(df, destination_frame="CatNum")
train, valid, test = h2o_data.split_frame([0.7, 0.15], seed=seed)


"""Explanation"""
ra_plot = model.residual_analysis_plot(test)
if ("AutoML" and "DLE") not in model_path:
    summary_pot = model.shap_summary_plot(test)
shapr_plot = model.shap_explain_row_plot(test, row_index=0)
pd_plot = model.pd_plot(test, column=used_cols[-1])
# ice_plot = model.ice_plot(test, column=used_cols[-1])
# mc_plot = model.model_correlation_heatmap(test)


"""Secondary Shapley Explanation"""


