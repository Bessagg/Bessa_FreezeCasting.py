from database2dataframe import db_to_df
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


class DataParser:
    def __init__(self):
        self.target = 'Porosity'
        self.ratios = [0.8]
        self.selected_cols_all = ['name_part1', 'material',  # name_part2
                                  'name_fluid1', 'sublimated', 'technique', 'direction',
                                  'material_group',
                                  'temp_cold', 'cooling_rate',
                                  'time_sub',
                                  'time_sinter_1', 'temp_sinter_1', 'vf_total', 'porosity']

        self.selected_cols_reduced = ['name_part1', 'name_part2',
                                      'name_fluid1',  # 'sublimated', 'technique', 'direction',
                                      'material_group',
                                      'temp_cold', 'cooling_rate',
                                      'time_sub',
                                      'time_sinter_1', 'temp_sinter_1', 'vf_total', 'porosity']

        self.selected_cols_v2 = ['name_part1', 'name_part2',
                                 'name_fluid1',  # 'sublimated', 'technique', 'direction',
                                 'material_group',
                                 'name_mold_mat',
                                 'name_disp_1', 'wf_disp_1',
                                 'name_bind1', 'wf_bind_1',
                                 'temp_cold', 'cooling_rate',
                                 'time_sub',
                                 'time_sinter_1', 'temp_sinter_1', 'vf_total', 'porosity']

        self.selected_cols_nameparts = ['name_part1', 'name_part2',  # material
                                        'name_fluid1', 'sublimated', 'technique', 'direction',
                                        'material_group',
                                        'temp_cold', 'cooling_rate',
                                        'time_sub',
                                        'time_sinter_1', 'temp_sinter_1', 'vf_total', 'porosity']

        # Selected Cols:
        self.selected_cols = self.selected_cols_reduced
        self.col_dtypes = {'name_part1': 'enum', 'name_part2': 'enum', 'name_fluid1': 'enum',
                           'name_mold_mat': 'enum',
                           'name_disp_1': 'enum', 'wf_disp_1': 'numeric', 'name_bind1': 'enum', 'wf_bind_1': 'numeric',
                           'material_group': 'enum', 'temp_cold': 'numeric', 'cooling_rate': 'numeric',
                           'time_sub': 'numeric',
                           'time_sinter_1': 'numeric', 'temp_sinter_1': 'numeric', 'vf_total': 'numeric',
                           'porosity': 'numeric'}

        self.col_dtypes_renamed = {'Solid Name': 'enum', 'Solid Name 2': 'enum', 'Fluid Name': 'enum',
                                   'Mold Material Name': 'enum',
                                   'Disp. Name': 'enum', 'Disp. wf.': 'numeric', 'Binder Name': 'enum',
                                   'Binder wf.': 'numeric',
                                   'Group': 'enum', 'Temp. Cold': 'numeric', 'Cooling Rate': 'numeric',
                                   'Time Sub.': 'numeric',
                                   'Time Sinter.': 'numeric', 'Temp Sinter': 'numeric', 'Solid Loading': 'numeric',
                                   'Porosity': 'numeric'}

        self.col_rename = {'name_part1': 'Solid Name', 'name_part2': 'Solid Name 2', 'name_fluid1': 'Fluid Name',
                           'name_mold_mat': 'Mold Material Name',
                           'name_disp_1': 'Disp. Name', 'wf_disp_1': 'Disp. wf.', 'name_bind1': 'Binder Name',
                           'wf_bind_1': 'Binder wf.',
                           'material_group': 'Group', 'temp_cold': 'Temp. Cold', 'cooling_rate': 'Cooling Rate',
                           'time_sub': 'Time Sub.',
                           'time_sinter_1': 'Time Sinter.', 'temp_sinter_1': 'Temp Sinter', 'vf_total': 'Solid Loading',
                           'porosity': 'Porosity'}

        # technique', 'sublimated' only have one value / direction' only has 600 rows

    def load_complete_data_from_mysql(self):
        df = db_to_df().copy()
        df['vf_solid'] = df['vf_part_1'] / (df['vf_part_1'] + df['vf_fluid_1'])  # wrong values, use vf_total
        df = df.fillna(value=np.nan)
        return df

    def load_complete_data_from_pickle(self):
        df = pd.read_pickle('freeze_casting_df_complete.pkl')
        return df

    def save_df_to_pickle(self, pkl_filename, df):
        with open(pkl_filename, 'wb') as file:
            pickle.dump(df, file)

    def preprocess_dropna(self, df):
        # Drop samples with no porosity values
        df = df[df['porosity'].notna()]
        return df

    def preprocess_drop_not_sublimated(self, df):
        # Drop samples with no porosity values
        df = df[df['sublimated'] != 'N']
        return df

    def rename_columns_df(self, df):
        return df.rename(columns=self.col_rename)

    def rename_columns_h2o(self, h2o_data):
        return h2o_data.rename(self.col_rename)


if __name__ == '__main__':
    data_parser = DataParser()
    df = data_parser.load_complete_data_from_mysql()
    # save dataframe
    data_parser.save_df_to_pickle('freeze_casting_df_complete.pkl', df)
    df = df.fillna(value=np.nan)
    print(df['direction'].value_counts(dropna=False))
