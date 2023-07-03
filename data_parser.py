from database2dataframe import db_to_df
import pickle
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")


class DataParser:
    def __init__(self):
        self.selected_cols = ['name_part1', 'name_fluid1', 'sublimated', 'material', 'technique', 'direction',
                              'material_group',
                              'temp_cold', 'cooling_rate',
                              'time_sub',
                              'time_sinter_1', 'temp_sinter_1', 'vf_total', 'porosity']

        self.selected_cols_reduced = ['name_part1', 'name_fluid1', 'material', 'material_group',
                                      'temp_cold', 'cooling_rate',
                                      'time_sub',
                                      'time_sinter_1', 'temp_sinter_1',  'vf_total', 'porosity']
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

    def preprocess_0(self, df):
        # Drop samples with no porosity values
        df = df[df['porosity'].notna()]
        df = df[self.selected_cols]
        return df


if __name__ == '__main__':
    data_parser = DataParser()
    df = data_parser.load_complete_data_from_mysql()
    # save dataframe
    data_parser.save_df_to_pickle('freeze_casting_df_complete.pkl', df)
    df = df.fillna(value=np.nan)
    print(df['direction'].value_counts(dropna=False))