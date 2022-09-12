import pickle
from database2dataframe import db_to_df

df = db_to_df().copy()
df['vf_solid'] = df['vf_part_1']/(df['vf_part_1'] + df['vf_fluid_1'])
df.drop('vf_part_1', axis=1, inplace=True)
df.drop('vf_fluid_1', axis=1, inplace=True)
pkl_filename = "freeze_casting_df_v2.0.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(df, file)

print(f"saved df as {pkl_filename}")
