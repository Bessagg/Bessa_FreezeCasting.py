import pickle
from database2dataframe import db_to_df

df = db_to_df().copy()
pkl_filename = "freeze_casting_df.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(df, file)
