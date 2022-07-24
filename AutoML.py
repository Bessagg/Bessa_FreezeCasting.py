import database2dataframe
import datetime
import time
import pandas as pd
# Analysis with df
# import database2dataframe
# df = database2dataframe.db_to_df().copy()

df = pd.read_pickle('freeze_casting_df.pkl')
print("Used columns:", df.columns)
df.head()


import h2o
for seed in [6, 18, 25, 34, 42]:
    h2o.init()
    # Split the dataset into a train and valid set:
    h2o_data = h2o.H2OFrame(df, destination_frame="CatNum")
    train, valid, test = h2o_data.split_frame([0.7, 0.15], seed=seed)
    train.frame_id = "Train"
    valid.frame_id = "Valid"
    test.frame_id = "Test"
    from h2o.automl import H2OAutoML
    import h2o.estimators
    from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
    start = time.time()
    aml = H2OAutoML(max_models=50, seed=seed, stopping_metric='AUTO')
    print("Training")
    X = df[df.columns.drop('porosity')].columns.values.tolist()
    y = "porosity"
    aml.train(x=X, y=y,
              training_frame=train,
              validation_frame=valid)
    print("Leaderboard")
    print(aml.leaderboard)
    best_model = aml.get_best_model(criterion="deviance")
    test_y = aml.leader.predict(test)
    test_y = test_y.as_data_frame()

    print("Testing best model")
    print(best_model.model_performance(test))
    print("Validating")
    print(best_model.model_performance(valid))
    print(best_model.show())

    r2, mae = best_model.r2(valid=True), best_model.mae(valid=True)
    r2, r2_train = "{:.04f}".format(r2), "{:.04f}".format(mae)
    mrd = best_model.mean_residual_deviance(valid=True)
    mrd = "{:.04f}".format(mrd)

    print(f"R2: train {best_model.r2()} \n valid {best_model.r2(valid=True)} \n "
          f"diff {best_model.r2() - best_model.r2(valid=True) }")

    print("R2 and mae", r2, best_model.mae(valid=True))
    now = datetime.datetime.now().strftime("%y%m%d%H%M")
    h2o.save_model(best_model, path="temp/AutoML_model", filename=f"AutoML_{seed}_{now}_{r2}_{mae}_{mrd}", force=True)
    print("Elapsed {:.04f} minutes".format((time.time() - start)/60))
# h2o.save_model(best_model, path="temp/AutoML_model", force=True)

# com filter treino: 0.084    stackedensemble
# com filter mae 0.84 no test, 0.56r^2  | train : 0.63 e 0.75 r^2
# sem dropar  0.0816 0.65 r^2 valid e 0.0637 no treino r^0.8 no treino
# sem dropar 0.0816 0.65 r^2 | 0.637 e 0.8r^2
# 50 modelos 0.814 0.648r^2 | 0.64 e 0.8r^2

# Without pore_structure:
# com freezing temp, e direction 0.084 e 0.64R^2 | 0.0699 e 0.78R^2
# com freezing temp: 0.0823 e 0.64r^2 e 0.65 e 0.8R^2
# both dropped: 0.0821 e 0.66R^2 e 0.7 e 0.775R^2


# with freezing temp 0.81 | 0.635
# without 0.812 | 0.631
