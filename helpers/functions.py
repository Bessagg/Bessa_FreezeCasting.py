import os


def best_model_results(best_model, test, train):
    r2 = best_model.model_performance(test_data=test)['r2']
    mae = best_model.model_performance(test_data=test)['mae']
    print("Mean residual deviance")
    mrd = best_model.model_performance(test_data=test)['mean_residual_deviance']

    r2, mae, mrd = "{:.04f}".format(r2), "{:.04f}".format(mae), "{:.04f}".format(mrd)

    print("Mean residual deviance: ", mrd)
    print("Mean average error: ", mae)
    print("Pearson Coefficient R^2: ", r2)
    print("Difference of r^2 train - test: ",
          best_model.model_performance(test_data=train)['r2'] - best_model.model_performance(test_data=test)['r2'])
    return r2, mae, mrd


def save_preprocessor(preprocessor, preprocessor_filename):
    import pickle
    with open(preprocessor_filename + '.pickle', 'wb') as handle:
        pickle.dump(preprocessor, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_selected_model_from_each_folder(selected_models_path="selected_models"):
    """Gets list of files of all models in selected_models folder (not automl or stacked)"""
    folders = os.listdir(selected_models_path)
    selected_files = []
    # add files with highest [3] of each folder
    for folder in folders:
        if not folder.__contains__('AutoML') or folder.__contains__('Stacked'):
            files = os.listdir(os.path.join(selected_models_path, folder))
            if not files == []:
                print(folder, files)
                # sorted_files = sorted(files, key=lambda x: float(x.split('_')[3]))
                # selected_file = sorted_files[0]
                for file in files:
                    selected_files.append(os.path.join(selected_models_path, folder, file))  # files[0]
    return selected_files


def save_varimps(best_model, model_name):
    varimps = best_model.varimp(use_pandas=True)[['variable', 'scaled_importance']]
    # Open the file in write mode
    with open(f"varimps/{model_name}.txt", "w") as f:
        f.write(varimps.to_string())

    print("Varimps written to the file successfully!")
