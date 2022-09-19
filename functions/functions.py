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
