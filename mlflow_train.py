
if __name__ == "__main__":
    # Load the data
    data_path = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(data_path, sep=';')

    # Hold out 1 column for validation
    val_data = data.sample(
        n=1,
        random_state=42)

    # Remove the predictor
    val_data = val_data.drop(
        'quality',
        axis=1)

    # Remove from the trainin/testing data
    data = data.drop(val_data.index[0])

    # Save the data to upload to gcs later (create a folder first)
    data.to_csv(
        './training_data/train.csv',
        index=False)

    def eval_metrics(actual, pred):
        # compute relevant metrics
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def load_data():
        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        # The predicted column is "quality" which is a scalar from [3, 9]
        train_x = train.drop(["quality"], axis=1)
        test_x = test.drop(["quality"], axis=1)
        train_y = train[["quality"]]
        test_y = test[["quality"]]
        return train_x, train_y, test_x, test_y
    
    # train a model with given parameters
    def train(alpha, l1_ratio):
        np.random.seed(40)

        train_x, train_y, test_x, test_y = load_data()

        # Useful for multiple runs (only doing one run in this sample notebook)    
        with mlflow.start_run(run_name='example'):
            # Execute ElasticNet
            lr = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                random_state=42)

            lr.fit(train_x, train_y)

            # Evaluate Metrics
            predicted_qualities = lr.predict(test_x)
            (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

            # Print out metrics
            print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
            print("  RMSE: %s" % rmse)
            print("  MAE: %s" % mae)
            print("  R2: %s" % r2)

            # Log parameter, metrics, and model to MLflow
            mlflow.log_param(key="alpha", value=alpha)
            mlflow.log_param(key="l1_ratio", value=l1_ratio)
            mlflow.log_metric(key="rmse", value=rmse)
            mlflow.log_metrics({"mae": mae, "r2": r2})

            print("Save to: {}".format(mlflow.get_artifact_uri()))

            mlflow.sklearn.log_model(lr, "model")
