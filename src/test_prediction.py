import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def get_mae(candidate_n_estimator, train_X, val_X, train_y, val_y):
    model = RandomForestRegressor(n_estimators=candidate_n_estimator, random_state=1)
    model.fit(train_X, train_y)
    pred_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, pred_val)
    return mae


def get_n_estimators(train_X, val_X, train_y, val_y):
    maes = {}
    for n_estimator in range(100):
        maes[n_estimator+1] = get_mae(n_estimator+1, train_X, val_X, train_y, val_y)
    best_n_estimator = min(maes, key=maes.get)
    return best_n_estimator


def get_prediction():
    train_data = pd.read_csv("../data/train.csv")
    y = train_data.SalePrice
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd',
                'MSSubClass', 'OverallQual', 'OverallCond', 'GrLivArea']
    X = train_data[features]
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    n = get_n_estimators(train_X, val_X, train_y, val_y)
    print(n)
    prices_model = RandomForestRegressor(n_estimators=n, random_state=1)
    prices_model.fit(train_X, train_y)
    val_predictions = prices_model.predict(val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    print("Validation MAE = ${:,.0f}".format(val_mae))


get_prediction()
