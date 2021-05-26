import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def get_output():
    train_data = pd.read_csv("../data/train.csv")
    test_data = pd.read_csv("../data/test.csv")
    y = train_data.SalePrice
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd',
                'MSSubClass', 'YrSold', 'MoSold']
    X = train_data[features]
    test_X = test_data[features]
    prices_model = RandomForestRegressor(n_estimators=68, random_state=1)
    prices_model.fit(X, y)
    test_predictions = prices_model.predict(test_X)
    output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_predictions})
    output.to_csv('../data/prediction.csv', index=False)


get_output()
