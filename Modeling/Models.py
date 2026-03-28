from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

from catboost import CatBoostRegressor

class Models:

    def __init__(self, data, target_name):

        self.data = data
        self.target_name = target_name


    def train_test_split(self):
        
        X = self.data.drop(self.target_name, axis = 1)
        y = self.data[self.target_name]
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


    # def fit(self):

    #     pass

    def get_metrics(self):

        pass

class Catboost(Models):
    
    def __init__(self, data, target_name):
        super().__init__(data, target_name)

        dtypes = data.dtypes
        cat_features = list(dtypes.loc[dtypes=='str'].index)
        self.cat_features = cat_features
        data[cat_features] = data[cat_features].fillna('missing')


    def fit(self, **kwargs):

        model = CatBoostRegressor(**kwargs, cat_features=self.cat_features)
        model.fit(self.X_train, self.y_train)

        self.model = model


    def get_metrics(self):

        pred_train = self.model.predict(self.X_train)
        pred_test = self.model.predict(self.X_test)

        rmse_train = root_mean_squared_error(self.y_train, pred_train)
        rmse_test = root_mean_squared_error(self.y_test, pred_test)

        print('RMSE train', rmse_train)
        print('RMSE test', rmse_test)