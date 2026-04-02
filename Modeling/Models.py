from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
import numpy as np
from sklearn.preprocessing import StandardScaler
import time

from catboost import CatBoostRegressor
from models.MLP import MLPNET
import torch.nn as nn
from sklearn.neural_network import MLPRegressor

from loguru import logger
logger.add("models_logs.log")


def timecount(func):
    """Декоратор, измеряющий время выполнения функции"""
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Время начала
        result = func(*args, **kwargs)  # Вызов функции
        end_time = time.time()  # Время окончания
        print(f"Функция {func.__name__} выполнена за {end_time - start_time:.4f} сек.")
        return result
    return wrapper

class Models:

    def __init__(self, data, target_name, test_size = 0.3, stratify = True):

        self.data = data
        self.target_name = target_name
        self.test_size = test_size
        self.stratify = stratify

        dtypes = data.dtypes
        cat_features = list(dtypes.loc[dtypes=='str'].index)
        self.cat_features = cat_features
        data[cat_features] = data[cat_features].fillna('missing')

    def train_test_split(self):
        
        X = self.data.drop(self.target_name, axis = 1)
        y = self.data[self.target_name]
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=42)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        class_name = self.__class__.__name__
        logger.info(f"Class: {class_name} | test_size: {self.test_size}, train_shape: {X_train.shape}, test_shape: {X_test.shape}")

    def fit(self,**kwargs):
        print('nothing')

    def get_metrics(self):

        pred_train = self.model.predict(self.X_train)
        pred_test = self.model.predict(self.X_test)

        rmse_train = root_mean_squared_error(self.y_train, pred_train)
        rmse_test = root_mean_squared_error(self.y_test, pred_test)
        mape_train = 100*mean_absolute_percentage_error(self.y_train, pred_train)
        mape_test = 100*mean_absolute_percentage_error(self.y_test, pred_test)

        class_name = self.__class__.__name__
        logger.info(f"Class: {class_name} | RMSE train: {rmse_train:.4f}")
        logger.info(f"Class: {class_name} | RMSE test: {rmse_test:.4f}")
        logger.info(f"Class: {class_name} | MAPE train: {mape_train:.4f}")
        logger.info(f"Class: {class_name} | MAPE train: {mape_test:.4f}")

    def __log_hyperparmeters__(self, params):
        class_name = self.__class__.__name__
        logger.info(f"Class: {class_name} | hyperparams: {params}")

    def __save_model__(self):
        pass

    def main(self, params):
        self.train_test_split()
        self.__log_hyperparmeters__(params)
        self.fit(**params)
        self.get_metrics()


class FeatureProcessing:

    def __init__(self):
        pass

    def __preprocess_categorical_features__(self):
        for col in self.cat_features:
            vc = self.X_train[col].value_counts()/ len(self.X_train)
            self.X_train[col] = self.X_train[col].map(vc)
            self.X_test[col] = self.X_test[col].map(vc)

    def __simple_fillna__(self):
        self.X_train = self.X_train.fillna(0)
        self.y_train = self.y_train.fillna(0)
        self.X_test = self.X_test.fillna(0)
        self.y_test = self.y_test.fillna(0)

    def __normalizing__(self):

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        X_train = self.scaler_X.fit_transform(self.X_train.fillna(0))
        y_train = self.scaler_y.fit_transform(self.y_train.values.reshape(-1, 1)).ravel()
        
        X_test = self.scaler_X.transform(self.X_test.fillna(0))
        y_test = self.scaler_y.transform(self.y_test.values.reshape(-1, 1)).ravel()

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

class Catboost(Models):
    
    def __init__(self, data, target_name):
        super().__init__(data, target_name)


    def fit(self, **kwargs):

        model = CatBoostRegressor(**kwargs, cat_features=self.cat_features)
        model.fit(self.X_train, self.y_train)

        self.model = model


class MLPNet(Models,FeatureProcessing):
    
    def __init__(self,data, target_name, fs, sizes, activation = nn.ReLU()):
        Models.__init__(self, data, target_name)
        FeatureProcessing.__init__(self)
        self.fs = fs
        self.sizes = sizes
        self.activation = activation

    def fit(self, **kwargs):
        mlp = MLPNET(self.fs, self.sizes, self.activation)
        mlp.train_net(self.X_train, self.y_train, **kwargs)
        self.model = mlp

    def get_metrics(self):

        pred_train = self.model.predict(self.X_train)
        pred_test = self.model.predict(self.X_test)

        self.X_train = self.scaler_X.inverse_transform(self.X_train)
        self.X_test = self.scaler_X.inverse_transform(self.X_test)
        self.y_train = self.scaler_y.inverse_transform(self.y_train.reshape(-1, 1)).ravel()
        self.y_test = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1)).ravel()
        pred_train = self.scaler_y.inverse_transform(pred_train.reshape(-1, 1)).ravel()
        pred_test = self.scaler_y.inverse_transform(pred_test.reshape(-1, 1)).ravel()

        rmse_train = root_mean_squared_error(self.y_train, pred_train)
        rmse_test = root_mean_squared_error(self.y_test, pred_test)
        mape_train = 100*mean_absolute_percentage_error(self.y_train, pred_train)
        mape_test = 100*mean_absolute_percentage_error(self.y_test, pred_test)

        class_name = self.__class__.__name__
        logger.info(f"Class: {class_name} | RMSE train: {rmse_train:.4f}")
        logger.info(f"Class: {class_name} | RMSE test: {rmse_test:.4f}")
        logger.info(f"Class: {class_name} | MAPE train: {mape_train:.4f}")
        logger.info(f"Class: {class_name} | MAPE train: {mape_test:.4f}")


    @timecount
    def main(self, params):
        
        self.train_test_split()
         # Добавляется обработка категориальных признаков
        self.__preprocess_categorical_features__()
        self.__normalizing__()
        # self.__simple_fillna__()
        self.__log_hyperparmeters__(params)
        # print(params)
        self.fit(**params)
        self.get_metrics()

        
class MLPRegressor_sklearn(Models,FeatureProcessing):

    def __ini__(self, data, target_name):

        Models.__init__(self, data, target_name)
        FeatureProcessing.__init__(self)

    @timecount
    def fit(self,**kwargs):

        model = MLPRegressor(random_state=1, **kwargs)
        model.fit(self.X_train, self.y_train)
        self.model = model

    def get_metrics(self):

        pred_train = self.model.predict(self.X_train)
        pred_test = self.model.predict(self.X_test)

        self.X_train = self.scaler_X.inverse_transform(self.X_train)
        self.X_test = self.scaler_X.inverse_transform(self.X_test)
        self.y_train = self.scaler_y.inverse_transform(self.y_train.reshape(-1, 1)).ravel()
        self.y_test = self.scaler_y.inverse_transform(self.y_test.reshape(-1, 1)).ravel()
        pred_train = self.scaler_y.inverse_transform(pred_train.reshape(-1, 1)).ravel()
        pred_test = self.scaler_y.inverse_transform(pred_test.reshape(-1, 1)).ravel()

        rmse_train = root_mean_squared_error(self.y_train, pred_train)
        rmse_test = root_mean_squared_error(self.y_test, pred_test)
        mape_train = 100*mean_absolute_percentage_error(self.y_train, pred_train)
        mape_test = 100*mean_absolute_percentage_error(self.y_test, pred_test)

        class_name = self.__class__.__name__
        logger.info(f"Class: {class_name} | RMSE train: {rmse_train:.4f}")
        logger.info(f"Class: {class_name} | RMSE test: {rmse_test:.4f}")
        logger.info(f"Class: {class_name} | MAPE train: {mape_train:.4f}")
        logger.info(f"Class: {class_name} | MAPE train: {mape_test:.4f}")

    
    @timecount
    def main(self, params):

        self.train_test_split()
         # Добавляется обработка категориальных признаков
        self.__preprocess_categorical_features__()
        self.__normalizing__()
        self.__log_hyperparmeters__(params)
        # print(params)
        self.fit(**params)
        self.get_metrics()