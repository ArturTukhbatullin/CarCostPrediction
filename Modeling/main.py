import pandas as pd
from Models import Catboost, MLPNet

path = '../DataCollection/data/output'
target_name = 'cost'

data = pd.read_parquet(fr'{path}/data.parquet')
data = data.drop(['id','url_hashed'], axis = 1)


# # Построение Catboost модели
# cb = Catboost(data, target_name)
# params = {"iterations":1000, "depth":5, "verbose":100}
# cb.main(params)


# Построение MLP нейронной сети с Pytorch
fs = data.shape[1] - 1
sizes = [fs,64,1]
mlp = MLPNet(data, target_name, fs, sizes)
params = {"epochs":1_000, "lr":0.01, "verbose":100}
mlp.main(params)


