import pandas as pd
from Models import Catboost

path = '../DataCollection/data/output'
target_name = 'cost'

data = pd.read_parquet(fr'{path}/data.parquet')
print(data.columns)
data = data.drop(['id','url_hashed'], axis = 1)


cb = Catboost(data, target_name)
# params = {"iterations":1000, "learning_rate":0.1, "depth":5, "verbose": 100}
params = {"iterations":1000, "depth":5, "verbose": 100}
cb.train_test_split()
cb.fit(**params)
cb.get_metrics()

