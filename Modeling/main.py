import pandas as pd
import pickle
import os


class Prediction:

    def __init__(self, model_name, case_to_pred):

        self.model_name = model_name
        self.case_to_pred = case_to_pred
        self.model = pickle.load(open(fr'Modeling/saved_models/{model_name}.pickle', 'rb'))


    def get_predict(self):
        pred = self.model.predict(self.case_to_pred)
        if len(pred)==1:
            return pred[0]
        return pred
    

# if __name__ == '__main__':
#     input_file = os.getenv('INPUT_FILE', 'cases_to_pred/x_test_0.json')
#     case_to_pred = pd.read_json(input_file, orient='records', lines=True)
#     print(case_to_pred.T)
#     print(case_to_pred.columns)

#     p = Prediction('Catboost_model',case_to_pred)
#     print(p.get_predict())