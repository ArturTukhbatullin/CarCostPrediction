import pandas as pd
import numpy as np
import time
import hashlib

def timecount(func):
    """Декоратор, измеряющий время выполнения функции"""
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Время начала
        result = func(*args, **kwargs)  # Вызов функции
        end_time = time.time()  # Время окончания
        print(f"Функция {func.__name__} выполнена за {end_time - start_time:.4f} сек.")
        return result
    return wrapper


# todo hash url

class DataPreprocess:


    def __init__(self, input_path):

        self.input_path = input_path

    @timecount
    def read_input_file(self):

        input_data = pd.read_parquet(fr'{self.input_path}/autoru.parquet')
        self.input_data = input_data

    @timecount
    def preprocess(self):

        data = self.input_data

        # Разбиваю название автомобиля на бренд и модель
        data['brand'] = data['name'].apply(lambda x: x.split()[0])
        data['model'] = data['name'].apply(lambda x: " ".join(x.split()[1:]))
        data['brand'] = data['brand'].astype(str)
        data['model'] = data['model'].astype(str)

        # Хеширую url
        data['url_hashed'] = data['url'].apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
        data.drop(['url'], axis = 1, inplace = True)
        data['url_hashed'] = data['url_hashed'].astype(str)

        # Пробег
        data['millege'] = np.where(data['millege']=='Новый', 0, data['millege'])
        data['millege'] = data['millege'].str.replace(' км','')
        data['millege'] = data['millege'].astype(float)

        # Объем двигателя
        data['engine_volume'] = np.where(data['engine_volume'].str.contains('л.с.'), np.nan,
                                         data['engine_volume'].str.replace(' л', ''))
        data['engine_volume'] = data['engine_volume'].astype(float)

        # Мощность мотора (оставляю как строку, т.к. иногда измеряется в киловаттах)
        data['motor_power'] = data['motor_power'].str.replace(' ', '')
        data['motor_power'] = data['motor_power'].fillna('missing')
        data['motor_power'] = data['motor_power'].astype(str)

        # Тип двигателя
        data['fuel_type'] = data['fuel_type'].str.lstrip().fillna('missing').astype(str)

        # Тип кузова
        data['body_type'] = data['body_type'].str.lstrip().fillna('missing').astype(str)

        # Тип коробки
        data['gearbox_type'] = data['gearbox_type'].str.lstrip().fillna('missing').astype(str)

        # Число владельцев
        data['owners_num'] = data['owners_num'].str.lstrip().fillna('missing').astype(str)

        # Конфигурация
        data['configuration'] = data['configuration'].str.lstrip().fillna('missing').astype(str)

        # Тип руля
        data['steering_wheel_type'] = data['steering_wheel_type'].str.lstrip().fillna('missing').astype(str)

        # Цвет
        data['color'] = data['color'].str.lstrip().fillna('missing').astype(str)

        # Таргет (стоимость авто)
        data['cost'] = data['cost'].astype(float)

        self.preprocess_data = data


    @timecount
    def main(self):

        self.read_input_file()
        self.preprocess()