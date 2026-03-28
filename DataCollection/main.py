from DataPreprocess import DataPreprocess

input_path = 'data/input'
output_path = 'data/output'

dp = DataPreprocess(input_path)
dp.main()



data = dp.preprocess_data
data.to_parquet(fr'{output_path}/data.parquet')

