import pandas as pd

data_frame = pd.read_parquet("data_test/train-00000-of-00001.parquet", engine="pyarrow")

print(data_frame)