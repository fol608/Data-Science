import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shutil

df = pd.read_csv("data/raw/train.csv",low_memory=False)

df = df[df["Open"] == 1]
df = df.reset_index(drop=True)

df["Date"] = pd.to_datetime(df["Date"]).dt.date

table = pa.Table.from_pandas(df)
pq.write_table(table, "train_clean.parquet")
shutil.move("train_clean.parquet","data/processed/")

print("Limpieza básica de los datos realizada correctamente")