import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def clean_data(dataset):
    df = pd.read_csv(f"../data/raw/{dataset}.csv",low_memory=False)

    df = df[df["Open"] == 1]
    df = df.reset_index(drop=True)

    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    table = pa.Table.from_pandas(df)
    pq.write_table(table, f"../data/processed/{dataset}_clean.parquet")

    print("✅ Limpieza básica de los datos realizada correctamente")
