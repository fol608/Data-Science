from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import joblib

def get_features(dataset):
    # 1️⃣ Cargar dataset limpio
    df = pd.read_parquet(f"../data/processed/{dataset}_clean.parquet")
    df['Date'] = pd.to_datetime(df['Date'])

    # 2️⃣ Partes de fecha
    year, month, day = [], [], []
    for i in range(0, df.shape[0]):
        year.append(df["Date"][i].year)
        month.append(df["Date"][i].month)
        day.append(df["Date"][i].day)
    df = df.join(pd.DataFrame({"Year": year, "Month": month, "Day": day}))

    # 🧠 Nuevas features temporales
    df['DayOfWeek'] = df['Date'].dt.dayofweek + 1  # 1=Lunes, 7=Domingo
    df['IsWeekend'] = (df['DayOfWeek'] >= 6).astype(int)
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Quarter'] = df['Date'].dt.quarter
    df['trend'] = df.groupby('Store').cumcount()
    # Interacción
    df['Promo_SchoolHoliday'] = df['Promo'] * df['SchoolHoliday']

    # 3️⃣ TRAIN
    if dataset == "train":
        # Eliminar columnas no usadas
        df = df.drop(columns=[c for c in ['Open', 'Customers'] if c in df.columns])

        # Lags y rolling (solo con pasado)
        df['Sales_lag_1'] = df.groupby('Store')['Sales'].shift(1)
        df['Sales_lag_7'] = df.groupby('Store')['Sales'].shift(7)
        df['rolling_mean_7'] = df.groupby('Store')['Sales'].transform(lambda x: x.shift(1).rolling(7).mean())
        df['rolling_std_7'] = df.groupby('Store')['Sales'].transform(lambda x: x.shift(1).rolling(7).std())

        # ✅ Eliminar filas sin histórico suficiente
        df = df.dropna(subset=['Sales_lag_1', 'Sales_lag_7', 'rolling_mean_7'])

        # One-hot para categóricas
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        if len(categorical_columns) > 0:
            one_hot_encoded = encoder.fit_transform(df[categorical_columns])
            one_hot_df = pd.DataFrame(one_hot_encoded,
                                      columns=encoder.get_feature_names_out(categorical_columns),
                                      index=df.index)
            df_encoded = pd.concat([df, one_hot_df], axis=1)
            df_encoded = df_encoded.drop(categorical_columns, axis=1)
        else:
            df_encoded = df.copy()

        # Guardar encoder
        joblib.dump(encoder, "../models/encoder.pkl")

        # Eliminar 'Date' antes de guardar
        if 'Date' in df_encoded.columns:
            df_encoded = df_encoded.drop(['Date'], axis=1)

        # Guardar parquet
        pq.write_table(pa.Table.from_pandas(df_encoded),
                       f"../data/processed/{dataset}_features.parquet")
        print("✅ Features de TRAIN creadas con nuevas variables y encoder guardado")

    # 4️⃣ TEST
    else:
        # Mantener Date
        df = df.drop(columns=[c for c in ['Open'] if c in df.columns])

        # Cargar histórico del train para calcular lags
        df_train = pd.read_parquet("../data/processed/train_clean.parquet")
        df_train['Date'] = pd.to_datetime(df_train['Date'])

        full = pd.concat([df_train[['Store','Date','Sales']], df[['Store','Date']]], ignore_index=True)
        full = full.sort_values(['Store','Date'])

        full['Sales_lag_1'] = full.groupby('Store')['Sales'].shift(1)
        full['Sales_lag_7'] = full.groupby('Store')['Sales'].shift(7)
        full['rolling_mean_7'] = full.groupby('Store')['Sales'].shift(1).rolling(7).mean()
        full['rolling_std_7'] = full.groupby('Store')['Sales'].shift(1).rolling(7).std()

        # Unir al test
        df = pd.merge(df,
                      full[['Store','Date','Sales_lag_1','Sales_lag_7','rolling_mean_7','rolling_std_7']],
                      on=['Store','Date'],
                      how='left')

        for col in ['Sales_lag_1', 'Sales_lag_7', 'rolling_mean_7', 'rolling_std_7']:
            df[col] = df[col].fillna(df[col].median())

        # Cargar encoder entrenado
        encoder = joblib.load("../models/encoder.pkl")
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        if len(categorical_columns) > 0:
            one_hot_encoded = encoder.transform(df[categorical_columns])
            one_hot_df = pd.DataFrame(one_hot_encoded,
                                      columns=encoder.get_feature_names_out(categorical_columns),
                                      index=df.index)
            df_encoded = pd.concat([df, one_hot_df], axis=1)
            df_encoded = df_encoded.drop(categorical_columns, axis=1)
        else:
            df_encoded = df.copy()

        # Quitar columnas innecesarias
        drop_cols = [c for c in ['Date', 'Id'] if c in df_encoded.columns]
        if drop_cols:
            df_encoded = df_encoded.drop(columns=drop_cols)

        # Alinear columnas con train
        train_cols = pd.read_parquet("../data/processed/train_features.parquet").columns
        train_cols_wo_target = [c for c in train_cols if c != 'Sales']

        for c in train_cols_wo_target:
            if c not in df_encoded.columns:
                df_encoded[c] = 0
        extra_cols = [c for c in df_encoded.columns if c not in train_cols_wo_target]
        if extra_cols:
            print(f"⚠️ Eliminando columnas extra en test: {extra_cols}")
            df_encoded = df_encoded.drop(columns=extra_cols)

        df_encoded = df_encoded[train_cols_wo_target]

        # Guardar
        pq.write_table(pa.Table.from_pandas(df_encoded),
                       f"../data/processed/{dataset}_features.parquet")
        print("✅ Features de TEST creadas (con nuevas variables, NaN conservados)")
