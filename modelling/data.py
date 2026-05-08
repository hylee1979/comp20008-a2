import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from modelling.config import ALL_PREDICTOR_COLS, SESSION_COL, TARGET_COL
from modelling.weather_category import weather_category

def fix_missing_values(df):
    # above_ground_numeric
    # fill 0 to missing values in above_ground_numeric if location_missing is True
    df.loc[df["location_missing"], "above_ground_numeric"] = df.loc[df["location_missing"], "above_ground_numeric"].fillna(0)
    # extract rows where is_above_ground is True and fill missing values in above_ground_numeric with median
    median_value = df.loc[df["is_above_ground"], "above_ground_numeric"].median()
    df.loc[df["is_above_ground"], "above_ground_numeric"] = df.loc[df["is_above_ground"], "above_ground_numeric"].fillna(median_value)
    assert df["above_ground_numeric"].isna().sum() == 0, "above_ground_numeric still has missing values after imputation"

    # fix Litter missing values by filling "Unknown"
    df["Litter"] = df["Litter"].fillna("Unknown")
    assert df["Litter"].isna().sum() == 0, "Litter still has missing values after imputation"
    return df

def extract_weather_condition(df):
    weather_raw = df["Sighter Observed Weather Data"]
    df['weather_condition'] = weather_category(weather_raw)
    assert df['weather_condition'].isna().sum() == 0, "weather_condition still has missing values after extraction"
    return df

def encode_categorical_columns(df):
    # split the string by comma and strip the whitespace
    df['Hectare Conditions'] = df['Hectare Conditions'].apply(lambda x: [item.strip() for item in x.split(',')])
    # use MultiLabelBinarizer to encode the Hectare Conditions column, named as hectare_condition_<condition>
    mlb = MultiLabelBinarizer()
    hectare_conditions_encoded = mlb.fit_transform(df['Hectare Conditions'])
    hectare_conditions_df = pd.DataFrame(hectare_conditions_encoded, columns=mlb.classes_)
    hectare_conditions_df.columns = [f"hectare_condition_{col}" for col in hectare_conditions_df.columns]
    df = pd.concat([df, hectare_conditions_df], axis=1)
    # drop the original Hectare Conditions column
    df = df.drop(columns=['Hectare Conditions'])

    # these can be one-hot encoded beforehand since they are defined in the description and have a limited number of possible values
    ohe_columns = ['Shift', 'Age', 'Primary Fur Color', 'Litter', 'weather_condition']
    # # check the unique values of these columns to make sure they are suitable for one-hot encoding
    # for col in ohe_columns:
    #     print(f"{col}: {df[col].unique()}")
    # one-hot encode the columns and concatenate with the original dataframe
    df = pd.get_dummies(df, columns=ohe_columns, prefix=ohe_columns)

    return df

def preprocess_loaded_data(df):
    df = fix_missing_values(df)
    df = extract_weather_condition(df)
    # df = encode_categorical_columns(df)
    return df

def select_model_columns(df):
    # select only the columns needed for modelling
    columns = [TARGET_COL, SESSION_COL] + ALL_PREDICTOR_COLS
    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(f"WARNING: missing columns ({len(missing)}): {missing}")
    df = df[columns]
    return df

def select_response_subset(df):
    # select only the rows with response values in [0, 1]
    print(f"Original class balance: {df[TARGET_COL].value_counts(dropna=False).to_dict()}")
    df = df[df[TARGET_COL].isin([0, 1])]
    print(f"Filtered class balance: {df[TARGET_COL].value_counts(dropna=False).to_dict()}")
    return df

def load_processed_data(path=None):
    # load either parquet or csv depending on the file extension
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    df = preprocess_loaded_data(df)
    df = select_model_columns(df)
    df = select_response_subset(df)

    print(f"Loaded {len(df)} rows x {df.shape[1]} columns from {path}")

    return df
