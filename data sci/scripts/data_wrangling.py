import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        return data, None
    except Exception as e:
        return None, str(e)
    
def preprocess_data(df):
    """Process data by filling and converting dates."""
    try:
        df.ffill(inplace=True)
        df.dropna(inplace=True)  # Drop any remaining missing values
        return df, None  # Successfully processed data, no error
    except Exception as e:
        return None, str(e)  # Return None for data and an error message

def encode_and_scale_data(df):
    """Encode categorical features and scale numerical features."""
    try:
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

        numerical_pipeline = Pipeline([
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_features),
                ('cat', categorical_pipeline, categorical_features)
            ])

        df_transformed = preprocessor.fit_transform(df)
        columns = numerical_features + list(preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names(categorical_features))
        df_encoded_scaled = pd.DataFrame(df_transformed, columns=columns, index=df.index)
        return df_encoded_scaled, None  # Successfully encoded and scaled data, no error
    except Exception as e:
        return None, str(e)  # Return None for data and an error message

def execute_full_wrangling(file_path):
    """Execute all data wrangling steps in sequence."""
    data, error = load_data(file_path)
    if error:
        return None, error  # Return early if loading data failed
    
    print("Input data:")
    print(data)

    processed_data, error = preprocess_data(data)
    if error:
        return None, error  # Return early if preprocessing failed

    final_data, error = encode_and_scale_data(processed_data)
    if error:
        return None, error  # Return early if encoding/scaling failed

    return final_data, None  # Successfully completed all steps, no error

