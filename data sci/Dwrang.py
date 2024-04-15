import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

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
   

def encode_and_scale_data(data, manual_num_cols=None, manual_cat_cols=None):
    # If manual overrides are provided, use them. Otherwise, detect automatically.
    if manual_num_cols is not None:
        numerical_cols = manual_num_cols
    else:
        numerical_cols = data.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    
    if manual_cat_cols is not None:
        categorical_cols = manual_cat_cols
    else:
        categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns

    # Define the transformers for numerical and categorical data
    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a ColumnTransformer to apply the transformations
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    # Fit and transform the data
    df_transformed = preprocessor.fit_transform(data)

    # Accessing encoded feature names after fitting
    categorical_feature_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_cols)
    print("Categorical Feature Names:", categorical_feature_names)

    # Combining feature names
    all_features = list(numerical_cols) + list(categorical_feature_names)
    print("All Feature Names:", all_features)

    # Create DataFrame from transformed data
    df_encoded_scaled = pd.DataFrame(df_transformed, columns=all_features, index=data.index)
    print("Transformed DataFrame:\n", df_encoded_scaled)

    return df_encoded_scaled

def execute_full_wrangling():
    """Execute all data wrangling steps in sequence using a hardcoded file path."""
    # Hardcoded file path
    file_path = r'E:\DOWNLOADS\project-datasci\test values\datawrang (2).csv'

    data, error = load_data(file_path)
    if error:
        return None, error  # Return early if loading data failed
    print(data)
    processed_data, error = preprocess_data(data)
    if error:
        return None, error  # Return early if preprocessing failed
    print(processed_data)
    final_data, error = encode_and_scale_data(processed_data)
    if error:
        return None, error  # Return early if encoding/scaling failed
    print(final_data)
    return final_data, None  # Successfully completed all steps, no error

execute_full_wrangling()