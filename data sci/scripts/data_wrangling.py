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
   
def encode_and_scale_data(data):
    
    #Sample DataFrame
    #pd.DataFrame({
    #    'Age': [25, 35, 45],
    #    'Gender': ['Male', 'Female', 'Female'],
    #    'Occupation': ['Engineer', 'Doctor', 'Artist']
    #})

    #Define the transformers for numerical and categorical data
    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    #Create a ColumnTransformer to apply the transformations
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, ['Age']),
        ('cat', categorical_pipeline, ['Gender', 'Occupation'])
    ])

    #Fit and transform the data
    df_transformed = preprocessor.fit_transform(data)

    #Accessing encoded feature names after fitting
    categorical_feature_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out()
    print(categorical_feature_names)

    #Combining feature names
    numerical_features = ['Age']
    all_features = numerical_features + list(categorical_feature_names)
    print(all_features)

    #Create DataFrame from transformed data
    df_encoded_scaled = pd.DataFrame(df_transformed, columns=all_features, index=data.index)
    print(df_encoded_scaled)



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
    
    print("pro data:")
    print(processed_data)

    final_data, error = encode_and_scale_data(processed_data)
    if error:
        return None, error  # Return early if encoding/scaling failed
    
    print(final_data)
    return final_data, None  # Successfully completed all steps, no error

