import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler


import pandas as pd

def load_data(filepath):
    try:
        # Read the data into a DataFrame initially
        initial_data = pd.read_csv(filepath, header=0)
        
        # Convert the DataFrame to a tuple of tuples
        data_tuple = tuple(map(tuple, initial_data.values))
        
        # Convert the tuple back to a DataFrame with specified column names
        data = pd.DataFrame(data_tuple, columns=['Age', 'Income', 'Education Years', 'House Price'])
        
        # Check if the data is indeed a DataFrame
        if isinstance(data, pd.DataFrame):
            # Remove rows with any missing values
            clean_data = data.dropna()
            print(clean_data)
            return clean_data
        else:
            print("Data is not a DataFrame.")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def preprocess_data(data, target_column):
    if data is None or data.empty:
        print("No data available to process.")
        return None, None  # Ensure two items are returned even in case of no data

    try:
        y = data[target_column]
        X = data.drop(columns=[target_column])
        X = X.dropna()  # Optionally, handle missing data here
        y = y[X.index]  # Ensure target aligns with the features after dropping NA
        return X, y
    except KeyError:
        print(f"Column '{target_column}' not found in the data.")
        return None, None
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        return None, None


def train_model(X_train, y_train, model_type='linear_regression'):
    """ Train a model based on the specified type. """
    if model_type == 'linear_regression':
        model = LinearRegression()
    elif model_type == 'knn_regression':
        model = KNeighborsRegressor()
    elif model_type == 'svr':
        model = SVR()
    elif model_type == 'logistic_regression':
        model = LogisticRegression()
    elif model_type == 'perceptron':
        model = Perceptron()
    elif model_type == 'svm':
        model = SVC()
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier()
    elif model_type == 'knn_classifier':
        model = KNeighborsClassifier()
    elif model_type == 'naive_bayes':
        model = GaussianNB()
    else:
        raise ValueError("Invalid model type provided.")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, continuous=True):
    """ Evaluate the model using the appropriate metric. """
    y_pred = model.predict(X_test)
    if continuous:
        mse = mean_squared_error(y_test, y_pred)
        return {'Mean Squared Error': mse}
    else:
        accuracy = accuracy_score(y_test, y_pred)
        return {'Accuracy': accuracy}

def execute_full_supervised_learning(target_column, model_type='logistic_regression'):
    """Execute all supervised learning steps in sequence."""
    file_path = r'E:\DOWNLOADS\project-datasci\test values\continuous_data.csv'    
    data = load_data(file_path)
    X, y = preprocess_data(data, target_column)
    model, X_test, y_test = train_model(X, y, model_type)
    evaluation_results = evaluate_model(model, X_test, y_test)
    print('results')
    print(evaluation_results)
    return evaluation_results

execute_full_supervised_learning('House Price')