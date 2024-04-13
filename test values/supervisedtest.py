import pandas as pd
import numpy as np

def generate_continuous_data(n_samples=1000):
    np.random.seed(0)
    age = np.random.normal(40, 10, n_samples)  # Normal distribution, mean=40, std=10
    income = np.random.normal(50000, 15000, n_samples)  # Normal distribution, mean=50000, std=15000
    education_years = np.random.normal(12, 2, n_samples)  # Normal distribution, mean=12, std=2
    house_price = 50000 + income * 0.3 + age * 1000 + education_years * 2000  # Linear combination

    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'Education Years': education_years,
        'House Price': house_price
    })
    return df

def generate_discrete_data(n_samples=1000):
    np.random.seed(1)
    age = np.random.normal(40, 10, n_samples)  # Normal distribution, mean=40, std=10
    income = np.random.normal(50000, 15000, n_samples)  # Normal distribution, mean=50000, std=15000
    education_years = np.random.normal(12, 2, n_samples)  # Normal distribution, mean=12, std=2
    employment_status = np.random.choice(['Employed', 'Unemployed', 'Self-Employed'], n_samples)

    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'Education Years': education_years,
        'Employment Status': employment_status
    })
    return df

# Generate the datasets
df_continuous = generate_continuous_data()
df_discrete = generate_discrete_data()

# Save the datasets to CSV files
df_continuous.to_csv('continuous_data.csv', index=False)
df_discrete.to_csv('discrete_data.csv', index=False)

print("Datasets have been generated and saved to 'continuous_data.csv' and 'discrete_data.csv'.")
