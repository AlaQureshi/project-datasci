import pandas as pd
import numpy as np

# Creating a DataFrame with numeric and categorical data
data = {
    'Age': np.random.randint(18, 65, size=100),
    'Salary': np.random.normal(50000, 15000, 100),
    'Department': np.random.choice(['HR', 'Marketing', 'Finance', 'Sales'], 100),
    'Gender': np.random.choice(['Male', 'Female'], 100)
}

df = pd.DataFrame(data)
df['Salary'] = df['Salary'].round(2)  # Rounding salary values to two decimal places

# Save the DataFrame to a CSV file
file_path = 'test_data.csv'
df.to_csv(file_path, index=False)

print(f"Data file created: {file_path}")
