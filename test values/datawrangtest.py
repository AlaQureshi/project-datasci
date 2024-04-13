# Import pandas package
import pandas as pd

# Assign data
data = {'Name': ['Jai', 'Princi', 'Gaurav', 
	           'Anuj', 'Ravi', 'Natasha', 'Riya'],
             'Age': [17, 17, 18, 17, 18, 17, 17],
             'Gender': ['M', 'F', 'M', 'M', 'M', 'F', 'F'],
             'Marks': [90, 76, 'NaN', 74, 65, 'NaN', 71]}

# Convert into DataFrame
df = pd.DataFrame(data)

# Display data
df.to_csv('example_dataset.csv', index=False)
