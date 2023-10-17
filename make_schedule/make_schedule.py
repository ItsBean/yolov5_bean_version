import pandas as pd

# Assuming the raw data is in a file named 'data.csv'
df = pd.read_csv('raw_data.csv', delimiter='\t')  # Assuming the delimiter is a tab character

print(df.columns)
print(df.shape)
# Show the first few rows to verify
print(df.head())
