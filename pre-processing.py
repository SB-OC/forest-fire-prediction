import pandas as pd
df = pd.read_csv('/mnt/import/Algerian_forest_fires_dataset_CLEANED.csv')

print(f"df columns: {df.columns}")


df.columns
df.drop(['day','month','year'], axis=1, inplace=True)
df.to_csv('/mnt/dataset/Algerian_forest_fires_dataset_CLEANED_NEW.csv', index=False)


print("CSV file updated!")
