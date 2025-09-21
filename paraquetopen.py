import pandas as pd

# read the parquet file
df = pd.read_parquet(r"C:\Users\devri\nisram-hindi-text-0.0\data\train-00000-of-00006.parquet")

# look at first 5 rows
print(df.head())
df.to_csv("output.txt", sep="\t", index=False)

print("Converted to output.txt ✅")