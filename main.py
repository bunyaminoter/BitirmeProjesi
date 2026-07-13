import pandas as pd

train_df = pd.read_csv(r"C:\Projelerim\BitirmeProjesi\AUTSL\train.csv")

print("Boyut:", train_df.shape)
print()
print("Kolonlar:")
print(train_df.columns)
print()
print(train_df.head())