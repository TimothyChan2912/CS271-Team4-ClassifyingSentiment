import os
import pandas as pd
from sklearn.model_selection import train_test_split

os.makedirs("splits", exist_ok=True)
df = pd.read_parquet("data/processed/clean.parquet")

train, temp = train_test_split(df, test_size=0.2, random_state=42, stratify=df["sentiment"])
val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp["sentiment"])

for name, part in [("train", train), ("val", val), ("test", test)]:
    part[["id"]].to_csv(f"splits/{name}_ids.csv", index=False)
    print(name, len(part), part["sentiment"].value_counts().to_dict())