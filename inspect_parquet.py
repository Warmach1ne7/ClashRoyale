import pandas as pd
import argparse

def print_distinct_arena_replay(parquet_path):
    df = pd.read_parquet(parquet_path)
    print(f"File: {parquet_path}")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(df.info())
    print(df.describe(include='all'))
    if 'arena' not in df.columns or 'replay' not in df.columns:
        print("The parquet file must contain 'arena' and 'replay' columns.")
        return
    pairs = df[['arena', 'replay']].drop_duplicates()
    print(f"Distinct arena/replay pairs: {len(pairs)}")
    for _, row in pairs.iterrows():
        print(f"{row['arena']}/{row['replay']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print distinct arena and replay values from a parquet file.")
    parser.add_argument("parquet", help="Path to the parquet file")
    args = parser.parse_args()
    print_distinct_arena_replay(args.parquet)