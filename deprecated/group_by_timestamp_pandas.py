import pandas as pd

def group_by_timestamp(input_path, gap_threshold_s=0.1):
    df = pd.read_csv(input_path, sep=",", encoding="utf-8")

    # Combine datetime and subsectime into a Unix timestamp in seconds
    ts = pd.to_datetime(df["datetime"].str.strip(), format="%Y:%m:%d %H:%M:%S").astype("int64") / 1e6
    ts += df["subsectime"].astype(float) / 1000.0

    # Assign group IDs based on timestamp gaps larger than threshold
    df["shot_id"] = (ts.sort_values().diff() > gap_threshold_s).cumsum()
    df["shot_id"] = df["shot_id"].fillna(0).astype(int)

    df.to_csv(input_path, index=False, encoding="utf-8")
    print(f"Append shot_id column to {input_path}")
    return df

if __name__ == "__main__":
    import sys
    group_by_timestamp(sys.argv[1])
