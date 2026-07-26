import csv
from datetime import datetime

def parse_timestamp(datetime_str: str, subsectime_str: str) -> float:
    """Combine datetime and subsectime into a Unix timestamp in seconds."""
    dt = datetime.strptime(datetime_str.strip(), "%Y:%m:%d %H:%M:%S")
    subsec = float(subsectime_str.strip()) / 1000.0
    return dt.timestamp() + subsec


def assign_group_ids(rows, gap_threshold=0.1):
    """Assign group IDs based on timestamp gaps larger than threshold."""
    # Attach timestamp to each row for sorting
    indexed = []
    for i, row in enumerate(rows):
        ts = parse_timestamp(row["datetime"], row["subsectime"])
        indexed.append((ts, i, row))

    indexed.sort(key=lambda x: x[0])

    group_ids = [0] * len(rows)
    current_group = 0
    prev_ts = None

    for ts, original_index, row in indexed:
        if prev_ts is not None and (ts - prev_ts) > gap_threshold:
            current_group += 1
        group_ids[original_index] = current_group
        prev_ts = ts

    return group_ids


def group_by_timestamp(input_path="IMAGE_METAINFO.csv", output_path="IMAGE_METAINFO_grouped.csv"):
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    group_ids = assign_group_ids(rows, gap_threshold=0.1)

    new_fieldnames = fieldnames + ["shot_id"]
    for row, gid in zip(rows, group_ids):
        row["shot_id"] = gid

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows with group_id to {output_path}")


if __name__ == "__main__":
    group_by_timestamp()
