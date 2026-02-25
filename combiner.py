import sys
from pathlib import Path
import pandas as pd

# Add your files here (or pass file paths as CLI args: python combiner.py file1.csv file2.csv ...)
FILES = sorted(str(p) for p in Path(__file__).parent.glob("report*.csv") if p.is_file())

OUTPUT_FILE = "combined_reports.csv"

def load_csv(path: Path, src_ts: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize date columns for sorting
    if "Day" in df.columns:
        df["Day"] = pd.to_datetime(df["Day"], errors="coerce")
    if "Start Date" in df.columns:
        df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")
    # Track source file timestamp for "latest" conflict resolution
    df["_source_ts"] = pd.to_datetime(src_ts, unit="s")
    df["_source_file"] = path.name
    return df

def main():
    # Resolve files from CLI or FILES list
    file_list = [Path(p) for p in (sys.argv[1:] if len(sys.argv) > 1 else FILES)]
    file_list = [p for p in file_list if p.exists()]

    if not file_list:
        print("No input CSV files found. Add file paths to FILES or pass as CLI args.")
        sys.exit(1)

    dfs = []
    ref_cols = None
    for p in file_list:
        try:
            df = load_csv(p, p.stat().st_mtime)
            if df.empty:
                continue
            # Align columns to the first file's schema
            if ref_cols is None:
                ref_cols = list(df.columns)
            else:
                # Reindex to reference columns; add any new unseen columns to the end
                new_cols = [c for c in df.columns if c not in ref_cols]
                if new_cols:
                    ref_cols += new_cols
                df = df.reindex(columns=ref_cols, fill_value=pd.NA)
            dfs.append(df)
            print(f"Loaded {len(df):,} rows from {p.name}")
        except Exception as e:
            print(f"Skipping {p}: {e}")

    if not dfs:
        print("No data loaded from provided files.")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)

    # Sort chronologically; when dates tie, use source file timestamp to keep latest
    sort_cols = [c for c in ["Day", "Start Date"] if c in combined.columns]
    if sort_cols:
        sort_cols.append("_source_ts")
        combined = combined.sort_values(by=sort_cols, ascending=True, na_position="last")

    # Drop duplicates, keeping the latest (because we sorted ascending and keep='last')
    if "Campaign ID" in combined.columns:
        if "Day" in combined.columns:
            combined = combined.drop_duplicates(subset=["Campaign ID", "Day"], keep="last")
        elif "Start Date" in combined.columns:
            combined = combined.drop_duplicates(subset=["Campaign ID", "Start Date"], keep="last")
    elif "Campaign Name" in combined.columns:
        if "Day" in combined.columns:
            combined = combined.drop_duplicates(subset=["Campaign Name", "Day"], keep="last")
        elif "Start Date" in combined.columns:
            combined = combined.drop_duplicates(subset=["Campaign Name", "Start Date"], keep="last")

    # Write output (restore Day/Start Date to ISO strings for CSV)
    for col in ["Day", "Start Date"]:
        if col in combined.columns:
            combined[col] = pd.to_datetime(combined[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S").str.replace(" 00:00:00", "", regex=False)

    # Drop helper columns
    combined = combined.drop(columns=[c for c in ["_source_ts", "_source_file"] if c in combined.columns])

    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"Written {len(combined):,} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()