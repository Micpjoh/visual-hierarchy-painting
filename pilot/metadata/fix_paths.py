from pathlib import Path
import pandas as pd

csv_path = Path.home() / "thesis_models/pilot/metadata/renaissance_pilot_final(pilot_dataset).csv"
out_path = Path.home() / "thesis_models/pilot/metadata/renaissance_pilot_server.csv"

clean_dir = Path.home() / "thesis_models/pilot/images/clean"
annot_dir = Path.home() / "thesis_models/pilot/images/annotated"

encodings_to_try = ["utf-8", "cp1252", "latin-1"]
separators_to_try = [",", ";", "\t"]

df = None
for enc in encodings_to_try:
    for sep in separators_to_try:
        try:
            df = pd.read_csv(csv_path, encoding=enc, sep=sep)
            print(f"Read CSV successfully with encoding={enc}, sep={repr(sep)}")
            print("Columns:", df.columns.tolist())
            break
        except Exception as e:
            print(f"Failed with encoding={enc}, sep={repr(sep)}: {e}")
    if df is not None:
        break

if df is None:
    raise RuntimeError("Could not read CSV with tried encodings/separators.")

def to_server_path(p, folder):
    name = Path(str(p)).name
    return str(folder / name)

df["local_image_path"] = df["local_image_path"].apply(lambda p: to_server_path(p, clean_dir))
df["annotated_image_path"] = df["annotated_image_path"].apply(lambda p: to_server_path(p, annot_dir))

df.to_csv(out_path, index=False, encoding="utf-8")
print(f"Saved: {out_path}")
print(df[["id", "local_image_path", "annotated_image_path"]].head())
