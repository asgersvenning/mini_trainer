"""Apply quality control model to gbifxdl parquet.
The model has been trained to infer if a given image contains
an image of an adult moth at a good resolution, 
or one of a number of other "classes".

Script included for reference on how quality filtering was done, 
but is NOT intended to actually be run.
"""

# # Create index of retained images from quality control
# Retains images that match the following heuristics:

# 1) If the inferred quality class is "Valid" or "Dead" **and** the confidence is 50% or higher.
# 2) The actual class must have 50 or more images **after** applying the first heuristic.

import csv
import os
from collections import Counter
from tempfile import tempdir

import pyarrow
import pyarrow.compute as cp
import pyarrow.dataset as ds
from pyarrow.parquet import ParquetWriter
from pyremotedata.implicit_mount import IOHandler
from tqdm.auto import tqdm

src = "quality/global_lepi.csv"
init_class_counts = Counter()
init_selected = []
non_selected = []

with open(src) as f:
    lines = f.readlines()
    reader = csv.DictReader(lines)
    for row in tqdm(reader, total=len(lines) - 1, desc="Selecting from predictions"):
        path, pred, conf = row["path"], row["prediction"], row["confidence"]
        if pred in ["Valid", "Dead"] and float(conf) >= 0.5:
            init_selected.append(path)
            init_class_counts.update([path.split("/")[0]])
        else:
            non_selected.append(path)

classes_with_enough_images = set([c for c, n in init_class_counts.items() if n >= 50])
class_counts = Counter()
selected = []
for proposed in init_selected:
    if (cls := proposed.split("/")[0]) in classes_with_enough_images:
        selected.append(proposed)
        class_counts.update([cls])
    else:
        non_selected.append(proposed)

summary_str = "\n".join(
    f"{lab:<20}: {{:>10}}" for lab in [
        "Total images", 
        "Passed QC", 
        "Enough per-class", 
        "Removed", 
        "Total classes", 
        "Retained classes"
    ]
).format(
    len(lines) - 1, len(init_selected), len(selected), len(non_selected), len(init_class_counts), len(class_counts)
)
print(summary_str)

with open("quality/selected_images.txt", "w") as f:
    f.writelines(selected)

with open("quality/non_selected_images.txt", "w") as f:
    f.writelines(non_selected)

# ## Apply selected index to parquet
# *Crucially this is done lazily on disk to avoid OOM on large parquet files!*

# Original parquet is downloaded from ERDA and updated parquet is uploaded to ERDA if it doesn't exist.

with IOHandler(user="On0rdRltgS", password="On0rdRltgS") as io:
    io.cd("global_lepi")
    src = io.pget("0032836-250426092105405_processing_metadata_postprocessed.parquet", tempdir)
    if src is None:
        raise RuntimeError("Remote parquet not downloaded?")

dst = os.path.join(tempdir, "0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet")
if os.path.exists(dst):
    os.remove(dst)
select_index = "quality/selected_images.txt"

data = ds.dataset(src, format="parquet")
if not isinstance(data, ds.Dataset):
    raise RuntimeError("...")

with open(select_index, "r") as f:
    selected = [line.strip() for line in f.readlines()]

selected_uuid = set([s.split("/")[1].split(".")[0] for s in selected])

scanner = ds.Scanner.from_dataset(
    data,
    filter=cp.is_in(
        cp.list_element(cp.split_pattern(cp.field("filename"), ".", max_splits=1), 0),
        pyarrow.array(selected_uuid)
    )
)

with (
    ParquetWriter(dst, scanner.dataset_schema, compression="ZSTD") as writer, 
    tqdm(total=scanner.count_rows(), desc="Writing parquet...") as pbar
    ):
    for batch in scanner.scan_batches():
        batch = batch.record_batch
        writer.write_batch(batch)
        pbar.update(batch.num_rows)

with IOHandler(user="On0rdRltgS", password="On0rdRltgS") as io:
    io.cd("global_lepi")
    if os.path.basename(dst) not in io.ls():
        io.put(dst)
        print("Updated parquet uploaded to ERDA!")
    else:
        print("Updated parquet already on ERDA!")

os.remove(dst)
os.remove(src)