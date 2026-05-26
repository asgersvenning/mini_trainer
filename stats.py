import math
import os
import queue
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map

from mini_trainer.deploy import Predictor
from mini_trainer.hierarchical.gbif import id_to_name
from mini_trainer.utils.io import is_image
from mini_trainer.utils.parquet import iter_parquet

if __name__ == "__main__":
    model = Predictor()

    # --- 0. Process parquet and get test-set files for all species
    pq_file = "examples/global_lepi/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet"
    nrow = pq.read_metadata(pq_file).num_rows
    test_files = {}
    for e in tqdm(iter_parquet(pq_file, ["speciesKey", "set", "filename"]), unit="row", total=nrow):
        sp_id, set_id, file = int(e["speciesKey"]), int(e["set"]), str(e["filename"])
        if sp_id not in test_files:
            test_files[sp_id] = set()
        if set_id == 0:
            test_files[sp_id].add(file)

    # --- 1. Setup and Resuming ---
    csv_file = "test_sp_accuracy_results.csv"
    processed_ids = set()
    reader = model.reader

    if os.path.exists(csv_file):
        with open(csv_file, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                if line.strip():
                    processed_ids.add(int(line.split(",")[0]))
        print(f"Resuming: Found {len(processed_ids)} already processed species.")

    print("Load europe list")
    with open("europe_gbif_id_list.txt") as f:
        europe_list = list(map(int, filter(bool, map(str.strip, f.readlines()))))

    pending_list = [sp for sp in europe_list if sp not in processed_ids]

    _local_template = "examples/global_lepi/images/{}"
    _ece_template = "/home/george/data/classif/global_lepi/images/{}/"
    max_per_species = 256

    # --- 2. The Worker Function ---
    # maxsize=5 allows up to 5 fully downloaded batches to buffer in RAM
    download_queue = queue.Queue(maxsize=10)

    def process_species(sp_id):
        sp_name = "unresolved"
        dir = "unknown"
        try:
            sp_name = id_to_name(sp_id)
            dir = _local_template.format(sp_id)
            src = _ece_template.format(sp_id)
            abs_dir = os.path.abspath(dir)

            list_cmd = ["ssh", "ECE", f"ls -U1 {src} 2>/dev/null"]
            try:
                list_process = subprocess.run(list_cmd, capture_output=True, text=True, check=True)
                files_to_download = list_process.stdout
            except subprocess.CalledProcessError:
                download_queue.put((sp_id, sp_name, dir, [], f'error: failed to run `{" ".join(list_cmd)}`'))
                return

            if not files_to_download.strip():
                download_queue.put((sp_id, sp_name, dir, [], "empty"))
                return
            
            try:
                files_in_test = [
                    file.strip() 
                    for file in set(files_to_download.splitlines())
                    if file in test_files[sp_id]
                ]
                if len(files_in_test) > max_per_species:
                    files_in_test = files_in_test[:max_per_species]
            except Exception:
                download_queue.put((sp_id, sp_name, dir, [], 'error: failed while filtering for test files'))
                return
            
            if not files_in_test:
                download_queue.put((sp_id, sp_name, dir, [], "empty"))
                return

            files_to_download = "\n".join(files_in_test)

            rsync_cmd = [
                "rsync", "-aW", "--mkpath",
                "--ignore-missing-args", "--trust-sender",
                "-e", "ssh -c aes128-gcm@openssh.com", 
                "--files-from=-", 
                f"ECE:{src}", 
                abs_dir
            ]
            subprocess.run(rsync_cmd, input=files_to_download.encode('utf-8'), capture_output=True, check=True)
            
            images = torch.stack(
                thread_map(
                    reader, 
                    [p for n in os.listdir(dir) if is_image(p := os.path.join(dir, n))],
                    disable=True,
                    max_workers=4
                )
            )
            download_queue.put((sp_id, sp_name, dir, images, "success"))
        except subprocess.CalledProcessError as e:
            print(f"Rsync stderr: {e.stderr.decode('utf-8').strip()}")
            download_queue.put((sp_id, sp_name, dir, [], "error"))
        except Exception as e:
            print(f"Unexpected error processing {sp_id}: {e}")
            download_queue.put((sp_id, sp_name, dir, [], "error"))

    # --- 3. Start the Thread Pool (Producers) ---
    NUM_WORKERS = 4  # Adjust based on your network and CPU
    executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)

    # We submit all tasks to the executor. They won't all run at once; 
    # the executor manages them and the download_queue's maxsize will pause them 
    # when the buffer is full, preventing RAM overflow.
    for sp_id in pending_list:
        executor.submit(process_species, sp_id)

    # --- 4. Main Inference Loop (Consumer) ---
    write_header = not os.path.exists(csv_file)

    with open(csv_file, "a") as rf:
        if write_header:
            rf.write("speciesKey,scientificName,correct,incorrect,accuracy\n")
        
        # We know exactly how many items will come through the queue!
        outer_pbar = tqdm(total=len(pending_list), unit="species")
        
        for _ in range(len(pending_list)):
            # Blocks until a worker finishes a download
            item = download_queue.get()
            
            sp_id, sp_name, dir, images, status = item

            if status.startswith("error"):
                print(
                    'Error while processing species, results not saved. Please rerun this block after it finishes.\n',
                    f'\tINFO: {sp_id=}, {sp_name=}, {dir=}, {images=}, {status=}'
                )
                if os.path.exists(dir) and dir != "unknown":
                    try:
                        shutil.rmtree(dir)
                    except Exception:
                        pass
                # 2. Update the progress bar so it finishes cleanly
                outer_pbar.update(1)
                continue

            if status == "empty" or len(images) == 0:
                print(
                    "Warning no images found:\n"
                    f'\tINFO: {sp_id=}, {sp_name=}, {dir=}, {images=}, {status=}'
                )
                rf.write(f"{sp_id},{sp_name},0,0,0.0\n")
                rf.flush()
                if os.path.exists(dir):
                    shutil.rmtree(dir)
                outer_pbar.update(1)
                continue

            correct = []
            batch_size = 32
            
            with tqdm(total=len(images), unit="img", leave=False) as inner_pbar:
                for bi in range(math.ceil(len(images) / batch_size)):
                    bs, be = bi * batch_size, min(len(images), (bi + 1) * batch_size)
                    
                    # IMPORTANT: Move your pre-stacked tensors to the GPU here if not done inside the model
                    pred = model(images[bs:be])

                    correct.append(np.array(pred.labels)[:, 0, 0] == os.path.basename(dir))
                    inner_pbar.update(be - bs)

            correct = np.concatenate(correct)
            acc = float(np.mean(correct).item())
            correct_sum = int(np.sum(correct).item())
            incorrect_sum = int(np.sum(~correct).item())
            
            outer_pbar.set_description(f'{sp_name}[{sp_id}]: {acc:.1%} ({correct_sum}/{len(images)})')
            
            rf.write(f'{sp_id},"{sp_name}",{correct_sum},{incorrect_sum},{acc:.3f}\n')
            rf.flush()

            try:
                shutil.rmtree(dir)
            except Exception:
                pass
            
            outer_pbar.update(1)

    # Clean up the thread pool in the background
    executor.shutdown(wait=False)