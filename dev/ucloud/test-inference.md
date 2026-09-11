# Full in-domain test inference

From `/work/mini_trainer`, run in tmux:

```bash
git pull --ff-only
bash dev/ucloud/test-inference.sh /work/test-full-1
```

This selects every `test` row from the production `data_index.json`, preserves
labels, class indices and ordering, and copies encoded files into `/dev/shm`
with 512 I/O threads and bounded pending work. It does not select training or
validation rows, resize images, or resolve taxonomy. The 128 GiB staging cap
and free-space check apply before copying. Size inspection precedes copy progress.

The normal `mt_hpredict` CLI runs on GPU 0 with input, weights and the staged
index supplied; other defaults come from the inference CLI and model metadata.
The existing isolated inference source overlay is used without reinstalling.
Each stage has a one-hour timeout. Stop any old stalled test inference before
starting this run. This helper does not terminate unrelated processes.

```bash
tail -n 5 /work/test-full-1/stage.log
tail -n 5 /work/test-full-1/inference.log
```

The CSV is `/work/test-full-1/predictions/mini_metric.csv`; configuration,
source-to-staged manifest and logs remain in `/work/test-full-1`. RAM files do
not survive job termination. Completed staging can be reused after an inference
failure, without reading source images again:

```bash
bash dev/ucloud/test-inference.sh /work/test-full-2 --reuse-stage /work/test-full-1
```

A completed manifest is required for reuse. An interrupted copy is not resumable.
The selected test count should match the production index (expected 632,913
from the original dataset; the saved index is authoritative).
