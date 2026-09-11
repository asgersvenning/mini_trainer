# Production release storage and deployment integration

## Existing convention

The `MAMBO_v2` tag's `mini_trainer/deploy.py` downloaded named checkpoint files
from the public `/Models` share, below `MAMBO/`. Aliases such as `europe`,
`north_europe`, and `full` selected different files. A local weights path bypassed
the download. Retain that useful separation between a model identifier, an
immutable download, and an explicit user override.

Do not replace the historical MAMBO files: this release uses a different backbone
and has different class coverage. Its final checkpoint identity is
`174b9214bfea2df69e4f5c5d16afd841fec961db4274f3e6bf474cef9cab5e8a`.

## This publication

Publish the extracted `global-lepi-production-release-20260911T150236Z` directory
under `/Models`, through SFTP. It contains 1,224 original files and 25,455,849,324
bytes. The uploaded ZIP's checksum and all original internal file checksums passed.
Remote inventory sizes matched for all files; read-back SHA-256 checks also matched
for the final PyTorch weights, FP32 ONNX external weights and PTQ ONNX graph.
The remote size inventory is not a claim of full remote checksum readback.

Original `SHA256SUMS` covers the original release. The added `viewer/` distribution
has a separate `SHA256SUMS` and browser-bundle artifact hashes. Keep these scopes
explicit instead of editing the original research-release manifest after the fact.
Local image parity fixtures are excluded from publication.

The viewer uses a new FP32 export with prediction and embedding outputs. The main
release's original FP32 and PTQ exports remain untouched. A successful PTQ export
or runtime check does not establish PTQ accuracy or a deployment speedup.

## Browser hosting

Serve the viewer and its model/runtime bundle from the same public origin. The
existing share serves HTML and `.js` correctly, but did not provide MIME types for
`.mjs` or `.wasm` in the checked responses. The runtime module is copied unchanged
with a `.js` extension and selected through ONNX Runtime's explicit `wasmPaths`
override. Actual Chromium inference through the public share was verified. Avoid
assuming a separate website can fetch these files: the checked responses did not
advertise cross-origin permission.

A standalone site on a different origin needs an explicit CORS and worker-loading
strategy, or a same-origin static mirror of the bundle. No inference backend or
credential-bearing proxy is required for the current distribution. Model loading
is explicit; images are processed client-side. Keep optional GBIF photo/name
services separate from this guarantee.

## Rollout increments

1. **Freeze and catalogue.** Record final URLs, hashes, class order, preprocessing,
   output semantics, checkpoint identity, runtime requirements and evidence. Expose
   models and documentation as individual downloadable files. Keep release directories
   immutable once published; corrections become a new version.
2. **Add a small model registry.** Map human-facing model names to versioned manifests,
   not mutable raw checkpoint filenames. Preserve explicit file/URL overrides. Validate
   downloaded hashes before promoting a temporary download into a local cache. Keep
   model source/version visible in predictions and saved outputs.
3. **Qualify the intended deployment.** Measure real-image parity and throughput on the
   target runtime, confirm preprocessing and hierarchy semantics, and evaluate any PTQ
   or geographical class filtering separately. Use the held-out in-domain and expert
   evaluation protocols; choose thresholds on separate calibration data. Do not infer
   production acceptance from the browser smoke test.
4. **Offer a reviewed default.** Introduce a release alias only after acceptance;
   preserve rollback to the previous immutable manifest. The qualitative explorer can
   link to that same model identity without being the authoritative evaluation tool.
5. **Extend selectively.** Add additional browser/providers or dataset inputs based on
   measured benefit. Folder/ZIP inference, offline install, optional cached taxon names,
   and cross-origin bundle loading are separate bounded increments. Recomputing global
   t-SNE in the browser is unnecessary for inserting an image into this fixed map.

## Evidence limits

The first browser implementation was exercised with the final production checkpoint
in Chromium, using CPU WASM, one real-image fixture and synthetic preprocessing
patterns. It is not a Safari/Firefox/WebGPU qualification or a broad accuracy study.
Fixed-map t-SNE insertion is approximate; its measured neighbour retention accompanies
the image. Preserve both the numerical runtime comparison and the end-to-end image
comparison, since matching preprocessed tensors alone does not qualify image decoding.
