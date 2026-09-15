# Portable viewer qualification

Updated 2026-09-11. This increment extends `feature/prototype-browser-inference`.
The numerical report, model bundle, preprocessing and fixed-map insertion algorithm
are retained. UI and GBIF transport changes do not retrain or re-export the model.

## Evidence

- Shared browser client checks cover deduplication, cache hits, concurrency,
  cancellation, timeout, 429/offline/malformed responses, taxon identity, safe media
  URLs, taxonomy membership and attribution. Snapshot names need no network call.
- Static Chromium checks pass for one global namespace, legacy state migration,
  malformed legacy state rejection, prediction thumbnails and credits, generic-mode
  fallback and unchanged map transforms during label refresh.
- The retained browser harness passed 158 assertions across numerical views,
  photo interactions, landscape/portrait/mobile layouts and saved views. Its
  fixtures now mock the shared client instead of the removed Python transport.
- Plain static hosting successfully resolved `1775152` as *Dichonia aprilina*,
  returned six reference photos and loaded an image directly in the browser.
  The API returned `Access-Control-Allow-Origin: *`. No Python taxonomy/photo
  service was running. Local-launcher capability/state probes are still optional.
- Real-image browser inference returned 15 prediction rows and a 1,280-dimensional
  embedding. Fixed-map placement remained `[51.14414261925462, -33.22137451588429]`,
  with the same five of twelve angular neighbours retained. Insertion took about
  44–47 ms in these checks. This is functionality evidence, not broad accuracy or
  browser qualification.
- Sixteen affected Python tests passed in total. The launcher socket test required
  an unsandboxed rerun; the initial failure was a local socket permission error.
  Static lint/format/import contracts passed. Installed-wheel smoke passed; an
  additional archive inspection confirmed inclusion of `gbif.js`.

A small same-machine Chromium sample measured first usable view at 2.91 seconds
for the preceding viewer and 2.96 seconds for this candidate; ten synchronous class
selection updates averaged 6.99 and 6.86 ms respectively. Neither initial load
issued GBIF requests. These single samples are a responsiveness check, not evidence
of a performance improvement or a cross-machine baseline.

## Candidate and remaining review

The implementation is recorded in `1eb3efe` and `c3c4134`; merge `b0bd05f`
includes current local master `268d294` (including the testing-economy guidance).

The candidate includes a partial 100-taxon snapshot with retrieval dates. The
preparation command can resume to fill the remaining vocabulary. Images remain
external resources; missing/blocked images retain useful labels and scores.

Human-reviewable desktop/mobile previews use an actual query image and live GBIF
reference images. Generic-mode and error checks use deterministic fixtures.
The existing public production release is not overwritten. Human layout review
and PR integration are separate gates; passing these checks does not imply either
has happened. Firefox, Safari, WebGPU and complete offline image distribution are
outside this increment.

## Reproduce

Run from the feature checkout with the existing environment and a plain static
server serving an exported report and matching `browser-model/` directory:

```bash
node dev/prototype_space/check_gbif.cjs
CHROMIUM_BIN=/path/to/chrome node dev/prototype_space/check_portable.mjs \
  http://127.0.0.1:8767/explorer.html /tmp/portable-check --live
CHROMIUM_BIN=/path/to/chrome node dev/prototype_space/check_browser.mjs \
  http://127.0.0.1:8767/regression/explorer.html /tmp/explorer-regression
CHROME_PATH=/path/to/chrome node dev/prototype_space/check_inference.mjs \
  http://127.0.0.1:8767/explorer.html /path/to/query.jpg \
  /tmp/inference-check.json /tmp/prediction-preview
```

The general browser regression requires the production and synthetic cases made
by the existing exploration harness. The portable check needs `fixture.jpg` in its
static directory for mocked photo rendering; `--live` additionally tests GBIF.
The inference harness's last argument optionally captures desktop/mobile previews.
These are local development checks, not a requirement to repeat every suite after
subsequent documentation-only edits.
