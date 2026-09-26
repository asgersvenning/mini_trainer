# Publish MAMBO V3

Preparation does not publish anything. The owner performs the steps below after
reviewing the candidate and its qualification records. Packages and weights have
separate identities: training **`minitrainer==0.3.0`** (Python `mini_trainer`),
deployment **`mambo-v3==0.3.0`** (Python `mambo_deploy`), model **`MAMBO_v3`**.
`mini-trainer` on PyPI is an unrelated project; never publish or install it here.

## 1. Configure accounts and environments

Claim/create the PyPI projects `minitrainer` and `mambo-v3`. The names had no public
project during preparation, but this does not reserve them. Create pending trusted
publishers with owner `asgersvenning`, repository `mini_trainer` and these settings:

| Project | Workflow filename | GitHub environment |
| --- | --- | --- |
| `minitrainer` | `publish.yml` | `pypi-training` |
| `mambo-v3` | `publish-model.yml` | `pypi-model-mambo-v3` |

Create those GitHub environments plus `model-assets-mambo-v3` and `model-demo-mambo-v3`, with owner
review before public writes. No PyPI API token is needed. Create the Hugging Face
model repository and Gradio Space, both named `asgersvenning/MAMBO-v3` in their
respective namespaces. Choose CPU Basic initially; the demo does not need a GPU.

- `model-assets-mambo-v3`: variable `HF_MODEL_REPO=asgersvenning/MAMBO-v3`; secret `HF_TOKEN`
  with write access limited to that model repository.
- `model-demo-mambo-v3`: variable `HF_SPACE_REPO=asgersvenning/MAMBO-v3`; secret `HF_TOKEN`
  with write access limited to that Space.

These are the destinations linked in the public documentation. If using another
namespace, update those public links before final preparation as well as the
variables. Do not put tokens in source, release assets or CLI arguments.

## 2. Review and prepare without publishing

[Shared routing conventions](../README.md) use `release/packages/**`,
`release/models/**` and `release/demos/**` for automatic preparation. The existing
`release/mambo-v3` branch can stay in place and use manual preparation; it has no
special publication permission. `MAMBO_v3` is an explicit descriptor tag override.

Make the reviewed workflow changes available on the repository's default branch
before using manual Actions dispatch. Push the release source through the normal
review/merge process; no release tag is needed for preparation.

Run **Prepare and publish model** (`publish-model.yml`) manually on the intended
release revision with `product=mambo-v3`. Manual dispatch only downloads pinned inputs, builds artifacts,
qualifies installed CPU runtimes and uploads downloadable Action artifacts;
it does not run any publication job. Download and review:

- `model-candidate`: exact wheels, source distribution, offline bundle, evidence,
  source/hash manifest and `qualification/` results.
- `model-publication`: the explicit GitHub, model and Space upload directories.

The CLI equivalent, from a clean committed checkout with NumPy/Pillow and uv:

```sh
python -m dev.releases.mambo_v3.prepare_candidate \
  --source local-evidence/mambo-v3 --output local-evidence/mambo-v3-publication-candidate --download
python -m dev.releases.mambo_v3.qualify_candidate \
  local-evidence/mambo-v3-publication-candidate
python -m dev.releases.mambo_v3.publication_assets \
  local-evidence/mambo-v3-publication-candidate --output local-evidence/mambo-v3-publication-assets
```

Use a new output directory after a failed attempt. Qualification creates a separate
CPU environment; it never synchronizes the working `.venv`. CI uses synthetic
images for runtime contracts, not accuracy. Local qualification can use
`--dataset /path/to/flemming` for four retained real images. Existing model quality
and timing evidence is reused, with its original environment and methodology.

Review CC BY-NC-SA 4.0 weight terms, MIT code terms, attribution and the disclosed
missing original training revision/initial-checkpoint hash. Review the rendered
README figures, demo controls and public-file inventories. No photographs, private
prediction archives, credentials or datasets belong in the upload directories.

## 3. Publish training, then the model

At the reviewed commit, create and **publish a GitHub Release** with tag
`packages/minitrainer/v0.3.0`. A tag push alone does not publish. Approve `pypi-training`:
`publish.yml` builds, installs and exercises the wheel, then publishes the exact
retained wheel and the source distribution to PyPI. Model/Space jobs do not run.
Verify `minitrainer==0.3.0` is publicly available under the intended ownership.

Then publish the GitHub Release tagged **`MAMBO_v3`** at the reviewed model commit.
The model workflow prepares and qualifies its candidate before the public gates:

1. `pypi-model-mambo-v3` publishes only `mambo-v3` distributions. It first checks that the
   intended `minitrainer` version is public and points to this repository.
2. `model-assets-mambo-v3` attaches the offline bundle, deployment distributions, evidence,
   inventory and checksums to GitHub; it uploads the model repository and creates
   an immutable Hugging Face `v0.3.0` tag at that upload's commit.
3. `model-demo-mambo-v3` deploys the staged Space after package/model publication succeeds.

Prereleases prepare artifacts but do not publish packages or activate the demo.
Training versions and model versions need not advance together. Future trained
models use separate model-generation packages; V3 maintenance retains its weights
and existing preset identities. Introduce a distinct reviewed release identity
before publishing a future model or maintenance version; never reuse `MAMBO_v3`
or an existing package version for different bytes.

## 4. Verify public integration

From a fresh environment outside the checkout:

```sh
uv venv --python 3.13 .venv-mambo
source .venv-mambo/bin/activate
uv pip install 'mambo-v3[onnx]==0.3.0'
mambo_predict -i moth.jpg -o output --name onnx
uvx --from 'mambo-v3[onnx]==0.3.0' mambo_predict -i moth.jpg -o output --name isolated
```

Also follow the documented native installation, check cache/offline reuse and
compare downloaded assets against GitHub's `SHA256SUMS`. Open the model page and
Space; try both runtimes, a regional preset, a custom list and TTA. Confirm figures,
citation and cross-links resolve. These are post-publication checks, not claims
that public endpoints were exercised during local preparation.

## Demo updates and recovery

For a UI-only update, dispatch **Prepare or update model demo** (`publish-demo.yml`)
on a reviewed revision with `product=mambo-v3`. The default `publish=false` builds/checks a staged Space
without public writes. Set `publish=true` and approve `model-demo-mambo-v3` to deploy it.
This workflow uploads no package or model weights. It requires the public package
release to exist and pins that package and its CPU runtime dependencies.

For partial publication, rerun the failed jobs from the same workflow run so they
reuse the prepared artifacts. PyPI skips identical existing distributions; GitHub
compares existing asset bytes and refuses replacements; Hugging Face compares the
recorded immutable revision and refuses a different payload. Do not rebuild and
silently replace already-published files. A different release payload needs a new
reviewed version/identity. If Space upload succeeded but tag creation failed,
retrying records the same payload and revision.

Retain the reviewed Action artifacts externally before their 30-day expiry. To
withdraw a defective release, stop recommending it and publish a corrective
version; preserve immutable assets for existing pins. Revert an application to its
retained V2 environment if needed. V3 embeddings and thresholds do not transfer to
V2. Remote account configuration, publication and final live endpoint checks remain
owner tasks; preparation never performs those writes.
