# MAMBO V3 publication handoff

**Preparation only. No command in this document has published this release.**
The package target is `mambo-v3==0.3.0`; the model tag target is `MAMBO_v3`.
The Python import remains `mambo_deploy`. A future generation gets a separate
package, so upgrading this package cannot select a different trained model.

## Local candidate

The prepared review set is `local-evidence/mambo-v3-release-candidate-final/`.
Its manifest includes qualification records, and its checksum list includes the
manifest itself. See [final qualification](final-qualification.md) for the exact
source commit, wheel hashes and evidence-reuse scope.

From a clean, committed release checkout:

```sh
.venv/bin/python -m dev.releases.mambo_v3.prepare_candidate \
  --source local-evidence/mambo-v3 \
  --output local-evidence/mambo-v3-release-candidate
```

The command creates a relocatable model bundle and archive, standalone deployment
wheel and source distribution, matching training wheel, release README, public
comparison evidence, checksums and source-commit inventory. It never tags or uploads.
The output directory must be new. A failed preparation has no completion manifest;
inspect the failure, then use a fresh destination. The inventory is not a claim that
qualification or owner decisions have passed.

## Before any publication

- Review the selected CC BY-NC-SA 4.0 weight license, MIT code license and upstream
  notices in `NOTICES.md`, `MODEL_CARD.md` and `model-provenance.toml`. Training epoch
  30 is verified; initialization is reconstructed from source. The missing starting
  checkpoint hash and training Git revision remain explicit provenance limitations.
- Qualify these exact installed artifacts, record wheel/bundle hashes and actual
  runtime versions, and verify the README examples and single CLI owner.
- Confirm PyPI ownership/availability of `mambo-v3` and access to publish the matching
  `mini_trainer` dependency. This cannot be assumed from local package construction.
- Review the concrete source commit, release notes/changelog, artifacts, documented
  limitations, model notices and validation record. Do not overwrite old model assets.

## Intended distribution

PyPI hosts the small Python package; GitHub Releases is the discovery/migration
entry point and can attach wheels and the evidence archive. Existing immutable
ERDA URLs provide automatic verified downloads of the standard model weights.
The relocatable bundle archive is an optional offline download; its size/location
should be checked before selecting a GitHub versus ERDA attachment.

After approval, a human publisher can create the `MAMBO_v3` tag at the reviewed
commit, publish the exact built distributions, upload any new offline bundle under
an immutable path, and attach `RELEASE_README.md`, `SHA256SUMS` and the artifact
manifest. Use the organization's normal authenticated publishing process; no tokens
or credentials belong in this repository. Test downloading the published assets
and compare their hashes before announcing the release or updating default pointers.

The prepared README uses links to the planned `MAMBO_v3` tag, so those links become
publicly resolvable only after the reviewed tag exists. Do not silently retarget them
to a moving branch. Review the GitHub-rendered README/figures before announcement.

## Intended consumer commands after publication

```sh
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install 'mambo-v3[onnx]==0.3.0'
mambo_predict -i images --name results
```

Or an isolated CLI: `uvx --from 'mambo-v3[onnx]==0.3.0' mambo_predict -i images`.
Native users can install `mambo-v3[torch]==0.3.0` with an explicitly selected PyTorch
backend and pass `--backend torch --device cuda:0`. Offline users install the
provided wheels/dependencies and supply the extracted bundle with `--bundle`.
No repository checkout or dataset metadata is required for these consumer paths.

## Rollback and maintenance

Retain MAMBO V2 and its documented pinned environment/assets unchanged. If V3 needs
to be withdrawn, stop recommending the affected package version; preserve immutable
assets for existing pins and publish a corrective version rather than replacing
bytes. Applications can revert their environment lock or switch to their retained
V2 environment; no input-image migration is needed. Do not reuse V3 embeddings or
confidence thresholds with V2. Keeping both runtimes in separate environments avoids
CLI/import conflicts and makes rollback a deliberate application decision.
