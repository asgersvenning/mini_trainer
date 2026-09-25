# MAMBO V3 release handoff

Updated 25 September 2026. **Preparation is complete; publication is separate.**
This releases the model trained on UCloud on 10–11 September, without retraining
or quantization. The [freeze record](../dev/releases/mambo_v3/deployment-freeze.md)
owns the release contract and completion map; [final qualification](../dev/releases/mambo_v3/final-qualification.md)
owns candidate hashes, installed checks and remaining provenance limitations.

## Status and authoritative documents

| Concern | Maintained record |
| --- | --- |
| Installation, API/CLI, inputs/outputs, defaults and V2 migration | [Deployment README](../deployment/README.md) and [integration reference](mambo-integration.md) |
| Weight identity, original V2 contract and preset reconstruction | [Input audit](../dev/releases/mambo_v3/README.md), inventory and model provenance alongside it |
| Geographic scope and membership rules | [Preset catalogue](model-presets.md); 25 versioned lists, legacy and updated memberships preserved |
| Flemming and complementary in-domain quality | [Flemming evidence](mambo-deployment-evidence.md), [in-domain evidence](mambo-indomain-evidence.md) |
| Laptop and production-like speed | README figures and [HPC evidence](mambo-hpc-evidence.md), with historical/current measurement scopes separate |
| Runtime and artifact readiness | [Installed qualification](../dev/releases/mambo_v3/final-qualification.md) |
| Future model comparability | [Evidence policy](../dev/releases/mambo_v3/evidence-policy.md) |
| Human publication, integrity and rollback | [Publication handoff](../dev/releases/mambo_v3/publication.md) |

Distribution is `mambo-v3` version `0.3.0`, import `mambo_deploy`. Maintenance
releases retain this model; a future generation uses a separate package. The
standalone API/CLI defaults to global scope and supports PyTorch/standard ONNX,
custom or regional lists, optional embeddings and optional default TTA. Weight
license is CC BY-NC-SA 4.0; adapter code is MIT. No package, successor model asset,
tag or public pointer was published by preparation.

Remaining actions are the publisher's review and explicit publication steps,
including retrieval checks and rollback readiness. No additional full evaluation,
throughput campaign, quantized artifact or exhaustive hardware matrix is a freeze
prerequisite. Known untested platforms remain disclosed, not silently qualified.

## Branch and integration policy

`release/mambo-v3` was created after the minor version bump on master. Direct work
there is restricted to release adapters/assets, presets, packaging, documentation
and release-specific qualification. Shared-core fixes/refactors must originate on
master or a dedicated feature/fix branch, be reviewed/validated there, then merged
and checked for affected combined behavior. This includes existing browser/export
work. Classify by responsibility, not filename; release pressure does not relax
the boundary. Keep unrelated improvements on their own branches.

## Located production artifacts and existing browser work

The source production distribution was already published before this consumer
release: [global-lepi-production-release-20260911T150236Z](https://anon.erda.au.dk/share_redirect/HE90eyuZCT/global-lepi-production-release-20260911T150236Z/index.html).
The [pinned inventory](../dev/releases/mambo_v3/inventory.toml) owns file sizes,
URLs and hashes, including PyTorch `best.pt`, standard prediction ONNX, external
tensors and the separate prediction-plus-embedding graph. Experimental PTQ stays
in the historical archive and is excluded from this release.

The retrieved mappings agree across V2/V3: 12,632 species, 4,476 genera and 104
families, with unchanged class order and parent maps. The selected checkpoint and
best epoch 30 are verified. Exact training Git revision and original initial-file
hash are unavailable; packaging checkout identity must not be substituted.
See [model provenance](../dev/releases/mambo_v3/MODEL_CARD.md).

Existing browser work is recorded on `feature/prototype-browser-inference` at
`b174426a22e42618424fcb0345610ede4a415d01`:
[production integration](https://github.com/asgersvenning/mini_trainer/blob/b174426a22e42618424fcb0345610ede4a415d01/docs/production-release-integration.md)
and [qualification](https://github.com/asgersvenning/mini_trainer/blob/b174426a22e42618424fcb0345610ede4a415d01/dev/prototype_space/portable-qualification.md).
The [RC2 browser manifest](https://anon.erda.au.dk/share_redirect/HE90eyuZCT/global-lepi-viewer-20260915-rc2/browser-model/manifest.json)
identifies the same final checkpoint and embedding graph. Reuse its checked bundle,
external tensors, preprocessing, WASM worker and optional GBIF enrichment rather
than beginning a second browser exporter.

Existing browser evidence is one real-image fixture, not broad browser/accuracy
qualification: identical-input maximum prediction/embedding errors were ~1.72e-5 /
1.80e-7; end-to-end errors were ~0.00552 / 0.000250 with matching top predictions.
The adapter's EXIF/white-alpha policy differs from the historical core reader, so
the recipe name alone is not a complete input contract. ERDA hosting previously
needed explicit runtime paths and unchanged `.mjs` content served as `.js` because
of MIME/CORS constraints; recheck hosting before new claims. Authoring checksums
do not imply browser validation of every asset fetch.

The [repository roadmap](roadmap.md#deferred-portable-prototype-viewer-completion)
owns remaining viewer integration and human/physical-device acceptance. Static
browser inference is a useful deployment path, not a promise that every browser,
WebGPU provider or offline configuration is qualified by this release.
