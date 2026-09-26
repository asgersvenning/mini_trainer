# Release preparation and publication

Branches select preparation; publishing a GitHub Release selects publication.
A branch push or tag push never publishes packages, weights or a hosted demo.
Ordinary CI covers `master` and every `release/**` branch.

| Product | Preparation branch | GitHub Release tag | Workflow |
| --- | --- | --- | --- |
| Training package | `release/packages/mt-trainer` | `packages/mt-trainer/vVERSION` | `publish.yml` |
| Model package and assets | `release/models/PRODUCT` | `models/PRODUCT/vVERSION` | `publish-model.yml` |
| Demo only | `release/demos/PRODUCT` | No package/model release | `publish-demo.yml` |

Manual dispatch also prepares a selected revision; model/demo dispatch requires
`product`. Only a non-prerelease GitHub Release initiates package/model publication.
After a publisher fix, model dispatch may set `resume_run` to that original release
run: it validates and reuses retained qualified artifacts, skips building/PyPI, and
finishes assets/demo under the existing environment approvals.
Demo-only publication requires manual `publish=true`. Human environment approvals
remain the last gate. Release routing uses the tag, not `target_commitish`: the
latter may be a commit SHA and does not reliably identify a branch.

[The resolver](../release_route.py) checks product and version against the
checked-out `pyproject.toml`. [Model descriptors](../../.github/model-releases.toml)
select each model's project and preparation module. Future models add a descriptor
and their necessary release-specific preparation; the publication workflows do
not need another model-name condition. An explicit tag override requires a matching
`tag_version`, preventing accidental reuse for a maintenance version.

**MAMBO V3 keeps `MAMBO_v3` as its explicit release tag**, preserving prepared
public links. `release/mambo-v3` is its stable source and maintenance branch;
select it for manual preparation with `product=mambo-v3` and tag the qualified
commit there. No branch rename is required. Workflows must also exist on `master`
for manual dispatch discovery, but preparation checks out the selected revision
and publication checks out the release tag. Ongoing `master` development does not
flow into the release automatically. Future model branches can follow the
`release/models/PRODUCT` convention for automatic preparation on push.

Each model module owns `prepare_candidate`, `qualify_candidate`,
`publication_assets`, `publish_assets` and `recover_publication`, using the command interfaces shown in
the workflows. This keeps model-specific input inventories, fixture policy, bundle
layout and immutable-upload handling with the release that knows those contracts.
Demo staging exports `stage_space(output)` and its pinned requirements. Shared
workflows orchestrate these steps and retain their exact qualified artifacts.

Training uses `pypi-training`. Model environments are scoped per product:
`pypi-model-PRODUCT`, `model-assets-PRODUCT` and `model-demo-PRODUCT`. Configure
model/Space destination variables there and register trusted publishers on the
corresponding services; no stored publishing tokens are required. Both demo paths
share a per-product deployment concurrency group. A future model needs its own
accounts/environments; it cannot inherit another model's destinations implicitly.

Use [the MAMBO V3 handoff](mambo_v3/publication.md) for concrete accounts,
artifact review, publication order, endpoint checks and safe retries.
