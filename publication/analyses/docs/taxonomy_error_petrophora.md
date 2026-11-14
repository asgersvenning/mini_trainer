# Geometrids labelled as plume moths in the global-lepi dataset

Verified upstream 2026-10-21. Logged in `data/known_taxonomy_errors.csv` (`known_taxonomy_errors()`
in `helpers/embeddings/heads.R`).

## The issue

Five *Petrophora* species (Geometridae) carry the genus *Pterophorus* (8331787) and the family
Pterophoridae (8863) in the `gefion-global-lepi` taxonomy and labels. In the GBIF backbone the species
records have status DOUBTFUL and the issue `NAME_PARENT_MISMATCH`.

| Species | GBIF key | Train / val / test images |
|---|---|---|
| *Petrophora chlorosata* | [1973467](https://www.gbif.org/species/1973467) | 1,559 / 194 / 196 |
| *Petrophora convergata* | [4525132](https://www.gbif.org/species/4525132) | 507 / 65 / 62 |
| *Petrophora divisata* | [1973458](https://www.gbif.org/species/1973458) | 50 / 7 / 6 |
| *Petrophora narbonea* | [12046692](https://www.gbif.org/species/12046692) | 435 / 55 / 54 |
| *Petrophora subaequaria* | [1973466](https://www.gbif.org/species/1973466) | 713 / 89 / 88 |

Image counts are those of the source dataset, not of every ablation subset.

Old and new taxonomy in one record: the
[occurrence API response for 6170897041](https://api.gbif.org/v1/occurrence/6170897041) (a *P.
chlorosata* observation) holds two classifications under `classifications`. The Catalogue of Life
(`7ddf754f-d193-4cc9-b351-99906754a03b`, ACCEPTED) puts it in Geometridae > Petrophora. The legacy
GBIF backbone (`d7dddbf4-2cf0-4f39-9b2a-bb099caae36c`, PROVISIONALLY_ACCEPTED) puts it in
Pterophoridae > Pterophorus, and its keys (8863, 8331787, 1973467) are the ones in the top-level
`family`, `genus` and `speciesKey` fields and in our labels. The genus record of
[*Petrophora* Hübner (1973454)](https://www.gbif.org/species/1973454) is a synonym of *Pterophorus*.

The response, condensed (requested 2026-10-23T18:34:49Z; the record's `lastCrawled` and `lastParsed`
are below):

```json
{
  "key": 6170897041,
  ...
  "lastCrawled": "2026-10-20T14:46:47.524+00:00",
  "lastParsed": "2026-10-21T03:59:58.114+00:00",
  ...
  "classifications": {
    "7ddf754f-d193-4cc9-b351-99906754a03b": {
      "usage": {
        "key": "776HX",
        "name": "Petrophora chlorosata (Scopoli, 1763)",
        "rank": "SPECIES",
        "code": "ZOOLOGICAL",
        "authorship": "(Scopoli, 1763)",
        "genericName": "Petrophora",
        "specificEpithet": "chlorosata",
        "formattedName": "<i>Petrophora</i> <i>chlorosata</i> (Scopoli, 1763)"
      },
      "acceptedUsage": { ... same as usage ... },
      "taxonomicStatus": "ACCEPTED",
      "classification": [
        { "key": "CS5HF", "name": "Eukaryota", "rank": "DOMAIN" },
        { "key": "N", "name": "Animalia", "rank": "KINGDOM" },
        { "key": "RT", "name": "Arthropoda", "rank": "PHYLUM" },
        { "key": "L2655", "name": "Hexapoda", "rank": "SUBPHYLUM" },
        { "key": "H6", "name": "Insecta", "rank": "CLASS" },
        { "key": "B6L67", "name": "Lepidoptera", "rank": "ORDER" },
        { "key": "626YP", "name": "Geometroidea", "rank": "SUPERFAMILY" },
        { "key": "C76TY", "name": "Geometridae", "rank": "FAMILY" },
        { "key": "C6ZT3", "name": "Ennominae", "rank": "SUBFAMILY" },
        { "key": "BMLPY", "name": "Petrophora", "rank": "GENUS" },
        { "key": "776HX", "name": "Petrophora chlorosata", "rank": "SPECIES" }
      ],
      "issues": ["TAXON_ID_NOT_FOUND"]
    },
    "d7dddbf4-2cf0-4f39-9b2a-bb099caae36c": {
      "usage": {
        "key": "1973467",
        "name": "Petrophora chlorosata (Scopoli, 1763)",
        "rank": "SPECIES",
        "code": "ZOOLOGICAL",
        "authorship": "(Scopoli, 1763)",
        "genericName": "Petrophora",
        "specificEpithet": "chlorosata",
        "formattedName": "<i>Petrophora</i> <i>chlorosata</i> (Scopoli, 1763)"
      },
      "acceptedUsage": { ... same as usage ... },
      "taxonomicStatus": "PROVISIONALLY_ACCEPTED",
      "classification": [
        { "key": "1", "name": "Animalia", "rank": "KINGDOM" },
        { "key": "54", "name": "Arthropoda", "rank": "PHYLUM" },
        { "key": "216", "name": "Insecta", "rank": "CLASS" },
        { "key": "797", "name": "Lepidoptera", "rank": "ORDER" },
        { "key": "8863", "name": "Pterophoridae", "rank": "FAMILY" },
        { "key": "8331787", "name": "Pterophorus", "rank": "GENUS" },
        { "key": "1973467", "name": "Petrophora chlorosata", "rank": "SPECIES" }
      ],
      "issues": ["TAXON_ID_NOT_FOUND"]
    }
  },
  ...
}
```

## Why it affected us

GBIF's backbone resolves the name "Petrophora" to the plume-moth synonym, and the five species records
inherited that parent. Our download (0032836-250426092105405, 15 May 2025) took the classification as
GBIF gave it, and the labels propagated unchanged through the source Parquet, the hierarchy built by
mini_trainer and the exported vocabulary taxonomy.

## What we did

The labels are used as they are everywhere (the models were trained with them), and the species are
marked in the figures: they are the red "Outlier" points of the embedding-structure figures
(`embedding_structure.R`) and the targets of the outlyingness figure (`outlyingness.R`).
*Pterophorus* has 10 species in the labels, five of them geometrids, which makes it the outlying genus
of the Lepidoptera embedding figures.
