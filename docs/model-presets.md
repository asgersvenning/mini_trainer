# Model preset scope

Generated release catalogue: edit `dev/releases/mambo_v3/preset-definitions.toml`, then use the build command below.

These overlapping deployment presets aim to avoid most geographically nonsensical predictions while allowing species that **can be found** in a region. They do not describe native or natural distributions, or where species should occur. Recorded introduced species, migrants and vagrants are eligible: no native-status or establishment filter is applied. Exclusion is not evidence that a species cannot occur there. Country codes are ISO alpha-2 metadata values.

Each geographic preset applies the minimum row count shown below. Counts use all existing splits, including held-out rows, without further deduplication. Each row counts once even if it matches both a country and a continent predicate. Full uses all model species without a regional threshold. Lists retain the model's species order.

Europe and northern Europe preserve MAMBO_v2 membership. Parenthesized countries in northern Europe's scope have ambiguous historical inclusion and do not change its membership. The other presets are new release definitions. These are release assets; adapter/API discovery integration and preset-specific inference qualification are still pending.

## Presets

| ID | Species | Minimum rows per species | Selected rows | Geographic scope |
| --- | ---: | ---: | ---: | --- |
| `full` | 12,632 | — | — | All species in the pinned model. |
| `europe` | 3,014 | 26 | 2,079,617 | Records assigned EUROPE by the metadata, including European-labelled portions of transcontinental countries. |
| `north_europe` | 1,977 | 26 | 768,497 | Germany, Denmark, Estonia, Finland, Lithuania, Latvia, Netherlands, Norway, Poland, Sweden (Ireland, Iceland, Åland, Faroe Islands, Guernsey, Isle of Man, Jersey, Svalbard/Jan Mayen: ambiguous historical inclusion; adding any or all leaves the species list unchanged). |
| `australia` | 1,907 | 1 | 465,726 | All Australian records, including Tasmania and other territories recorded under AU; not all Oceania. |
| `tasmania` | 401 | 1 | 4,457 | Australian records explicitly assigned stateProvince Tasmania. Species recorded there, not only endemic species; blank/other state values are excluded. |
| `north_america` | 4,551 | 1 | 2,300,391 | Canada, United States, Mexico, Greenland, Bermuda, Saint Pierre and Miquelon. Whole countries, including US records outside the continental mainland. |
| `central_america` | 2,022 | 1 | 210,173 | Mexico, Belize, Guatemala, Honduras, El Salvador, Nicaragua, Costa Rica, Panama. Mexico deliberately overlaps North America. |
| `south_america` | 1,683 | 1 | 257,026 | All SOUTH_AMERICA records or records from Argentina, Bolivia, Brazil, Chile, Colombia, Ecuador, Falklands, French Guiana, Guyana, Paraguay, Peru, Suriname, Uruguay, Venezuela, Costa Rica or Panama. Costa Rica and Panama deliberately overlap Central America. |
| `caribbean` | 1,092 | 1 | 28,652 | Caribbean islands and territories, Bahamas, Bermuda, Belize and the Guianas (Guyana, Suriname, French Guiana). Does not include every mainland country with a Caribbean coast. |
| `south_asia` | 1,929 | 1 | 177,926 | Afghanistan, Bangladesh, Bhutan, India, Maldives, Nepal, Pakistan, Sri Lanka, Myanmar and Iran; deliberately broad western/eastern overlap. |
| `asia` | 5,336 | 1 | 1,033,931 | All ASIA records plus all records from the listed Asian countries and territories, including Russia, Turkey, Georgia, Armenia, Azerbaijan and Cyprus in full. Includes their European-labelled records and records with blank continent. |
| `japan` | 974 | 1 | 30,076 | All records assigned countryCode JP, including islands. |
| `africa` | 1,129 | 1 | 149,267 | All AFRICA records plus the listed African countries and island territories. AFRICA-labelled records from transcontinental/overseas countries remain included. |
| `subsaharan_africa` | 904 | 1 | 144,918 | Broad African selection excluding Algeria, Egypt, Libya, Morocco, Tunisia and Western Sahara. Includes Sudan, Mauritania, Mali, Niger, Chad, the Horn, Madagascar and island territories; this is not a Sahara boundary polygon. |
| `mediterranean` | 2,874 | 1 | 660,065 | Whole Mediterranean coastal countries/territories plus Portugal, Andorra, San Marino, Vatican City, North Macedonia, Bulgaria, Serbia and Jordan. Includes inland and overseas records of selected countries, not only Mediterranean climate zones. |
| `arctic` | 6,096 | 1 | 2,496,358 | Canada, United States, Greenland, Iceland, Faroe Islands, Norway, Svalbard/Jan Mayen, Sweden, Finland, Åland and Russia. A deliberately expansive country proxy: includes southern records and is NOT an Arctic Circle or tundra filter. |
| `oceania` | 2,337 | 1 | 584,477 | All OCEANIA records plus Australia, New Zealand, Papua New Guinea and the listed Pacific countries/territories. Australia and Tasmania intentionally overlap. |
| `southeast_asia` | 2,031 | 1 | 172,424 | Brunei, Cambodia, Indonesia, Laos, Malaysia, Myanmar, Philippines, Singapore, Thailand, Timor-Leste, Vietnam and Papua New Guinea; whole-island-region overlap with Oceania is intentional. |
| `east_asia` | 4,447 | 1 | 659,767 | China, Hong Kong, Macao, Taiwan, Japan, North Korea, South Korea, Mongolia and Russia. All Russia is included because this preset uses whole-country filters. |
| `middle_east` | 1,330 | 1 | 24,805 | Turkey, Cyprus, Syria, Lebanon, Israel, Palestine, Jordan, Iraq, Iran, Kuwait, Saudi Arabia, Bahrain, Qatar, UAE, Oman, Yemen, Egypt, Armenia, Azerbaijan, Georgia, Afghanistan and Pakistan. Deliberate overlap with Mediterranean, Africa and South Asia. |

## Exact metadata filters

- **europe** (Europe (legacy)): `continent` in `EUROPE`.
- **north_europe** (Northern Europe (legacy)): `countryCode` in `DE, DK, EE, FI, LT, LV, NL, NO, PL, SE`.
- **australia** (Australia including Tasmania): `countryCode` in `AU`.
- **tasmania** (Tasmania only): (`countryCode` in `AU`) AND `stateProvince` in `Tasmania`.
- **north_america** (North America): `countryCode` in `CA, US, MX, GL, BM, PM`.
- **central_america** (Central America): `countryCode` in `MX, BZ, GT, HN, SV, NI, CR, PA`.
- **south_america** (South America): `continent` in `SOUTH_AMERICA` OR `countryCode` in `AR, BO, BR, CL, CO, EC, FK, GF, GY, PY, PE, SR, UY, VE, CR, PA`.
- **caribbean** (Caribbean): `countryCode` in `AG, AI, AW, BB, BL, BQ, BS, CU, CW, DM, DO, GD, GP, HT, JM, KN, KY, LC, MF, MQ, MS, PR, SX, TC, TT, VC, VG, VI, BM, BZ, GY, SR, GF`.
- **south_asia** (South Asia): `countryCode` in `AF, BD, BT, IN, MV, NP, PK, LK, MM, IR`.
- **asia** (Asia): `continent` in `ASIA` OR `countryCode` in `AF, AM, AZ, BH, BD, BT, BN, KH, CN, CY, GE, HK, IN, ID, IR, IQ, IL, JP, JO, KZ, KP, KR, KW, KG, LA, LB, MO, MY, MV, MN, MM, NP, OM, PK, PS, PH, QA, RU, SA, SG, LK, SY, TW, TJ, TH, TL, TR, TM, AE, UZ, VN, YE`.
- **japan** (Japan): `countryCode` in `JP`.
- **africa** (Africa): `continent` in `AFRICA` OR `countryCode` in `DZ, AO, BJ, BW, BF, BI, CV, CM, CF, TD, KM, CG, CD, CI, DJ, EG, GQ, ER, SZ, ET, GA, GM, GH, GN, GW, KE, LS, LR, LY, MG, MW, ML, MR, MU, YT, MA, MZ, NA, NE, NG, RE, RW, SH, ST, SN, SC, SL, SO, ZA, SS, SD, TZ, TG, TN, UG, EH, ZM, ZW`.
- **subsaharan_africa** (Sub-Saharan Africa (broad)): (`continent` in `AFRICA` OR `countryCode` in `AO, BJ, BW, BF, BI, CV, CM, CF, TD, KM, CG, CD, CI, DJ, GQ, ER, SZ, ET, GA, GM, GH, GN, GW, KE, LS, LR, MG, MW, ML, MR, MU, YT, MZ, NA, NE, NG, RE, RW, SH, ST, SN, SC, SL, SO, ZA, SS, SD, TZ, TG, UG, ZM, ZW`) AND country NOT in `DZ, EG, LY, MA, TN, EH`.
- **mediterranean** (Mediterranean (broad)): `countryCode` in `AL, DZ, BA, HR, CY, EG, FR, GR, IL, IT, LB, LY, MT, MC, ME, MA, PS, SI, ES, SY, TN, TR, PT, GI, AD, SM, VA, MK, BG, RS, JO`.
- **arctic** (Arctic / broad northern-country scope): `countryCode` in `CA, US, GL, IS, FO, NO, SJ, SE, FI, AX, RU`.
- **oceania** (Oceania): `continent` in `OCEANIA` OR `countryCode` in `AU, NZ, PG, FJ, SB, VU, NC, PF, WS, AS, TO, TV, KI, NR, FM, MH, PW, GU, MP, CK, NU, TK, WF, PN, NF`.
- **southeast_asia** (Southeast Asia): `countryCode` in `BN, KH, ID, LA, MY, MM, PH, SG, TH, TL, VN, PG`.
- **east_asia** (East Asia): `countryCode` in `CN, HK, MO, TW, JP, KP, KR, MN, RU`.
- **middle_east** (Middle East): `countryCode` in `TR, CY, SY, LB, IL, PS, JO, IQ, IR, KW, SA, BH, QA, AE, OM, YE, EG, AM, AZ, GE, AF, PK`.

## Interpretation and reproducibility

Mexico belongs to North and Central America; Costa Rica and Panama belong to Central and South America. Australia includes Tasmania; Tasmania-only uses the explicit state field and does not mean endemic-only. Arctic is a broad northern-country proxy and includes southern records from those countries. Regional restrictions change score normalization; excluded truth labels must remain visible in evaluation.

Blank geographic fields match no predicate unless another selected field matches. The Tasmania preset excludes Australian records with blank or different state values. Overlapping presets are expected; membership in one does not exclude another.

Run from the repository root with the existing PyArrow environment and the previously downloaded model manifest:

```sh
.venv/bin/python -m dev.releases.mambo_v3.build_presets \
  examples/global_lepi/0032836-250426092105405_processing_metadata_postprocessed_quality_filtered.parquet \
  --evidence-root local-evidence/mambo-v3
```

The default checks committed assets, hashes, counts and this catalogue against fresh reconstruction. Use `--write` after an intentional definition update to regenerate them. Legacy preset hashes must remain unchanged. Published preset membership changes require a new release/revision and an added/removed-ID report.

[Machine-readable definitions](../dev/releases/mambo_v3/preset-definitions.toml), [generated hashes and counts](../dev/releases/mambo_v3/preset-manifest.toml), [source provenance](../dev/releases/mambo_v3/construction.toml), [legacy reconstruction details](../dev/releases/mambo_v3/README.md#regional-scope-and-construction).
