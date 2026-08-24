# Ensembl mapping report

axis: `artifacts/gene_axis.parquet` (18533 symbols)
gencode: `gencode.v32.annotation.gtf.gz` (59299 unambiguous symbols, 69 ambiguous, 45 PAR_Y skipped)
hgnc: `hgnc_complete_set.txt` (42360 approved, 14621 prev, 41132 alias)

**mapped 18533/18533 (100.00%)**

| source | n |
|---|---|
| `gencode` | 18517 |
| `reference_var` | 15 |
| `gencode_hgnc_tiebreak` | 1 |

## GENCODE/HGNC disagreements on an approved symbol: 12

- `PRRC2B`: gencode `ENSG00000130723` vs hgnc `ENSG00000288701`
- `ACTL10`: gencode `ENSG00000182584` vs hgnc `ENSG00000288649`
- `NPBWR1`: gencode `ENSG00000183729` vs hgnc `ENSG00000288611`
- `LRTOMT`: gencode `ENSG00000184154` vs hgnc `ENSG00000284922`
- `CDR1`: gencode `ENSG00000184258` vs hgnc `ENSG00000288642`
- `HOMEZ`: gencode `ENSG00000215271` vs hgnc `ENSG00000290292`
- `UGT1A5`: gencode `ENSG00000240224` vs hgnc `ENSG00000288705`
- `UGT1A3`: gencode `ENSG00000243135` vs hgnc `ENSG00000288702`
- `PANO1`: gencode `ENSG00000274897` vs hgnc `ENSG00000288675`
- `CCL3L1`: gencode `ENSG00000276085` vs hgnc `ENSG00000277768`
- `F8A1`: gencode `ENSG00000277203` vs hgnc `ENSG00000288722`
- `ARHGAP11B`: gencode `ENSG00000284906` vs hgnc `ENSG00000285077`

## Symbols ambiguous in the source annotation, disambiguated: 16

- `ABCF2` -> `ENSG00000033050` (reference_var); HGNC said `ENSG00000033050`
    - `ENSG00000285292` v1 level=2 protein_coding hgnc_id=HGNC:71
    - `ENSG00000033050` v9 level=2 protein_coding hgnc_id=HGNC:71
- `AHRR` -> `ENSG00000063438` (reference_var); HGNC said `ENSG00000063438`
    - `ENSG00000286169` v1 level=2 protein_coding hgnc_id=HGNC:346
    - `ENSG00000063438` v17 level=1 protein_coding hgnc_id=HGNC:346
- `SOD2` -> `ENSG00000112096` (reference_var); HGNC said `ENSG00000291237`
    - `ENSG00000112096` v18 level=1 protein_coding hgnc_id=HGNC:11180
    - `ENSG00000285441` v1 level=2 protein_coding hgnc_id=None
- `CYB561D2` -> `ENSG00000114395` (reference_var); HGNC said `ENSG00000114395`
    - `ENSG00000114395` v10 level=1 protein_coding hgnc_id=HGNC:30253
    - `ENSG00000271858` v5 level=2 lncRNA hgnc_id=None
- `PDE11A` -> `ENSG00000128655` (reference_var); HGNC said `ENSG00000128655`
    - `ENSG00000128655` v18 level=2 protein_coding hgnc_id=HGNC:8773
    - `ENSG00000284741` v1 level=2 protein_coding hgnc_id=None
- `TMSB15B` -> `ENSG00000158427` (reference_var); HGNC said `ENSG00000158427`
    - `ENSG00000158427` v15 level=2 protein_coding hgnc_id=HGNC:28612
    - `ENSG00000269226` v7 level=2 protein_coding hgnc_id=None
- `ATXN7` -> `ENSG00000163635` (reference_var); HGNC said `ENSG00000163635`
    - `ENSG00000285258` v1 level=2 protein_coding hgnc_id=HGNC:10560
    - `ENSG00000163635` v18 level=2 protein_coding hgnc_id=HGNC:10560
- `HSPA14` -> `ENSG00000284024` (reference_var); HGNC said `ENSG00000187522`
    - `ENSG00000284024` v2 level=2 protein_coding hgnc_id=None
    - `ENSG00000187522` v16 level=2 protein_coding hgnc_id=HGNC:29526
- `GOLGA8M` -> `ENSG00000188626` (reference_var); HGNC said `ENSG00000188626`
    - `ENSG00000188626` v6 level=2 protein_coding hgnc_id=HGNC:44404
    - `ENSG00000261480` v1 level=2 lncRNA hgnc_id=HGNC:44404
- `SFTA3` -> `ENSG00000229415` (reference_var); HGNC said `ENSG00000229415`
    - `ENSG00000257520` v1 level=2 lncRNA hgnc_id=None
    - `ENSG00000229415` v9 level=1 protein_coding hgnc_id=HGNC:18387
- `PINX1` -> `ENSG00000254093` (reference_var); HGNC said `ENSG00000254093`
    - `ENSG00000258724` v1 level=2 protein_coding hgnc_id=HGNC:30046
    - `ENSG00000254093` v9 level=2 protein_coding hgnc_id=HGNC:30046
- `TBCE` -> `ENSG00000285053` (reference_var); HGNC said `ENSG00000284770`
    - `ENSG00000285053` v1 level=2 protein_coding hgnc_id=HGNC:11582
    - `ENSG00000284770` v2 level=2 protein_coding hgnc_id=HGNC:11582
- `CCDC39` -> `ENSG00000284862` (reference_var); HGNC said `ENSG00000284862`
    - `ENSG00000284862` v3 level=2 protein_coding hgnc_id=HGNC:25244
    - `ENSG00000145075` v13 level=2 lncRNA hgnc_id=HGNC:25244
- `POLR2J3` -> `ENSG00000168255` (gencode_hgnc_tiebreak); HGNC said `ENSG00000168255`
    - `ENSG00000168255` v20 level=2 protein_coding hgnc_id=HGNC:33853
    - `ENSG00000285437` v1 level=2 protein_coding hgnc_id=HGNC:33853
- `ZNF883` -> `ENSG00000285447` (reference_var); HGNC said `ENSG00000228623`
    - `ENSG00000228623` v6 level=2 transcribed_unprocessed_pseudogene hgnc_id=HGNC:27271
    - `ENSG00000285447` v1 level=3 protein_coding hgnc_id=None
- `GGT1` -> `ENSG00000286070` (reference_var); HGNC said `ENSG00000100031`
    - `ENSG00000286070` v1 level=2 protein_coding hgnc_id=HGNC:4250
    - `ENSG00000100031` v19 level=2 protein_coding hgnc_id=HGNC:4250

## Two symbols sharing one Ensembl ID: 0


## Unmapped: 0

