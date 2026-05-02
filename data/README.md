# Datasets

To comply with repository size limits, raw images are excluded from version control via `.gitignore`. Please run the provided bash scripts to download and format the datasets automatically before training.

**Requirements:** `wget`, `unzip`, and Python 3.x.

---

## 1. PACS Dataset (~500MB)
Downloads the 4 domains (Art Painting, Cartoon, Photo, Sketch) and deterministically applies an 80% train / 20% test split. An empty `unlabel` folder is generated for Semi-Supervised Learning.

```bash
cd data/pacs
bash download_pacs.sh
```
*Note: String classes are mapped to integer IDs: `0`:dog, `1`:elephant, `2`:giraffe, `3`:guitar, `4`:horse, `5`:house, `6`:person.*

---

## 2. DomainNet Dataset (~80GB)
Downloads the official ground-truth zips and text splits for all 6 domains (clipart, infograph, painting, quickdraw, real, sketch) from the official BU servers.

```bash
cd data/domainnet
bash download_domainnet.sh
```

---

## Expected Directory Structure
After running both scripts, your data directory will look like this:

```text
data/
├── README.md
├── pacs/
│   ├── download_pacs.sh
│   └── <domain_name>/          # art_painting, cartoon, photo, sketch
│       ├── train/              # Classes 0-6 (80%)
│       ├── test/               # Classes 0-6 (20%)
│       └── unlabel/            # Empty
└── domainnet/
    ├── download_domainnet.sh
    └── <domain_name>/          # clipart, infograph, painting, etc.
```