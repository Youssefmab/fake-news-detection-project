# Data Directory

## Dataset: COVID-19 Fake News Dataset

This project uses the **COVID-19 Fake News Dataset** available on Kaggle.

| Property       | Value                                              |
| -------------- | -------------------------------------------------- |
| **Source**      | [Kaggle](https://www.kaggle.com/datasets/elroyggj/covid19-fake-news-dataset-nlp) |
| **Size**        | ~10 700 posts and articles                        |
| **Labels**      | **Fake** (0) / **Real** (1)                       |
| **Format**      | CSV                                                |
| **Language**    | English                                            |

### Description

The dataset contains news articles and social-media posts related to the
COVID-19 pandemic. Each sample is labeled as either **Fake** or **Real**,
making it suitable for supervised binary classification.

### Directory Structure

After downloading and processing, the directory should look like this:

```
data/
├── README.md              ← This file
├── raw/                   ← Original downloaded files
│   ├── train.csv
│   └── test.csv
└── processed/             ← Cleaned / feature-engineered data
    ├── train_clean.csv
    ├── val_clean.csv
    └── test_clean.csv
```

### How to Download

#### Option 1 — Kaggle CLI (recommended)

1. Install the Kaggle Python package:

   ```bash
   pip install kaggle
   ```

2. Place your Kaggle API token (`kaggle.json`) in the appropriate location:
   - **Linux / macOS:** `~/.kaggle/kaggle.json`
   - **Windows:** `C:\Users\<username>\.kaggle\kaggle.json`

   You can download the token from <https://www.kaggle.com/settings> under
   *API → Create New Token*.

3. Run the download script from the project root:

   ```bash
   python download_dataset.py
   ```

#### Option 2 — Manual Download

1. Visit <https://www.kaggle.com/datasets/elroyggj/covid19-fake-news-dataset-nlp>.
2. Click **Download** (you need a free Kaggle account).
3. Extract the ZIP file into `data/raw/`.

### Data Fields

| Column  | Type   | Description                          |
| ------- | ------ | ------------------------------------ |
| `text`  | string | The news article or social-media post|
| `label` | int    | 0 = Fake, 1 = Real                  |
