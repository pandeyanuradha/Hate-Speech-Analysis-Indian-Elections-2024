# Indian Elections 2024 Hate Speech & Topic Analysis

This repository contains scripts for collecting, analyzing, and modeling Facebook data related to the 2024 Indian elections. It includes:

1. **CrowdTangle Data Download**
2. **Hate Speech Analysis**
3. **Topic Modeling**

---

## 1. CrowdTangle Data Download

Script: `crowdtangle_data_download.py`

### Description
Downloads Facebook posts from pages and groups using the CrowdTangle API. Images and post JSONs are saved into structured folders.

### Usage

```bash
python crowdtangle_data_download.py -f -o "searchTerms=IndiaElection2024" -t 4 -s 2024-01-01
```

**Parameters:**
- `-f` : Facebook platform
- `-i` : Instagram platform
- `-o` : Additional query parameters for CrowdTangle API
- `-t` : Time interval in days (default 4)
- `-s` : Start date (YYYY-MM-DD)

---

## 2. Hate Speech Analysis

Script: `hate_speech_analysis.py`

### Description
Analyzes hate speech in Facebook posts using a pre-trained HuggingFace model. Processes both text messages and image captions.

### Usage

```bash
python hate_speech_analysis.py 2024
```

**Parameters:**
- `year` : Year of the posts to process (folder structure `Groups/<year>/` and `Pages/<year>/` required)

**Output:**
- CSV file: `df_hate_<year>.csv` with columns:
  - `huggingface_classification_list_message`
  - `huggingface_classification_message`
  - `huggingface_classification_list_imagetext`
  - `huggingface_classification_imagetext`

---

## 3. Topic Modeling

Script: `topic_modeling.py`

### Description
Performs topic modeling using BERTopic across multiple years of Facebook data. Trains one model per year and a combined model for all years.

### Folder Structure
- Groups data: `Groups/<year>/*.csv`
- Pages data: `Pages/<year>/*.csv`

### Usage

```bash
python topic_modeling.py
```

**Notes:**
- The script iterates over all years (2013-2024) by default.
- Outputs:
  - Per-year BERTopic models: `model_year_<year>`
  - Combined model for all years: `model_all_years_combined`
- Saves models using `safetensors` serialization.

---

## Dependencies

```bash
pip install pandas numpy requests ratelimit transformers torch bertopic glob2
```

**Optional:**
- `safetensors` for BERTopic model saving
- `tqdm` for progress bars

---

## Folder Setup

```
Repository/
│
├── Groups/
│   ├── 2023/
│   ├── 2024/
│
├── Pages/
│   ├── 2023/
│   ├── 2024/
│
├── crowdtangle_data_download.py
├── hate_speech_analysis.py
├── topic_modeling.py
├── README.md
```

---

## Workflow

1. **Download posts** using `crowdtangle_data_download.py`.
2. **Run hate speech analysis** on downloaded posts using `hate_speech_analysis.py`.
3. **Perform topic modeling** using `topic_modeling.py` to explore discussion trends.

---

## Author

Anuradha Pandey

