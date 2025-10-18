import pandas as pd
import numpy as np
import glob
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import sys

# ------------------------------------------------------------
# Usage: python hate_speech_analysis.py <year>
# Example: python hate_speech_analysis.py 2024
# ------------------------------------------------------------

year = sys.argv[1]
print("Year being processed -", year)

# ------------------------------------------------------------
# Step 1: Load data from Pages and Groups directories
# ------------------------------------------------------------

df_combined = pd.DataFrame()

# Load data from Groups
path_groups = f"Groups/{year}/*.csv"
for fname in glob.glob(path_groups):
    df = pd.read_csv(fname)
    df_combined = pd.concat([df_combined, df])

# Load data from Pages
path_pages = f"Pages/{year}/*.csv"
for fname in glob.glob(path_pages):
    df = pd.read_csv(fname)
    df_combined = pd.concat([df_combined, df])

print("Number of records to process:", df_combined.shape[0])

# ------------------------------------------------------------
# Step 2: Load pre-trained multilingual hate speech model
# ------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/indic-abusive-allInOne-MuRIL")
model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/indic-abusive-allInOne-MuRIL")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# ------------------------------------------------------------
# Step 3: Initialize output DataFrame and process rows
# ------------------------------------------------------------

df_combined_huggingface = df_combined.copy()
df_combined_huggingface["huggingface_classification_message"] = "NaN"
df_combined_huggingface["huggingface_classification_imagetext"] = "NaN"

df_combined_huggingface.reset_index(drop=True, inplace=True)

for i in range(df_combined_huggingface.shape[0]):

    if i % 1000 == 0:
        print("Records processed:", i)

    # Classify message text
    try:
        prediction_message = classifier(str(df_combined_huggingface.loc[i, "message"]))
        df_combined_huggingface.loc[i, "huggingface_classification_message"] = prediction_message[0]["label"]
    except Exception as e:
        df_combined_huggingface.loc[i, "huggingface_classification_message"] = str(e)

    # Classify image text if available
    try:
        prediction_image = classifier(str(df_combined_huggingface.loc[i, "imageText"]))
        df_combined_huggingface.loc[i, "huggingface_classification_imagetext"] = prediction_image[0]["label"]
    except Exception as e:
        df_combined_huggingface.loc[i, "huggingface_classification_imagetext"] = str(e)

# ------------------------------------------------------------
# Step 4: Save processed results
# ------------------------------------------------------------

output_name = f"df_hate_{year}.csv"
df_combined_huggingface.to_csv(output_name, encoding="utf-8", index=False)
print(f"Saved results to {output_name}")
