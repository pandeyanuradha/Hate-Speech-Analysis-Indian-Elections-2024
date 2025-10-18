import pandas as pd
import numpy as np
import glob
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

import sys
year = sys.argv[1]

print("Year being processed - ", year)

df_combined = pd.DataFrame()

path = "Groups/" + year + "/*.csv"
for fname in glob.glob(path):
    df = pd.read_csv(fname)
    df_combined = pd.concat([df_combined, df])

path = "Pages/" + year + "/*.csv"
for fname in glob.glob(path):
    df = pd.read_csv(fname)
    df_combined = pd.concat([df_combined, df])

print("Number of records to process - ", df_combined.shape[0])

tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/indic-abusive-allInOne-MuRIL")
model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/indic-abusive-allInOne-MuRIL") 
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

df_combined_huggingface = df_combined.copy()
df_combined_huggingface["huggingface_classification_list_message"] = "NaN"
df_combined_huggingface["huggingface_classification_message"] = "NaN"
df_combined_huggingface["huggingface_classification_list_imagetext"] = "NaN"
df_combined_huggingface["huggingface_classification_imagetext"] = "NaN"

df_combined_huggingface.reset_index(drop=True, inplace=True)

# print(df_combined_huggingface.shape)

for i in range(df_combined_huggingface.shape[0]):

    if i%10000 == 0:
        print("Number of records processed = ", i)

    try:
        df_combined_huggingface.loc[i, "huggingface_classification_list_message"] = classifier(str(df_combined_huggingface.loc[i, "message"]))
        df_combined_huggingface.loc[i, "huggingface_classification_message"] = df_combined_huggingface.loc[i, "huggingface_classification_list_message"][0]["label"]

    except Exception as e:
        df_combined_huggingface.loc[i, "huggingface_classification_message"] = e

    # print(i)

    try:
        df_combined_huggingface.loc[i, "huggingface_classification_list_imagetext"] = classifier(str(df_combined_huggingface.loc[i, "imageText"]))
        df_combined_huggingface.loc[i, "huggingface_classification_imagetext"] = df_combined_huggingface.loc[i, "huggingface_classification_list_imagetext"][0]["label"]

    except Exception as e:
        df_combined_huggingface.loc[i, "huggingface_classification_imagetext"] = e

name = "df_hate_"+year+".csv"

# print(df_combined_huggingface.columns)

df_combined_huggingface.to_csv(name, encoding='utf-8', index=False)