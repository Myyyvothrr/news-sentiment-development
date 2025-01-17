import sqlite3
import re
import pandas as pd
from datasets import Dataset, DatasetDict, Split


clean_chars = re.compile(r'[^A-Za-züöäÖÜÄß ]', re.MULTILINE)

#clean_http_urls = re.compile(r'https*\S+', re.MULTILINE)
# also clean "http", which are preprocessed urls from xlm
clean_http_urls = re.compile(r'(https*\S+)|(http)', re.MULTILINE)

clean_at_mentions = re.compile(r'@\S+', re.MULTILINE)


def replace_numbers(text: str) -> str:
    return text.replace("0"," null").replace("1"," eins").replace("2"," zwei") \
        .replace("3"," drei").replace("4"," vier").replace("5"," fünf") \
        .replace("6"," sechs").replace("7"," sieben").replace("8"," acht") \
        .replace("9"," neun")


def mdraw_clean_text(text):
    # clean slightly different from oliverguhr:
    # no max len
    # no emoji replacements
    # removes punctuation

    text = text.replace("\n", " ")
    text = clean_http_urls.sub('', text)
    text = clean_at_mentions.sub('', text)
    text = replace_numbers(text)
    text = clean_chars.sub('', text) # use only text chars
    text = ' '.join(text.split()) # substitute multiple whitespace with single whitespace
    text = text.strip().lower()

    return text


db_dir = "/home/daniel/data/uni/masterarbeit-sentiment/data/datasets/experiments/de/3sentiment-exact"
db_file = f"{db_dir}/datasets.db"

# remove data samples that are duplicates and not in the same split
remove_duplicates = False

# Get combined dataset
con = sqlite3.connect(db_file)

if not remove_duplicates:
    df_combined = pd.read_sql("SELECT * FROM dataset", con=con)
else:
    # remove own data "german-news-sentiment-bert" and base data "oliverguhr/*"
    # also remove all duplicates from "umsab/german" that are not in the same split
    df_combined = pd.read_sql('SELECT * FROM dataset WHERE NOT dataset = "german-news-sentiment-bert" AND NOT dataset LIKE "oliverguhr/%" AND (NOT dataset == "umsab/german" OR (dataset == "umsab/german" AND id NOT IN (SELECT `umsab/german` as id FROM duplicates WHERE same_split=0)))', con=con)

con.close()

# Preprocess and export combined dataset
dataset_combined = DatasetDict()

dataset_combined["train"] = Dataset.from_pandas(
    df_combined[df_combined["split"] == "train"],
    split=Split.TRAIN
)
dataset_combined["test"] = Dataset.from_pandas(
    df_combined[df_combined["split"] == "test"],
    split=Split.TEST
)
dataset_combined["validation"] = Dataset.from_pandas(
    df_combined[df_combined["split"] == "dev"],
    split=Split.VALIDATION
)


def preprocess_combined(sample):
    # text preprocessing
    sample["text"] = mdraw_clean_text(sample["text"])

    # convert to ttlab labels
    ttlab_mapping = {
        1: 0,
        -1: 1,
        0: 2
    }
    sample["labels"] = ttlab_mapping[sample["ttlab_label"]]

    return sample


dataset_combined = dataset_combined.map(preprocess_combined)

dataset_combined = dataset_combined.remove_columns(['original_label', 'ttlab_label', 'sentiment', 'split', '__index_level_0__'])

print(dataset_combined["train"][0])
print(dataset_combined["train"][10])
print(dataset_combined["train"][565])

combined_dataset_name = "mdraw"
if remove_duplicates:
    combined_dataset_name += "-unseen"
combined_dataset_path = f"/home/daniel/data/uni/masterarbeit-sentiment/data/datasets/experiments/de/3sentiment-exact/{combined_dataset_name}"

# save preprocessed dataset
dataset_combined.save_to_disk(combined_dataset_path)
