import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(path):
    df = pd.read_csv(path, encoding='latin1')
    df["Sentence #"] = df["Sentence #"].ffill()
    grouped = df.groupby("Sentence #")
    full_sentences = [
        list(zip(row["Word"].tolist(), row["POS"].tolist(), row["Tag"].tolist()))
        for _, row in grouped
    ]
    return full_sentences

def split_sentences(sentences):
    train_val, test = train_test_split(sentences, test_size=0.2, random_state=42)
    val, train = train_test_split(train_val, test_size=0.875, random_state=42)
    return train, val, test
