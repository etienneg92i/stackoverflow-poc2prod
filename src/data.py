"""Chargement et préparation des données StackOverflow (titre -> tag principal)."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "stackoverflow_posts.csv"
MODELS_DIR = ROOT / "models"

TOP_N_TAGS = 20
TEST_SIZE = 0.2
SEED = 42


def load_dataset(path=DATA_PATH, top_n=TOP_N_TAGS):
    """Un post = une ligne : son titre et son tag principal (tag_position == 0).

    On ne garde que les `top_n` tags les plus fréquents : les autres ont trop
    peu d'exemples (souvent < 10) pour être appris correctement.
    """
    df = pd.read_csv(path)
    df = df[df["tag_position"] == 0][["post_id", "title", "tag_name"]]
    df = df.dropna().drop_duplicates("post_id")
    top_tags = df["tag_name"].value_counts().head(top_n).index
    df = df[df["tag_name"].isin(top_tags)].reset_index(drop=True)
    return df


def split(df):
    """Split train/test stratifié, identique pour les deux modèles."""
    return train_test_split(
        df["title"].tolist(),
        df["tag_name"].tolist(),
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["tag_name"],
    )
