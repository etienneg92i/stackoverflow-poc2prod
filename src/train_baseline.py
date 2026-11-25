"""Modèle 1 : TF-IDF (mots + caractères) + régression logistique."""
import json

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline

from src.data import MODELS_DIR, SEED, load_dataset, split


def build_pipeline():
    # Les n-grammes de caractères captent bien les tokens techniques
    # ("c#", "asp.net", "jquery", "UITableView"...).
    features = FeatureUnion([
        ("word", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True,
                                 token_pattern=r"(?u)[\w#+.]+")),
        ("char", TfidfVectorizer(lowercase=True, analyzer="char_wb",
                                 ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    return Pipeline([("features", features), ("clf", clf)])


def main():
    df = load_dataset()
    X_train, X_test, y_train, y_test = split(df)
    print(f"{len(df)} posts, {df.tag_name.nunique()} tags | train={len(X_train)} test={len(X_test)}")

    search = GridSearchCV(build_pipeline(), {"clf__C": [1, 5, 10, 30]},
                          cv=5, scoring="f1_macro", n_jobs=-1)
    search.fit(X_train, y_train)
    print("Meilleurs paramètres :", search.best_params_)

    model = search.best_estimator_
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "best_params": search.best_params_,
    }
    print(classification_report(y_test, y_pred, zero_division=0))
    print(metrics)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "baseline.joblib")
    (MODELS_DIR / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Modèle sauvegardé dans", MODELS_DIR / "baseline.joblib")


if __name__ == "__main__":
    main()
