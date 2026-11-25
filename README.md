# StackOverflow tag prediction — From PoC to Prod

Prédit automatiquement le tag principal d'une question StackOverflow à partir de son titre.

Données : `stackoverflow_posts.csv` (titre + tags). On garde le tag principal (`tag_position == 0`)
et les 20 tags les plus fréquents (~3 300 posts, 71 % du dataset), split train/test 80/20 stratifié.

## Modèles

| Modèle | Accuracy | F1 macro |
|---|---|---|
| TF-IDF (mots + caractères) + régression logistique | 0.60 | 0.54 |
| BERT (`bert-base-uncased`, fine-tuné 4 epochs) | 0.57 | 0.46 |

## Utilisation

```bash
pip install -r requirements.txt

python -m src.train_baseline   # modèle 1 -> models/baseline.joblib
python -m src.train_bert       # modèle 2 -> models/bert/  (~25 min sur CPU)

python -m src.predict "How to parse JSON in Python?"
python -m src.predict --model bert "How to parse JSON in Python?"
python -m src.predict          # mode interactif
```

En Python :

```python
from src.predict import predict_tags
predict_tags("jQuery click event fires twice", model="baseline", top_k=3)
# [('jquery', 0.72), ('javascript', 0.24), ...]
```
