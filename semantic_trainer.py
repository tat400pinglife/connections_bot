import pandas as pd
import itertools
import random
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader

CSV_FILE = "Connections_Data_Cleaned.csv"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # stronger backbone
SAVE_PATH = "connections_semantic"
BATCH_SIZE = 64
EPOCHS = 4
HARD_NEG_PER_GROUP = 3

def load_groups():
    df = pd.read_csv(CSV_FILE)
    df.rename(columns={'Group Name': 'Category'}, inplace=True)
    df['Word'] = df['Word'].astype(str).str.strip()
    return df.groupby('Category')['Word'].apply(list).to_dict()

def build_positive_pairs(groups):
    examples = []
    for category, words in groups.items():
        if len(words) < 4:
            continue
        for combo in itertools.combinations(words, 4):
            group_text = ", ".join(sorted(combo))
            examples.append(InputExample(texts=[category, group_text]))
    return examples

def hard_negative_mining(model, groups):
    hard_examples = []

    for category, words in groups.items():
        if len(words) < 4:
            continue

        all_words = list(set(sum(groups.values(), [])))

        valid_sets = set(tuple(sorted(c)) for c in itertools.combinations(words, 4))

        # generate many candidates
        candidates = list(itertools.combinations(all_words, 4))

        texts = [", ".join(sorted(c)) for c in candidates]
        embeddings = model.encode(texts, convert_to_tensor=True)

        cat_embedding = model.encode(category, convert_to_tensor=True)

        sims = util.cos_sim(cat_embedding, embeddings)[0].cpu().numpy()

        ranked = sorted(zip(candidates, sims), key=lambda x: -x[1])

        added = 0
        for combo, score in ranked:
            if tuple(sorted(combo)) not in valid_sets:
                group_text = ", ".join(sorted(combo))
                hard_examples.append(
                    InputExample(texts=[category, group_text])
                )
                added += 1
                if added >= HARD_NEG_PER_GROUP:
                    break

    return hard_examples

def train():
    groups = load_groups()

    model = SentenceTransformer(MODEL_NAME)

    positives = build_positive_pairs(groups)

    train_data = positives

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        train_loader = DataLoader(
            train_data,
            shuffle=True,
            batch_size=BATCH_SIZE
        )

        train_loss = losses.MultipleNegativesRankingLoss(model)

        model.fit(
            train_objectives=[(train_loader, train_loss)],
            epochs=1,
            show_progress_bar=True
        )

        print("Mining hard negatives...")
        hard_negs = hard_negative_mining(model, groups)

        train_data = positives + hard_negs

    model.save(SAVE_PATH)
    print(f"Saved SOTA semantic model to {SAVE_PATH}")

if __name__ == "__main__":
    train()
