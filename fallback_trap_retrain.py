import pandas as pd
import itertools
import random
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

CSV_FILE = "Connections_Data_Cleaned.csv"
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SAVE_PATH = "connections_trap_model"
BATCH_SIZE = 32
EPOCHS = 3

def load_data():
    df = pd.read_csv(CSV_FILE)
    df.rename(columns={'Group Name': 'Category'}, inplace=True)
    df['Word'] = df['Word'].astype(str).str.strip()
    return df.groupby('Category')['Word'].apply(list).to_dict()

def build_trap_dataset(groups):
    examples = []
    all_words = list(set(sum(groups.values(), [])))

    for category, words in groups.items():
        if len(words) < 4:
            continue

        # Positive (valid group = NOT trap)
        for combo in itertools.combinations(words, 4):
            text = ", ".join(sorted(combo))
            examples.append(InputExample(
                texts=["Are these words a misleading or trap group?", text],
                label=0.0
            ))

        # 3+1 traps
        for combo in itertools.combinations(words, 3):
            intruder = random.choice(all_words)
            while intruder in words:
                intruder = random.choice(all_words)
            trap_group = list(combo) + [intruder]
            random.shuffle(trap_group)
            text = ", ".join(trap_group)
            examples.append(InputExample(
                texts=["Are these words a misleading or trap group?", text],
                label=1.0
            ))

        # 2+2 blends
        other_cats = [c for c in groups if c != category]
        for _ in range(5):
            other = random.choice(other_cats)
            if len(groups[other]) >= 2:
                part1 = random.sample(words, 2)
                part2 = random.sample(groups[other], 2)
                trap_group = part1 + part2
                random.shuffle(trap_group)
                text = ", ".join(trap_group)
                examples.append(InputExample(
                    texts=["Are these words a misleading or trap group?", text],
                    label=1.0
                ))

    return examples

def train():
    groups = load_data()
    examples = build_trap_dataset(groups)

    model = CrossEncoder(MODEL_NAME, num_labels=1)

    loader = DataLoader(examples, shuffle=True, batch_size=BATCH_SIZE)

    model.fit(
        train_dataloader=loader,
        epochs=EPOCHS,
        show_progress_bar=True
    )

    model.save(SAVE_PATH)
    print("Trap model saved.")

if __name__ == "__main__":
    train()
