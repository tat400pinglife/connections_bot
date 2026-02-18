import pandas as pd
import random
import torch
import pickle
import os
from collections import defaultdict
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample


CSV_FILE = 'Connections_Data_Cleaned.csv'
MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
SAVE_PATH = 'my_polysemy_model'
POLYSEMY_MAP_FILE = 'polysemy_map.pkl'
BATCH_SIZE = 16
EPOCHS = 4


def build_polysemy_map(df):
    """
    Scans the dataset to find words that appear in multiple different categories.
    Returns a dictionary: { 'Word': penalty_score }
    """
    print("Scanning dataset for double meanings...")
    word_categories = defaultdict(set)
    
    # Map every word to all categories it has ever belonged to
    for index, row in df.iterrows():
        w = row['Word']
        c = row['Category']
        word_categories[w].add(c)
        
    polysemy_map = {}
    danger_count = 0
    
    for word, cats in word_categories.items():
        # If a word appears in 2+ distinct categories, it is dangerous.
        # Score = (Num_Categories - 1) * 0.5
        # 1 Category -> 0.0 (Safe)
        # 2 Categories -> 0.5 (Risky)
        # 3 Categories -> 1.0 (Very Risky)
        if len(cats) > 1:
            polysemy_map[word] = (len(cats) - 1) * 0.5
            danger_count += 1
            
    print(f"Found {danger_count} ambiguous words (e.g., Bass, Date, Tie).")
    
    # Save this map for the Solver to use later!
    with open(POLYSEMY_MAP_FILE, 'wb') as f:
        pickle.dump(polysemy_map, f)
        
    return polysemy_map


def load_data(csv_path):
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    df['Word'] = df['Word'].astype(str).str.strip()
    
    # 1. Build the Map
    polysemy_map = build_polysemy_map(df)
    
    # 2. Group Data
    if 'Difficulty' not in df.columns: df['Difficulty'] = 1

    groups = df.groupby(['Category', 'Difficulty'])['Word'].apply(list).to_dict()
    all_words = df['Word'].unique().tolist()
    
    train_examples = []

    for (category, difficulty), words in groups.items():
        if len(words) < 4: continue
        
        # --- OVERSAMPLING STRATEGY ---
        # If a group contains a Polysemous word, we actually want to TRAIN ON IT MORE.
        # Why? Because the model needs extra practice to learn the specific context
        # that differentiates "Bass (Fish)" from "Bass (Instrument)".
        
        has_polysemy = any(w in polysemy_map for w in words[:4])
        
        repeats = 1
        if difficulty == 0: repeats = 5      # Easy = 5x
        if has_polysemy: repeats += 2        # Ambiguous = +2x (Total 7x if easy+ambiguous)
        
        # Positive Samples
        correct_group = sorted(words[:4])
        for _ in range(repeats):
            train_examples.append(InputExample(
                texts=[", ".join(correct_group), ""], 
                label=1.0 
            ))
        
        # Negative Samples
        # (Standard logic)
        if len(all_words) > 10:
            intruder = random.choice(all_words)
            while intruder in words: intruder = random.choice(all_words)
            bad_group = words[:3] + [intruder]
            random.shuffle(bad_group)
            train_examples.append(InputExample(
                texts=[", ".join(bad_group), ""], 
                label=0.0
            ))

    return train_examples

def train():
    train_examples = load_data(CSV_FILE)
    
    print(f"Initializing {MODEL_NAME}...")
    model = CrossEncoder(MODEL_NAME, num_labels=1)
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    loss_fct = torch.nn.MSELoss()

    print("Starting Polysemy-Aware Training...")
    model.fit(
        train_dataloader=train_dataloader,
        epochs=EPOCHS,
        loss_fct=loss_fct,
        show_progress_bar=True
    )
    
    model.save(SAVE_PATH)
    print(f"Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    train()