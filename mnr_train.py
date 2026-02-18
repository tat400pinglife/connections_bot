import pandas as pd
import random
import torch
import numpy as np
import os
import itertools
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader

CSV_PATH = "wide.csv"
MODEL_SAVE_PATH = "connections_mnr_model"
EPOCHS = 10
BATCH_SIZE = 32

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    train_samples = []
    
    # We create chains of (Anchor, Positive)
    # MNRLoss expects: [Anchor, Positive, Negative(optional)]
    # But it implicitly uses other batch items as negatives, which is superior.
    
    print(f"Loading data from {csv_path}...")
    for idx, row in df.iterrows():
        # Clean words
        words = [str(row[c]).strip() for c in ['word1', 'word2', 'word3', 'word4'] 
                 if pd.notna(row[c])]
        
        if len(words) != 4: continue
        
        # Generate all valid pairs in the category
        # (w1, w2), (w2, w3), etc.
        # This teaches the model that ALL these words are interchangeable synonyms
        pairs = list(itertools.combinations(words, 2))
        for p1, p2 in pairs:
            train_samples.append(InputExample(texts=[p1, p2]))
            
    return train_samples

# TRAINING (Multiple Negatives Ranking Loss)
def train():
    train_samples = load_data(CSV_PATH)
    random.shuffle(train_samples)
    
    # Split for valid
    split = int(len(train_samples) * 0.9)
    train_data = train_samples[:split]
    
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
    
    # Initialize Model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # LOSS FUNCTION: MultipleNegativesRankingLoss
    # This maximizes similarity for (a, p) and minimizes it for (a, any_other_p_in_batch)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    print("Starting MNRLoss Training...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=100,
        show_progress_bar=True
    )
    
    model.save(MODEL_SAVE_PATH)
    print("Model Saved.")

# EXACT PARTITION SOLVER
class ConnectionsSolver:
    def __init__(self, model_path):
        self.model = SentenceTransformer(model_path)
        
    def solve(self, grid_words):
        # 1. Encode all words
        embeddings = self.model.encode(grid_words)
        
        # 2. Calculate Similarity Matrix
        # shape (16, 16)
        sim_matrix = self.model.similarity(embeddings, embeddings).numpy()
        
        # 3. Score all ~1820 possible groups of 4
        candidates = []
        indices = list(range(16))
        
        for group in itertools.combinations(indices, 4):
            # Internal coherence score
            # Average similarity of all pairs in the group
            # We subtract the mean similarity to the *outside* world (regularization)
            
            sub_sim = sim_matrix[list(group)][:, list(group)]
            coherence = (np.sum(sub_sim) - 4) / 12.0 # Avg of off-diagonals
            
            candidates.append({
                "indices": set(group),
                "words": [grid_words[i] for i in group],
                "score": coherence
            })
        
        # Sort by coherence
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 4. Find valid partition (Exact Cover Problem)
        # We need 4 disjoint groups
        return self._find_partition(candidates, set(range(16)))

    def _find_partition(self, candidates, remaining_indices, path=[]):
        # Recursive backtracking to find 4 non-overlapping groups
        if not remaining_indices:
            return path
        
        # Try top candidates that fit in remaining slots
        # Optimization: Only check candidates composed of remaining words
        for cand in candidates:
            # If this group uses used words, skip
            if not cand['indices'].issubset(remaining_indices):
                continue
            
            # Recurse
            result = self._find_partition(candidates, remaining_indices - cand['indices'], path + [cand])
            if result: return result
            
        return None

if __name__ == "__main__":
    if not os.path.exists(MODEL_SAVE_PATH):
        train()
        
    # Solve Sample
    print("Testing Solver on Sample Puzzle...")
    solver = ConnectionsSolver(MODEL_SAVE_PATH)
    
    # Load a real puzzle from your CSV to test memorization/generalization
    df = pd.read_csv(CSV_PATH)
    if len(df) >= 4:
        sample = df.sample(4)
        puzzle = sample[['word1', 'word2', 'word3', 'word4']].values.flatten().tolist()
        random.shuffle(puzzle)
        
        print(f"Puzzle: {puzzle}")
        
        solution = solver.solve(puzzle)
        
        if solution:
            print("\nAI Solution:")
            for i, group in enumerate(solution):
                print(f"Group {i+1} (Score {group['score']:.2f}): {group['words']}")
        else:
            print("Failed to find valid partition.")