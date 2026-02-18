import torch
import numpy as np
import pandas as pd
import itertools
import os
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
MODEL_PATH = "connections_mnr_model"
CSV_PATH = "connections_data.csv"

class ConnectionsAI:
    def __init__(self, model_path):
        print(f"Loading AI from {model_path}...")
        self.model = SentenceTransformer(model_path)
        self.history = set() # To prevent repeating wrong guesses

    def get_candidates(self, words):
        """Finds the best coherent groups of 4 from the remaining words."""
        embeddings = self.model.encode(words)
        sim_matrix = self.model.similarity(embeddings, embeddings).numpy()
        
        candidates = []
        indices = list(range(len(words)))
        
        for group in itertools.combinations(indices, 4):
            # Score = Average pairwise similarity
            sub_sim = sim_matrix[list(group)][:, list(group)]
            # We prioritize groups that are TIGHT (high min similarity)
            score = np.mean(sub_sim)
            
            candidates.append({
                "words": [words[i] for i in group],
                "indices": group,
                "score": score
            })
            
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates

    def resolve_one_away(self, words_in_guess, all_remaining_words):
        """
        Logic for when we are 'One Away'.
        1. Identify the outlier in the current guess.
        2. Find the best replacement from the remaining pool.
        """
        print("   -> AI is analyzing the 'One Away' error...")
        
        # 1. Find Outlier in the guess
        embs = self.model.encode(words_in_guess)
        sims = self.model.similarity(embs, embs).numpy()
        # The word with the lowest average similarity to the others is the outlier
        avg_sims = np.mean(sims, axis=1)
        outlier_idx = np.argmin(avg_sims)
        outlier_word = words_in_guess[outlier_idx]
        
        keep_words = [w for i, w in enumerate(words_in_guess) if i != outlier_idx]
        
        print(f"   -> Outlier detected: '{outlier_word}'. Keeping: {keep_words}")
        
        # 2. Find best replacement
        best_swap = None
        best_score = -1.0
        
        # Potential replacements are words in the grid NOT in the current guess
        pool = [w for w in all_remaining_words if w not in words_in_guess]
        
        for candidate in pool:
            new_group = keep_words + [candidate]
            # Check if we already tried this
            if tuple(sorted(new_group)) in self.history: continue
            
            # Score this new group
            group_embs = self.model.encode(new_group)
            score = np.mean(self.model.similarity(group_embs, group_embs).numpy())
            
            if score > best_score:
                best_score = score
                best_swap = new_group
        
        return best_swap

    def play(self, puzzle_words):
        remaining = puzzle_words.copy()
        print(f"\n--- NEW GAME ---")
        print(f"Words: {remaining}")
        
        lives = 4
        
        while len(remaining) >= 4 and lives > 0:
            candidates = self.get_candidates(remaining)
            
            # Pick best candidate that isn't in history
            guess = None
            for cand in candidates:
                g = tuple(sorted(cand['words']))
                if g not in self.history:
                    guess = list(g)
                    break
            
            if not guess:
                print("AI is out of ideas!")
                break
                
            print(f"\nAI Guesses: {guess}")
            self.history.add(tuple(sorted(guess)))
            
            # INPUT LOOP
            res = input("Result (C=Correct, O=One Away, W=Wrong): ").upper().strip()
            
            if res == 'C':
                print("Cluster Cleared!")
                for w in guess: remaining.remove(w)
            elif res == 'W':
                lives -= 1
                print(f"Wrong. Lives left: {lives}")
            elif res == 'O':
                lives -= 1
                print(f"One Away. Lives left: {lives}")
                
                # TRIGGER THE FIX LOGIC
                new_guess = self.resolve_one_away(guess, remaining)
                if new_guess:
                    print(f"\nAI Retrying immediately with: {new_guess}")
                    self.history.add(tuple(sorted(new_guess)))
                    res2 = input("Result (C=Correct, O=One Away, W=Wrong): ").upper().strip()
                    if res2 == 'C':
                        print("Cluster Cleared!")
                        for w in new_guess: remaining.remove(w)
                    elif res2 == 'W' or res2 == 'O':
                        lives -= 1
            
            if len(remaining) == 0:
                print("\n🎉 VICTORY! The AI solved the puzzle.")
                break

if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        ai = ConnectionsAI(MODEL_PATH)
        
        # The puzzle from your log
        puzzle = ['MUSHROOM', 'STONE', 'BLAST', 'LEE', 'COMPASS', 
                  'SWELL', 'BALLOON', 'SNOWBALL', 'BONG', 'RIOT', 
                  'T-SQUARE', 'FORD', 'KICK', 'RULER', 'STENCIL', 'BALL']
        
        ai.play(puzzle)
    else:
        print("Model not found. Run training script first.")