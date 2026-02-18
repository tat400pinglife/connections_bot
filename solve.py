import itertools
import os
import time
import pickle
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer, util


MODEL_PATH = 'my_polysemy_model'
EMBEDDER_NAME = 'sentence-transformers/all-mpnet-base-v2'
SEARCH_DEPTH = 100 # How many groups to feed the global partitioner


PUZZLE_WORDS = [
    "Heavy", "Crested", "Bad", "Feather",
    "Fly", "Tease", "Bantam", "Topical",
    "Curl", "Sour", "Wicked", "Leghorn",
    "shaving", "crimp", "Free-range", "rad"
]

# Set to None for manual play
ANSWER_KEY = None 


def get_coherence_penalty(group, embedder):
    vecs = embedder.encode(group)
    cos_sims = util.cos_sim(vecs, vecs).numpy()
    mask = ~np.eye(4, dtype=bool)
    pairs = cos_sims[mask]
    if np.min(pairs) < 0.25: return -0.5
    return 0.0

def find_best_partition(candidates, words_remaining):
    """
    Finds 4 disjoint groups that maximize Total Score.
    """
    if not words_remaining: return [], 0.0
    best_partition, best_score = None, -999.0

    for group, score in candidates:
        group_set = set(group)
        if group_set.issubset(words_remaining):
            sub_partition, sub_score = find_best_partition(candidates, words_remaining - group_set)
            if sub_partition is not None:
                total = score + sub_score
                if total > best_score:
                    best_score = total
                    best_partition = [group] + sub_partition
    return best_partition, best_score


def get_adaptive_guess(words_list, model, embedder, constraints):
    """
    constraints: A dictionary of logic derived from past guesses.
    {
        'banned': set(tuples),          # Exact groups that are wrong
        'dead_cores': set(tuples),      # 3-word combos proven invalid (from WRONG result)
        'must_have_cores': set(tuples)  # 3-word combos proven valid (from ONE AWAY result)
    }
    """
    print(f"   Thinking... (Constraints: {len(constraints['dead_cores'])} dead cores, {len(constraints['must_have_cores'])} valid cores)")
    
    # A. Generate All Combinations
    all_combos = list(itertools.combinations(words_list, 4))
    inputs = [(", ".join(sorted(c)), "") for c in all_combos]
    scores = model.predict(inputs)
    
    candidates = []
    
    for i, group in enumerate(all_combos):
        g_tuple = tuple(sorted(group))
        g_set = set(group)
        
        # EXACT BANS
        if g_tuple in constraints['banned']:
            continue

        # DEAD CORES (The "Wrong" Logic)
        # If we previously guessed [A,B,C,D] and got "WRONG",
        # then [A,B,C] is NOT a valid core.
        # So [A,B,C,X] must also be invalid.
        is_dead = False
        if constraints['dead_cores']:
            # Check if this group contains ANY dead core of 3 words
            # (We check combinations of 3 within this group of 4)
            subsets_of_3 = itertools.combinations(group, 3)
            for sub in subsets_of_3:
                if tuple(sorted(sub)) in constraints['dead_cores']:
                    is_dead = True
                    break
        if is_dead: continue

        # MUST HAVE CORES (The "One Away" Logic)
        # If we have "must_have" constraints, we ONLY consider groups 
        # that are built around those specific 3-word cores.
        # (Only applies if we are actively chasing a One Away hint)
        if constraints['must_have_cores']:
            is_valid_structure = False
            subsets_of_3 = itertools.combinations(group, 3)
            for sub in subsets_of_3:
                if tuple(sorted(sub)) in constraints['must_have_cores']:
                    is_valid_structure = True
                    break
            if not is_valid_structure:
                continue

        # Score & Add
        s = scores[i] + get_coherence_penalty(group, embedder)
        candidates.append((group, s))

    # B. Global Partitioning
    candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = candidates[:SEARCH_DEPTH]
    
    best_partition, _ = find_best_partition(top_candidates, set(words_list))
    
    if best_partition:
        # Return the highest scoring group from the partition
        return max(best_partition, key=lambda g: [c[1] for c in candidates if c[0]==g][0])
    elif top_candidates:
        return top_candidates[0][0]
    else:
        return None

def get_feedback_auto(guess, key):
    guess_set = set(guess)
    if guess_set in key: return 'c'
    for group in key:
        if len(guess_set.intersection(group)) == 3: return 'o'
    return 'w'

def get_feedback_manual():
    while True:
        res = input("Result? (c/o/w): ").lower().strip()
        if res in ['c', 'o', 'w']: return res

def run():
    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return

    model = CrossEncoder(MODEL_PATH)
    embedder = SentenceTransformer(EMBEDDER_NAME)
    
    remaining_words = set(PUZZLE_WORDS)
    lives = 4
    solved = 0
    
    # CONSTRAINT MEMORY
    constraints = {
        'banned': set(),
        'dead_cores': set(),      # Combinations of 3 words that CANNOT be together
        'must_have_cores': set()  # Combinations of 3 words that MUST be together
    }

    while solved < 4 and lives > 0:
        current_list = list(remaining_words)
        
        # 1. GET BEST GUESS (Respecting Constraints)
        guess = get_adaptive_guess(current_list, model, embedder, constraints)
        
        if not guess:
            print("Error: Constraints are too tight. No valid moves left.")
            # Fallback: clear constraints slightly
            constraints['must_have_cores'] = set()
            continue

        print(f"\n[Lives: {lives}] Guessing: {guess}")
        
        # 2. FEEDBACK
        if ANSWER_KEY:
            res = get_feedback_auto(guess, ANSWER_KEY)
            print(f" >> Auto: {res.upper()}")
            time.sleep(0.5)
        else:
            res = get_feedback_manual()
            
        guess_tuple = tuple(sorted(guess))
        
        # 3. ADAPTIVE LOGIC UPDATE
        if res == 'c':
            print(" >> CORRECT! Constraints reset.")
            remaining_words -= set(guess)
            solved += 1
            # Reset constraints for the remaining board
            constraints['banned'] = set()
            constraints['dead_cores'] = set()
            constraints['must_have_cores'] = set()
            
        elif res == 'w':
            print(" >> WRONG. Aggressive pruning enabled.")
            lives -= 1
            constraints['banned'].add(guess_tuple)
            
            # LOGIC: "Wrong" means at most 2 words are correct.
            # Therefore, ANY subset of 3 words from this group is INVALID.
            # We ban all 4 combinations of 3 words.
            subsets_3 = itertools.combinations(guess, 3)
            for sub in subsets_3:
                constraints['dead_cores'].add(tuple(sorted(sub)))
            
            # Also, if we were chasing a "One Away" lead, we just proved it wrong.
            # Clear the "must have" to unlock the board again.
            constraints['must_have_cores'] = set()

        elif res == 'o':
            print(" >> ONE AWAY! Locking onto valid cores.")
            lives -= 1
            constraints['banned'].add(guess_tuple)
            
            # LOGIC: "One Away" means exactly 3 words are correct.
            # We don't know WHICH 3, but we know it's one of the 4 possibilities.
            # We tell the solver: "Your next guess MUST contain one of these 4 cores."
            subsets_3 = itertools.combinations(guess, 3)
            
            # We add these to 'must_have_cores'
            # Note: We wipe the old must-haves because this is fresh info
            constraints['must_have_cores'] = set() 
            for sub in subsets_3:
                constraints['must_have_cores'].add(tuple(sorted(sub)))

    if solved == 4: print("VICTORY")
    else: print("DEFEAT")

if __name__ == "__main__":
    run()