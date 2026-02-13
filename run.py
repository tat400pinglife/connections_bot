import itertools
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from ortools.linear_solver import pywraplp

SEMANTIC_MODEL_PATH = "connections_semantic"
TRAP_MODEL_PATH = "connections_trap_model"

PUZZLE_WORDS = [
    "may", "yank", "a", "card",
    "frozen", "produce", "dancing", "dairy",
    "make", "jay", "fast", "form",
    "drag", "firm", "mold", "tight"
]

# ===============================
# LOAD MODELS
# ===============================
semantic_model = SentenceTransformer(SEMANTIC_MODEL_PATH)
trap_model = CrossEncoder(TRAP_MODEL_PATH)

# ===============================
# SCORE GROUPS
# ===============================
def score_groups(words):
    combos = list(itertools.combinations(words, 4))
    group_texts = [", ".join(sorted(c)) for c in combos]

    # Semantic internal coherence
    embeddings = semantic_model.encode(group_texts, convert_to_tensor=True)
    sims = util.cos_sim(embeddings, embeddings)

    semantic_scores = []
    for i in range(len(combos)):
        vec = embeddings[i]
        # average pairwise similarity within group
        word_embeds = semantic_model.encode(combos[i], convert_to_tensor=True)
        pairwise = util.cos_sim(word_embeds, word_embeds)
        mask = ~np.eye(4, dtype=bool)
        coherence = pairwise.cpu().numpy()[mask].mean()
        semantic_scores.append(coherence)

    # Trap probabilities
    trap_inputs = [["Are these words a misleading or trap group?", t] for t in group_texts]
    trap_probs = trap_model.predict(trap_inputs)

    final_scores = []
    for s, t in zip(semantic_scores, trap_probs):
        final = s - 1.5 * float(t)
        final_scores.append(final)

    return combos, final_scores

# ===============================
# ILP PARTITION SOLVER
# ===============================
def solve_partition(words):
    combos, scores = score_groups(words)

    solver = pywraplp.Solver.CreateSolver('SCIP')

    x = []
    for i in range(len(combos)):
        x.append(solver.BoolVar(f'x_{i}'))

    # Objective
    solver.Maximize(
        solver.Sum(scores[i] * x[i] for i in range(len(combos)))
    )

    # Each word used exactly once
    for word in words:
        solver.Add(
            solver.Sum(
                x[i] for i, combo in enumerate(combos)
                if word in combo
            ) == 1
        )

    # Exactly 4 groups
    solver.Add(solver.Sum(x) == 4)

    solver.Solve()

    solution = []
    for i in range(len(combos)):
        if x[i].solution_value() > 0.5:
            solution.append(combos[i])

    return solution

# ===============================
# RUN
# ===============================
if __name__ == "__main__":
    result = solve_partition(PUZZLE_WORDS)
    print("\nOptimal Partition:")
    for group in result:
        print(group)
