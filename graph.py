# plot_tetris_fitness.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

# ==== Config ====
CSV_FILE = "ga_weights.csv"   # keep your data in this file
TITLE = "Tetris Fitness Scores over Generations"

# ==== Load ====
df = pd.read_csv(CSV_FILE)

# Ensure types
df['gen'] = df['gen'].astype(int)
df['fitness'] = df['fitness'].astype(float)

# ==== Group by generation ====
gens = sorted(df['gen'].unique())
gen_to_fits = {g: df.loc[df['gen'] == g, 'fitness'].to_numpy() for g in gens}

# Sort each generation's fitness descending (ranked: 0=best)
ranked_by_gen = {g: np.sort(fits)[::-1] for g, fits in gen_to_fits.items()}
max_count = max(len(v) for v in ranked_by_gen.values())

# Build rank-series across gens: rank_lines[i] is the i-th ranked fitness over generations
rank_lines = []
for i in range(max_count):
    series = []
    for g in gens:
        fits = ranked_by_gen[g]
        if i < len(fits):
            series.append(fits[i])
        else:
            series.append(np.nan)  # if that rank doesn't exist for gen
    rank_lines.append(np.array(series))

# Compute summary curves
best = rank_lines[0]                             # rank 0
avg  = np.array([np.nanmean(ranked_by_gen[g]) for g in gens])
second_best = rank_lines[1] if len(rank_lines) > 1 else np.full_like(best, np.nan)
worst = np.array([np.nanmin(ranked_by_gen[g]) for g in gens])

# ==== Plot ====
plt.figure(figsize=(9.5, 6.8))

# Spaghetti of all ranks (except best) using a red→orange colormap by rank
cmap = get_cmap('autumn')  # red (low) → orange/yellow (high)
for i in range(1, len(rank_lines)):
    y = rank_lines[i]
    # color by rank percentile (i/max_count)
    color = cmap(i / max(2, len(rank_lines) - 1))
    plt.plot(gens, y, linewidth=0.6, alpha=0.65, color=color)

# Highlight lines
plt.plot(gens, best, marker='o', linewidth=2.2, color='green', label='Best Fitness')
plt.plot(gens, avg,  marker='o', linewidth=2.0, color='blue',  label='Average Fitness')
if len(rank_lines) > 1:
    plt.plot(gens, second_best, linewidth=1.5, color='red',   label='Second Best Fitness')
plt.plot(gens, worst, linewidth=1.2, color='orange', label='Worst Fitness')

plt.title(TITLE)
plt.xlabel("Generation")
plt.ylabel("Average Tetris score per move")
plt.grid(True, linestyle=':', linewidth=0.6, alpha=0.7)
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
