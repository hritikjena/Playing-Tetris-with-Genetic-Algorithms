#!/usr/bin/env python3
"""
tetris_ga_play.py
Play Tetris on 4x10 board using GA weights.
Real Tetris behavior: pieces cannot spawn if blocked at top.
"""

import numpy as np
import csv
import time
import os
import random  # <-- added for random sequence generation

H, W = 20, 10

# --- Load GA weights ---
def load_best_weights(csv_file="best_ga_weights.csv"):
    """
    Load the best genome from a CSV file.
    Expects columns: fitness, w1, w2, w3, w4
    Returns: numpy array of weights [w1, w2, w3, w4]
    """
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Read the first (and only) row
            return np.array([float(row[f"w{i}"]) for i in range(1, 5)])


# --- Pieces ---
PIECES = {
    'I': [(0,0),(1,0),(2,0),(3,0)],
    'O': [(0,0),(1,0),(0,1),(1,1)],
    'T': [(0,0),(1,0),(2,0),(1,1)],
    'J': [(0,0),(0,1),(1,0),(2,0)],
    'L': [(0,0),(1,0),(2,0),(2,1)]
}

def rotations(blocks):
    variants = set()
    for r in range(4):
        transformed = blocks
        for _ in range(r):
            transformed = [(y,-x) for (x,y) in transformed]
        minx = min(p[0] for p in transformed)
        miny = min(p[1] for p in transformed)
        norm = tuple(sorted(((x-minx,y-miny) for x,y in transformed)))
        variants.add(norm)
    return [list(v) for v in sorted(variants)]

PIECE_ROTATIONS = {k: rotations(v) for k,v in PIECES.items()}

# --- Board functions ---
def can_place(board, shape, x_off, y_off):
    for sx,sy in shape:
        x,y = sx+x_off, sy+y_off
        if x<0 or x>=W or y<0 or y>=H:
            return False
        if board[y,x]:
            return False
    return True

def drop_y_for_x(board, shape, x_off):
    # Top-most spawn position
    maxy = max(y for x,y in shape)
    spawn_y = H - 1 - maxy

    # If piece cannot spawn at top, return None
    if not can_place(board, shape, x_off, spawn_y):
        return None

    # Drop down as far as possible
    cur_y = spawn_y
    while cur_y-1 >= 0 and can_place(board, shape, x_off, cur_y-1):
        cur_y -= 1
    return cur_y

def place_and_clear(board, shape, x_off, y_off):
    newb = board.copy()
    for sx,sy in shape:
        newb[sy+y_off, sx+x_off] = 1
    # clear full lines
    full = [r for r in range(H) if all(newb[r,:])]
    lines = len(full)
    if lines > 0:
        remain = [newb[r,:] for r in range(H) if r not in full]
        while len(remain) < H:
            remain.append(np.zeros(W,dtype=int))
        newb = np.vstack(remain)
    return newb, lines

# --- Scoring helper (NES-like points) ---
def lines_to_points(lines: int) -> int:
    return {0: 0, 1: 2, 2: 5, 3: 15, 4: 60}.get(lines, 0)

def get_features(board, move_score):
    heights = [0] * W
    holes = 0

    for x in range(W):
        blocks=0
        blockpresent=False

        for y in range(H-1, -1, -1):
            if board[y][x] == 1:
                blockpresent = True
                blocks += 1
            else:
                if blockpresent:
                    holes += 1

        heights[x] = blocks  # ✅ height = number of cells in that column

    agg_height = sum(heights)
    bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(W - 1))
    return np.array([agg_height, move_score, holes, bumpiness])


def play_sequence(weights, piece_sequence):
    board = np.zeros((H,W),dtype=int)
    total_score = 0
    total_lines = 0  # <-- track total lines cleared
    step = 0
    for piece_type in piece_sequence:
        step += 1
        print("="*40)
        print(f"Step {step} | Incoming Piece: {piece_type}")
        print(f"Before move -> Total lines: {total_lines}, Total score: {total_score}")

        best_val, best_board, best_score_move = -1e9, None, 0
        best_lines_move = 0

        for shape in PIECE_ROTATIONS[piece_type]:
            max_x = max(x for x,y in shape)
            for x_off in range(W - max_x):
                y_drop = drop_y_for_x(board, shape, x_off)
                if y_drop is None:
                    continue
                new_board, lines = place_and_clear(board, shape, x_off, y_drop)
                score_move = lines_to_points(lines)
                feats = get_features(new_board, score_move)
                val = np.dot(weights, feats)
                if val > best_val:
                    best_val, best_board = val, new_board
                    best_score_move = score_move
                    best_lines_move = lines

        if best_board is None:
            print(f"Cannot place piece: {piece_type} — stopping sequence.")
            break

        # Print chosen move's immediate outcome (lines & score for this step)
        print(f"Chosen move -> Lines cleared this step: {best_lines_move}, "
              f"Score this step: {best_score_move}")

        # Apply chosen move
        board = best_board
        total_lines += best_lines_move
        total_score += best_score_move

        print(f"After move  -> Total lines: {total_lines}, Total score: {total_score}")
        print("\n" + pretty_print_board(board) + "\n")

    return board, total_score, total_lines

def pretty_print_board(board):
    return "\n".join("".join('#' if c else '.' for c in row) for row in board[::-1])

# --- helper to create random sequence of requested length ---
def make_random_sequence(n, allowed=None):
    if allowed is None:
        allowed = list(PIECE_ROTATIONS.keys())  # ['I','O','T','J','L']
    return [random.choice(allowed) for _ in range(n)]

if __name__=="__main__":
    best_weights = load_best_weights()

    # Ask user for sequence length and generate a random sequence
    try:
        n = int(input("Enter length of random piece sequence (e.g., 30): ").strip())
        if n <= 0:
            raise ValueError
    except Exception:
        print("Invalid input. Using default length n = 30.")
        n = 30

    piece_sequence = make_random_sequence(n)
    print("Random piece sequence:")
    print(piece_sequence)

    print("GAME STATES")
    final_board, total_score, total_lines = play_sequence(best_weights, piece_sequence)
    print(f"Total score: {total_score}")
    print(f"Total lines cleared: {total_lines}\n")
    print(pretty_print_board(final_board))


    # results_csv = "tetris_results_GA.csv"

    # # Check if file already exists
    # file_exists = os.path.exists(results_csv)

    # # Convert sequence list to string
    # sequence_str = ",".join(piece_sequence)

    # # Rounded elapsed time
    # end=time.time()
    # print(end)

    # elapsed_time = end - start

    # # Append row
    # with open(results_csv, mode='a', newline='') as f:
    #     writer = csv.writer(f)

    #     # Add header once
    #     if not file_exists:
    #         writer.writerow(["Sequence", "Time (seconds)", "Best Score", "Total Lines"])

    #     writer.writerow([sequence_str, elapsed_time, total_score, total_lines])

    # print("✅ CSV updated:", results_csv)
