import random
import numpy as np
import tensorflow as tf
import time
import pickle
from tqdm import tqdm
from board import board_to_bitboard, bitboard_to_board, generate_positions_clear, get_bitboard_sum, block_id_to_block_info, get_resulting_position, clear_position_info
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('bb3-10k-gen6.keras')

num_games_test = 250

def print_block_from_id(id):
    block = np.zeros((5, 5), dtype=int)
    for j in range(5):
        for i in range(5):
            if (id % 2 == 1):
                block[j, i] = 1
            id = id // 2
    # Print the block in a more visual way
    for row in block:
        print(' '.join('█' if cell == 1 else '⋅' for cell in row))

def evaluate_positions(positions):
    # Reshape positions to match model's expected input shape (batch_size, height, width, channels)
    positions_reshaped = np.array(positions).reshape(-1, 8, 8, 1)
    
    # Get model predictions for the batch
    predictions = model.predict(positions_reshaped, verbose=0)
    
    return predictions[:, 0]

def clear_lines(position):
    clear_rows = []
    # Clear full rows
    for i in range(8):
        if np.min(position[i, :]) == 1:
            clear_rows.append(i)
            # position[i, :] = 0
    
    clear_cols = []
    # Clear full columns
    for j in range(8):
        if np.min(position[:, j]) == 1:
            clear_cols.append(j)
            # position[:, j] = 0
    
    for row in clear_rows:
        position[row, :] = 0
    
    for col in clear_cols:
        position[:, col] = 0

def get_random_position(frequency=0.4):
    position = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            if random.random() < frequency:
                position[i][j] = 1
    clear_lines(position)
    return position

def print_position(position):
    # ANSI color codes for different colors
    colors = [
        '\033[92m',  # Green
    ]
    reset = '\033[0m'  # Reset color
    
    # Choose a random color for the entire position
    color = random.choice(colors)
    
    # Print the 8x8 position with color
    for row in position:
        print(''.join(f'{color}██{reset}' if cell == 1 else '⋅⋅' for cell in row))
    
    print()

positions = []
position_1 = np.zeros((8, 8))
position_2 = np.zeros((8, 8))
# Checker board
for i in range(8):
    for j in range(8):
        if (i+j) % 2 == 0:
            position_2[i][j] = 1

position_3 = np.zeros((8, 8))
# 6x5 block + 2
for i in range(6):
    for j in range(5):
        position_3[i][j] = 1
    if (i < 2):
        position_3[i][5] = 1

positions = [position_1, position_2, position_3]

evals = evaluate_positions(positions)

pos_evals = []
for position, eval in zip(positions, evals):
    pos_evals.append([position, eval])

pos_evals.sort(key = lambda x: x[1])

for pos, eval in pos_evals:
    print(f"Eval: {eval:.2f}")
    print_position(pos)


blocks_encountered = []
# Open the file in binary read mode ('rb')
with open('blocks_encountered_new.pkl', 'rb') as f:
    blocks_encountered = pickle.load(f)

blocks_normal = []
blocks_difficult = []

for bitboard, (block_left, block_middle, block_right) in blocks_encountered:
    if bitboard.bit_count() < 35:
        blocks_normal.append([block_left, block_middle, block_right])
    else:
        blocks_difficult.append([block_left, block_middle, block_right])

def get_best_move(position, block_ids, temperature=0):
    position_bitboard = board_to_bitboard(position)
    blocks = []
    for i, block_id in enumerate(block_ids):
        blocks.append([i, block_id_to_block_info(block_id)])

    positions_clear = {}
    positions_nonclear = {}
    generate_positions_clear(position_bitboard, blocks, positions_clear, positions_nonclear)

    if len(positions_clear) == 0:
        return None
    
    positions = positions_clear
    
    max_eval = 0
    best_permutation = None
    best_placement = None

    positions_2d = []
    positions_values = list(positions.values())  # Convert to list to maintain order
    for position in positions.keys():
        positions_2d.append(bitboard_to_board(position))
    positions_2d = np.array(positions_2d).reshape(-1, 8, 8, 1)

    raw_evals = model.predict(positions_2d, verbose=0)
    evals = np.log(1-raw_evals)/np.log(0.8)
    # Add random normal noise to each evaluation
    evals = evals + np.random.normal(0, 1, size=evals.shape)*temperature
    max_eval_index = np.argmax(evals)
    max_eval = evals[max_eval_index]

    best_permutation, best_placement = positions_values[max_eval_index]

    return best_permutation, best_placement, max_eval

def play_move(position, blocks, permutation, placement):
    position_bitboard = board_to_bitboard(position)

    for i in range(len(permutation)):
        current_block_id = blocks[permutation[i]]
        
        current_block = block_id_to_block_info(current_block_id)

        position_bitboard = get_resulting_position(position_bitboard, current_block, placement[i])
        position_bitboard, cc = clear_position_info(position_bitboard)
    
    return bitboard_to_board(position_bitboard)
    
def play_one_game(temperature=0):
    position = np.zeros((8, 8))
    position_history = [position]

    while True:
        if board_to_bitboard(position).bit_count() < 35:
            block_list = random.choice(blocks_normal)
        else:
            block_list = random.choice(blocks_difficult)
        random.shuffle(block_list)
        block_1, block_2, block_3 = block_list

        best_move = get_best_move(position, [block_1, block_2, block_3], temperature)
        if best_move is None:
            break
        
        permutation, placement, eval = best_move
        position = play_move(position, [block_1, block_2, block_3], permutation, placement)

        position_history.append(position)
    
    return position_history

game_lengths = []
for _ in tqdm(range(num_games_test)):
    game_length = len(play_one_game())
    while game_length == 1:
        game_length = len(play_one_game())
    game_lengths.append(game_length)

print(min(game_lengths))
print("Average game length:", np.mean(game_lengths))

mean_length = np.mean(game_lengths)

plt.figure(figsize=(12, 6))
plt.hist(game_lengths, bins=20, align='left', rwidth=0.8)
plt.axvline(mean_length, color='red', linestyle='--', label=f'Mean: {mean_length:.2f}')
plt.title('Distribution of Game Length')
plt.xlabel('Game Length')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()