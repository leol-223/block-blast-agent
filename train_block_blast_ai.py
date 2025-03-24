import pickle
import numpy as np
import random
import time
from tqdm import tqdm
import tensorflow as tf

from board import board_to_bitboard, bitboard_to_board, generate_positions_clear, get_bitboard_sum, block_id_to_block_info, get_resulting_position, clear_position_info

blocks_encountered = []
# Open the file in binary read mode ('rb')
with open('blocks_encountered_new.pkl', 'rb') as f:
    blocks_encountered = pickle.load(f)
model = None

blocks_normal = []
blocks_difficult = []

for bitboard, (block_left, block_middle, block_right) in blocks_encountered:
    if bitboard.bit_count() < 35:
        blocks_normal.append([block_left, block_middle, block_right])
    else:
        blocks_difficult.append([block_left, block_middle, block_right])

def get_position_hash(position):
    hash = 0
    for i in range(8):
        for j in range(8):
            if position[i][j] == 1:
                hash += 1
            hash *= 2
    return hash

def print_blocks_from_ids(ids):
    # Create blocks
    blocks = []
    for id in ids:
        block = np.zeros((5, 5), dtype=int)
        for j in range(5):
            for i in range(5):
                if (id % 2 == 1):
                    block[j, i] = 1
                id = id // 2
        blocks.append(block)
    
    # ANSI color codes for different colors
    colors = [
        '\033[91m',  # Red
        '\033[92m',  # Green
        '\033[93m',  # Yellow
        '\033[94m',  # Blue
        '\033[95m',  # Purple
    ]
    reset = '\033[0m'  # Reset color
    
    # Assign a random color to each block
    block_colors = [random.choice(colors) for _ in blocks]
    
    # Print all blocks row by row
    for row_idx in range(5):
        # For each block, print the current row
        for block, color in zip(blocks, block_colors):
            row = block[row_idx]
            print(''.join(f'{color}██{reset}' if cell == 1 else '⋅⋅' for cell in row), end=' ')
        print()  # New line after each row
    print()  # Extra newline for spacing

def print_block_from_id(id):
    block = np.zeros((5, 5), dtype=int)
    for j in range(5):
        for i in range(5):
            if (id % 2 == 1):
                block[j, i] = 1
            id = id // 2
            
    # ANSI color codes for different colors
    colors = [
        '\033[91m',  # Red
        '\033[92m',  # Green
        '\033[93m',  # Yellow
        '\033[94m',  # Blue
        '\033[95m',  # Purple
    ]
    reset = '\033[0m'  # Reset color
    
    # Choose a random color for this block
    color = random.choice(colors)
    
    # Print the block with color
    for row in block:
        print(''.join(f'{color}██{reset}' if cell == 1 else '⋅⋅' for cell in row))
    print()  # Add an extra newline for better spacing

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
    print()  # Add an extra newline for better spacing

def play_move(position, blocks, permutation, placement):
    position_bitboard = board_to_bitboard(position)

    for i in range(len(permutation)):
        current_block_id = blocks[permutation[i]]
        
        current_block = block_id_to_block_info(current_block_id)

        position_bitboard = get_resulting_position(position_bitboard, current_block, placement[i])
        position_bitboard, cc = clear_position_info(position_bitboard)
    
    return bitboard_to_board(position_bitboard)

def get_best_move_simple(position, block_ids):
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

    for position, (permutation, placement) in positions.items():
        eval = random.random()
        if eval > max_eval:
            max_eval = eval
            best_permutation = permutation
            best_placement = placement
    
    return best_permutation, best_placement, max_eval

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
    # evals = np.log(1-raw_evals)/np.log(0.8)
    # Add random normal noise to each evaluation
    evals = raw_evals + np.random.normal(0, 1, size=raw_evals.shape)*temperature
    max_eval_index = np.argmax(evals)
    max_eval = evals[max_eval_index]

    best_permutation, best_placement = positions_values[max_eval_index]

    return best_permutation, best_placement, max_eval

def play_one_game_simple():
    position = np.zeros((8, 8))
    position_history = [position]

    while True:
        if board_to_bitboard(position).bit_count() < 35:
            block_list = random.choice(blocks_normal)
            # print("Picking normal")
            # print_blocks_from_ids(block_list)
        else:
            block_list = random.choice(blocks_difficult)
            # print("Picking difficult")
            # print_blocks_from_ids(block_list)
        random.shuffle(block_list)
        block_1, block_2, block_3 = block_list

        best_move = get_best_move_simple(position, [block_1, block_2, block_3])
        if best_move is None:
            break
        
        permutation, placement, eval = best_move
        position = play_move(position, [block_1, block_2, block_3], permutation, placement)

        position_history.append(position)
    
    return position_history

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

inputs = []
outputs = []
generation_start = 3
generation_end = 15
decay_rate = 0.8
name = "bb4-10k"

num_games = 10000
learning_rate_decay = 0.95
initial_temperature = 0.2
temperature_decay = 0.9

for generation in range(generation_start, generation_end):
    temperature = initial_temperature * temperature_decay ** (generation - 1)
    inputs = []
    outputs = []
    if generation == 0:
        for _ in tqdm(range(num_games)):
            pos_history = play_one_game_simple()
            pos_count = len(pos_history)

            for i, pos in enumerate(pos_history):
                eval = 1 - decay_rate ** (pos_count - (i + 1))
                # print_position(pos)
                # print(eval)
                inputs.append(pos)
                outputs.append(eval)
    else:
        model = tf.keras.models.load_model(f'{name}-gen{generation}.keras')

        for _ in tqdm(range(num_games)):
            pos_history = play_one_game(temperature)
            pos_count = len(pos_history)

            for i, pos in enumerate(pos_history):
                eval = 1 - decay_rate ** (pos_count - (i + 1))
                # print_position(pos)
                inputs.append(pos)
                outputs.append(eval)

    print(len(inputs), len(outputs))
    print("Average game length:", len(inputs)/num_games)
    # print("Output sample:", outputs)
    # Convert inputs and outputs to numpy arrays and reshape inputs
    X = np.array(inputs)
    y = np.array(outputs)

    # Reshape inputs to (samples, height, width, channels)
    X = X.reshape(-1, 8, 8, 1)
    
    # les train this model :D
    model = tf.keras.Sequential([
        # Input layer - shape is 8x8 with 1 channel
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(8, 8, 1), padding='same'),
        tf.keras.layers.BatchNormalization(),
        
        # Second conv layer with pooling to reduce dimensionality
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Flatten and dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()
    
    # Create optimizer with custom learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001*(learning_rate_decay**generation))

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )

    if generation > 0:
        model = tf.keras.models.load_model(f"{name}-gen{generation}.keras")

    # Create callbacks for early stopping and model checkpointing
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,  # Number of epochs with no improvement after which training will stop
        restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f'{name}-gen{generation+1}.keras',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=0
    )

    history = model.fit(
        X, y,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=[early_stopping, model_checkpoint]
    )
