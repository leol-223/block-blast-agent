from PIL import Image
import cv2
import numpy as np
import pyautogui
import time
import random
from pynput import keyboard
import math
import pickle
import tensorflow as tf
import time
import math

from board import board_to_bitboard, bitboard_to_board, generate_positions_clear, get_bitboard_sum, block_id_to_block_info, get_resulting_position, clear_position

positions_evaluated = 0
blocks_encountered = []
save_data = False

# Open the file in binary read mode ('rb')
with open('blocks_encountered_new.pkl', 'rb') as f:
    blocks_encountered = pickle.load(f)

block_list_all = set()
for block_board, block_triple in blocks_encountered:
    for block in block_triple:
        block_list_all.add(block)

model = tf.keras.models.load_model('bb4-10k-gen9.keras')
transposition_table = {}

def get_block_counts(image, points, x_offset, shift_x=False, shift_y=False):
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Define block size
    block_size = 14
    half_block = block_size // 2
    
    # Apply shifts if flags are set
    x_offset_initial = half_block if shift_x else 0
    y_offset_initial = half_block if shift_y else 0
    
    # Calculate number of blocks in each dimension
    # Adjust width/height to account for shifts if needed
    adjusted_width = width - x_offset_initial if shift_x else width
    adjusted_height = height - y_offset_initial if shift_y else height
    num_blocks_x = (adjusted_width + block_size - 1) // block_size
    num_blocks_y = (adjusted_height + block_size - 1) // block_size
    
    # Initialize array to store point counts for each block
    block_counts = np.zeros((num_blocks_y, num_blocks_x), dtype=int)
    
    # Count points in each block
    for point in points:
        # Add half block since the point is the top right
        x, y = point[0] - x_offset + half_block, point[1] + half_block
        # Apply the shift offsets to x and y before calculating block position
        x -= x_offset_initial
        y -= y_offset_initial
        block_x = x // block_size
        block_y = y // block_size
        if (0 <= block_y < num_blocks_y and 0 <= block_x < num_blocks_x):
            block_counts[block_y, block_x] += 1
    
    return block_counts

def block_counts_to_block_id(block_counts):
    leftmost_x = 100
    top_y = 100
    for i in range(block_counts.shape[0]):
        for j in range(block_counts.shape[1]):
            if block_counts[j, i] > 0:
                leftmost_x = min(leftmost_x, i)
                top_y = min(top_y, j)

    true_blocks = np.zeros(block_counts.shape)
    for i in range(block_counts.shape[0]-leftmost_x):
        for j in range(block_counts.shape[1]-top_y):
            if block_counts[j+top_y, i+leftmost_x] > 0:
                true_blocks[j, i] = 1
    id = 0
    # in binary
    for j in range(block_counts.shape[1]-1, -1, -1):
        for i in range(block_counts.shape[0]-1, -1, -1):
            id *= 2
            if true_blocks[j, i] > 0:
                id += 1
    return id

def get_best_move(position, block_ids, temperature=0):
    position_bitboard = board_to_bitboard(position)
    blocks = []
    for i, block_id in enumerate(block_ids):
        blocks.append([i, block_id_to_block_info(block_id)])

    positions_clear = {}
    positions_nonclear = {}
    generate_positions_clear(position_bitboard, blocks, positions_clear, positions_nonclear)

    if len(positions_clear) + len(positions_nonclear) == 0:
        return None
    
    positions = {}
    if len(positions_clear) > 0:
        positions = positions_clear
    else:
        print("Could not find a clearing arrangement.")
        positions = positions_nonclear
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

def get_current_game():
    display_screen_region(688, 276, 934-688, 522-276, "temp_image_game")
    # BGR cause cv2 is weird
    empty = (66, 35, 32)
    img = cv2.imread("temp_image_game.jpg")
    height, width = img.shape[:2]
    increment = height / 8
    start = increment / 2
    result_arr = np.zeros((8, 8))

    for i in range(8):
        for j in range(8):
            x = round(start + increment * i)
            y = round(start + increment * j)
            pixel = img[y, x].astype(np.int32)  # Convert to int32
            img[y, x] = (0, 0, 255) 
            sqr_dif = ((pixel[0] - empty[0]) ** 2) + ((pixel[1] - empty[1]) ** 2) + ((pixel[2] - empty[2]) ** 2)
            if sqr_dif > 20:
                result_arr[j, i] = 1
    cv2.imwrite("a_image.jpg", img)
    return result_arr

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

def move_left_block(i, j, w, h, dur):
    # move center to this position
    x_index = i + (w - 1) / 2
    y_index = j + (h - 1) / 2
    
    x = 709 + (875 - 709) * x_index / 7
    y = 417 + (583 - 417) * y_index / 7

    pyautogui.moveTo(695+14*2.5, 559+14*2.5)
    dist = math.sqrt((695+14*2.5-x)**2+(559+14*2.5-y)**2)
    pyautogui.dragTo(x, y, dur, button='left')  # drag mouse to X of 300, Y of 400 over 2 seconds while holding down left mouse button

def move_center_block(i, j, w, h, dur):
    # move center to this position
    x_index = i + (w - 1) / 2
    y_index = j + (h - 1) / 2

    x = 727 + (893 - 727) * x_index / 7
    y = 416 + (583 - 416) * y_index / 7

    pyautogui.moveTo(695+81+14*2.5, 559+14*2.5)
    dist = math.sqrt((695+81+14*2.5-x)**2+(559+14*2.5-y)**2)
    pyautogui.dragTo(x, y, dur, button='left')  # drag mouse to X of 300, Y of 400 over 2 seconds while holding down left mouse button

def move_right_block(i, j, w, h, dur):
    # move center to this position
    x_index = i + (w - 1) / 2
    y_index = j + (h - 1) / 2

    x = 747 + (912 - 747) * x_index / 7
    y = 419 + (582 - 419) * y_index / 7

    pyautogui.moveTo(695+162+14*2.5, 559+14*2.5)
    dist = math.sqrt((695+162+14*2.5-x)**2+(559+14*2.5-y)**2)
    pyautogui.dragTo(x, y, dur, button='left')  # drag mouse to X of 300, Y of 400 over 2 seconds while holding down left mouse button

def move_block(block_list, move_index, block_index, move, ws, hs, safety_pause=0.05, move_duration=0.25):
    if block_index == 0:
        move_left_block(move[0], move[1], ws[0], hs[0], move_duration)
        time.sleep(safety_pause)
        if move_index == 2:
            time.sleep(0.2)
        id_left, id_m, id_r = get_block_counts_screen()
        if move_index == 2 and not (id_m == 0 or id_r == 0):
            return True
        if (id_left == block_list[0]):
            return False
    elif block_index == 1:
        move_center_block(move[0], move[1], ws[1], hs[1], move_duration)
        time.sleep(safety_pause)
        if move_index == 2:
            time.sleep(0.2)
        id_l, id_middle, id_r = get_block_counts_screen()
        if move_index == 2 and not (id_l == 0 or id_r == 0):
            return True
        # new blocks appearing
        if (id_middle == block_list[1]):
            return False
    else:
        move_right_block(move[0], move[1], ws[2], hs[2], move_duration)
        time.sleep(safety_pause)
        if move_index == 2:
            time.sleep(0.2)
        id_l, id_m, id_right = get_block_counts_screen()
        if move_index == 2 and not (id_l == 0 or id_m == 0):
            return True
        # new blocks appearing
        if (id_right == block_list[2]):
            return False
    
    return True

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
        position_bitboard = clear_position(position_bitboard)
    
    return bitboard_to_board(position_bitboard)

def get_block_counts_new(offset=0):
    display_screen_region(695+82*offset, 559, 70, 70, f"temp_image_{offset}")
    img = cv2.imread(f"temp_image_{offset}.jpg")
    
    background_bgr = np.array([128, 74, 57])
    diff = np.sum((img.astype(np.int32) - background_bgr) ** 2, axis=2)
    mask = diff > 40 * 40

    # Find the coordinates where mask is True
    # X - 13 (odd - 3), 21 (even - 2), 19 (even - 2)
    # Y - 19 (even - 2), 21 (even - 2), 14 (odd - 3), 13 (odd - 3)
    non_bg = np.where(mask)
    if len(non_bg[0]) > 0:  # Check if any True values exist
        leftmost_x = np.min(non_bg[1])  # x coordinates are in the second array
        top_y = np.min(non_bg[0])       # y coordinates are in the first array
        mask_image = mask.astype(np.uint8) * 255
        
        x_shift = leftmost_x % 14 in [3, 4, 5, 6, 7, 8, 9]
        y_shift = top_y % 14 in [3, 4, 5, 6, 7, 8, 9]

        start_x = 0 if x_shift else 7
        start_y = 0 if y_shift else 7

        block_counts = np.zeros((5, 5))

        for j in range(5):
            for i in range(5):
                y = start_y + 14 * j
                x = start_x + 14 * i
                if mask[y, x]:
                    img[y, x] = [0, 0, 255]
                    block_counts[j, i] = 1
                mask_image[y, x] = 128

        cv2.imwrite(f"matched_points_{offset}.jpg", img)
        # Convert boolean mask to uint8 format (False->0, True->255)
        
        cv2.imwrite(f"mask_{offset}.jpg", mask_image)

        return block_counts
    else:
        return np.zeros((5, 5))


def get_block_counts_screen():
    block_left, block_middle, block_right = get_block_counts_new(0), get_block_counts_new(1), get_block_counts_new(2)

    id_left = block_counts_to_block_id(block_left)
    id_middle = block_counts_to_block_id(block_middle)
    id_right = block_counts_to_block_id(block_right)

    return id_left, id_middle, id_right
    

def act_immediately():
    global last_id_left
    global last_id_middle
    global last_id_right
    global expected_pos
    global last_cost
    global positions_evaluated
    global is_first
    print("starting to act")
    id_counts = {}
    positions_evaluated = 0
    
    id_left, id_middle, id_right = get_block_counts_screen()
    if (id_left == 0 or id_left == 1):
        print("exit 1")
        return
    if (id_middle == 0 or id_middle == 1):
        print("exit 2")
        return
    if (id_right == 0 or id_right == 1):
        print("exit 3")
        return
    if (id_left == last_id_left and id_middle == last_id_middle and id_right == last_id_right):
        print("exit 4")
        return
    
    last_id_left = id_left
    last_id_middle = id_middle
    last_id_right = id_right

    if is_first:
        position = get_current_game()
    else:
        position = expected_pos
    print_position(position)
    if expected_pos is not None:
        if not np.array_equal(position, expected_pos):
            print("EXPECTED")
            print_position(expected_pos)
            raise ValueError("Position does not match expected position")

    print_block_from_id(id_left)
    if id_left not in block_list_all:
        print("New block encountered (left)")
        pass#print(1/0)
    print_block_from_id(id_middle)
    if id_middle not in block_list_all:
        print("New block encountered (middle)")
        pass#print(1/0)
    print_block_from_id(id_right)
    if id_right not in block_list_all:
        print("New block encountered (right)")
        pass#print(1/0)
    if save_data:
        blocks_encountered.append([board_to_bitboard(position), [id_left, id_middle, id_right]])

    _, left_w, left_h = block_id_to_block_info(id_left)
    _, middle_w, middle_h = block_id_to_block_info(id_middle)
    _, right_w, right_h = block_id_to_block_info(id_right)

    best_move = get_best_move(position, [id_left, id_middle, id_right])
    if (best_move is None):
        print("No continuation found.")
        return

    permutation, placement, eval = best_move
    print(permutation, placement)
    
    # Color code the eval based on thresholds
    if eval < 3:
        color = '\033[91m'  # Red
    elif eval < 8:
        color = '\033[93m'  # Yellow
    else:
        color = '\033[92m'  # Green
    reset = '\033[0m'  # Reset color
    print(f"{color}Eval: {float(eval.item()):.2f}{reset}")
    block_list = [id_left, id_middle, id_right]
    for i in range(3):
        current_block = permutation[i]
        was_moved = move_block(block_list, i, current_block, placement[i], [left_w, middle_w, right_w], [left_h, middle_h, right_h])
        safety_limit = 5
        start = 1
        while not was_moved and start < safety_limit:
            print("Trying to move block again.")
            time.sleep(0.1)
            was_moved = move_block(block_list, i, current_block, placement[i], [left_w, middle_w, right_w], [left_h, middle_h, right_h])
            start += 1
        if start >= safety_limit:
            raise ValueError("Block was not moved properly.")

    expected_pos = play_move(position, [id_left, id_middle, id_right], permutation, placement)


def get_screen_region(x, y, width, height):
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    pixel_data = list(screenshot.getdata())
    return pixel_data

def display_screen_region(x, y, width, height, name="screenshot"):
    # Capture screenshot of specified region
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    # Convert RGBA to RGB
    rgb_screenshot = screenshot.convert('RGB')
    # Save the image as JPG
    rgb_screenshot.save(name+".jpg")

last_id_left = 0
last_id_middle = 0
last_id_right = 0
last_cost = 0

# Add this near the other global variables
running = True
expected_pos = None
is_first = True

def on_press(key):
    global running
    try:
        if key == keyboard.Key.space:
            print("Stopping program...")
            running = False
    except AttributeError:
        pass

# Add this before the main loop
listener = keyboard.Listener(on_press=on_press)
listener.start()

time.sleep(2)
while running:  # Changed from while True
    transposition_table = {}
    act_immediately()
    is_first = False
    # make sure there's no "great!" or other annoying artifacts in the way
    time.sleep(0.65)

with open('blocks_encountered_new.pkl', 'wb') as f:
    pickle.dump(blocks_encountered, f)

listener.stop()  # Clean up the listener when done