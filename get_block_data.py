import pyautogui
from PIL import Image
import time

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

iteration = 56
time.sleep(3)
while True:
    time.sleep(1)
    iteration += 1
    # padding - 6
    display_screen_region(695, 559, 232, 70, "screenshot"+str(iteration))
