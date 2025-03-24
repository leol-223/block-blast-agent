from pynput import keyboard
import pyautogui
import time

def on_press(key): 
    try:
        if key == keyboard.Key.space:
            print(pyautogui.position())
    except AttributeError:
        pass

# Create and start the keyboard listener
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()

    