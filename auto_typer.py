import time
import pyautogui
import argparse

def auto_type(text, delay):
    print(f"[INFO] Starting auto typer. Typing every {delay} seconds...")
    try:
        while True:
            for char in text:
                pyautogui.press(char)
                time.sleep(delay)
    except KeyboardInterrupt:
        print("\n[INFO] Auto typer stopped by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto Typer Simulator")
    parser.add_argument('--text', type=str, default="hello world ", help="Text to type repeatedly")
    parser.add_argument('--delay', type=float, default=0.1, help="Delay between key presses in seconds")
    args = parser.parse_args()
    auto_type(args.text, args.delay)
