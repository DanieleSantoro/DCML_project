import argparse
import time
import pandas as pd
from pynput import keyboard
from datetime import datetime
import os

events = []

def on_press(key):
    try:
        events.append({'event': 'keydown', 'key': key.char, 'timestamp': time.time()})
    except AttributeError:
        events.append({'event': 'keydown', 'key': str(key), 'timestamp': time.time()})

def on_release(key):
    try:
        events.append({'event': 'keyup', 'key': key.char, 'timestamp': time.time()})
    except AttributeError:
        events.append({'event': 'keyup', 'key': str(key), 'timestamp': time.time()})

def save_events(label, output_dir):
    df = pd.DataFrame(events)
    df['label'] = label
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/keyboard_events_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f"[INFO] Saved {len(events)} events to {filename}")

def main(label, duration, output_dir):
    print(f"[INFO] Recording keyboard events for {duration} seconds with label '{label}'...")
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    time.sleep(duration)
    listener.stop()
    save_events(label, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keyboard typing rhythm recorder")
    parser.add_argument('--label', choices=['human', 'bot'], required=True, help="Label for this recording")
    parser.add_argument('--duration', type=int, default=60, help="Duration in seconds")
    parser.add_argument('--output_dir', default='dataset', help="Directory to save recordings")
    args = parser.parse_args()
    main(args.label, args.duration, args.output_dir)
