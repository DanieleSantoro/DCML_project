import psutil
import numpy as np
import pandas as pd
from datetime import datetime, time

def collect_system_metrics(sample_interval=1, total_samples=1000):
    data = []
    for _ in range(total_samples):
        timestamp = datetime.now()
        cpu_percent = psutil.cpu_percent(interval=sample_interval)
        ram_percent = psutil.virtual_memory().percent
        gpu_temp = 60.0  # Mock (use `nvidia-smi` for real GPU data)
        data.append([timestamp, cpu_percent, ram_percent, gpu_temp])
        time.sleep(sample_interval)
    return pd.DataFrame(data, columns=["timestamp", "cpu", "ram", "gpu_temp"])

# Collect data (run this in background)
df = collect_system_metrics()
df.to_csv("pc_metrics.csv", index=False)
