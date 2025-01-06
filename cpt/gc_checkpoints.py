import shutil
from pathlib import Path
import re
import time

def cleanup_checkpoints(checkpoint_dir: str, keep_latest: int = 5):
    checkpoints = []
    for directory in Path(checkpoint_dir).iterdir():
        match = re.match(r'epoch_(\d+)', directory.name)
        if match:
            checkpoints.append((directory, int(match.group(1))))
    
    checkpoints.sort(key=lambda x: x[1], reverse=True)
    
    for dir_path, epoch_num in checkpoints[keep_latest:]:
        if (epoch_num + 1) % 10 == 0:
            continue
        
        shutil.rmtree(dir_path)
        print(f"Deleted {dir_path}")


if __name__ == "__main__":
    path = "ft_models/llama3_1_70B/fulltext_conditioned_20epochs_lrcosine"
    while True:
        print("Running cleanup")
        cleanup_checkpoints(path, keep_latest=2)
        time.sleep(300)  # Sleep for 300 seconds (5 minutes)
