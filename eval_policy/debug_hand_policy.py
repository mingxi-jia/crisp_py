import os
import re
from pathlib import Path
from datetime import datetime

import numpy as np

def run():
    data_folder = "debug_data"
    data_file = find_latest_actions_file(data_folder)
    data_path = os.path.join(data_folder, data_file)
    
    pass

def find_latest_actions_file(debug_dir: str = "debug_data") -> Path:
    """
    Find the most recent actions_YYYYMMDD_HHMMSS.npy file in the debug directory.
    
    Args:
        debug_dir: Path to the directory containing action files
        
    Returns:
        Path object pointing to the latest actions file
        
    Raises:
        FileNotFoundError: If no actions files are found
    """
    debug_path = Path(debug_dir)
    
    # Pattern to match actions_YYYYMMDD_HHMMSS.npy
    pattern = re.compile(r'actions_(\d{8})_(\d{6})\.npy')
    
    latest_file = None
    latest_datetime = None
    
    for file in debug_path.iterdir():
        match = pattern.match(file.name)
        if match:
            date_str, time_str = match.groups()
            # Parse datetime from filename
            file_datetime = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            
            if latest_datetime is None or file_datetime > latest_datetime:
                latest_datetime = file_datetime
                latest_file = file
    
    if latest_file is None:
        raise FileNotFoundError(f"No actions_*.npy files found in {debug_dir}")
    
    return latest_file

if __name__ == "__main__":
    run()