import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config_files.tm_config import Config




config = Config()
print(config.time_step_duration)