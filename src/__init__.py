import logging
import os
import sys
from datetime import datetime

time_tag = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
time_tag = ""
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
log_dir = "artifacts/logs"
log_filepath = os.path.join(log_dir, f"running_logs{time_tag}.log")
os.makedirs(log_dir, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("Dev-Logger")
