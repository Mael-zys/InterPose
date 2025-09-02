import os
import logging
import sys

class DualOutput:
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def episode_logs(save_folder, scene_name, seq_name, pidx):
    save_motion_folder = os.path.join(save_folder, f"{scene_name}_{seq_name}_{pidx}")
    os.makedirs(save_motion_folder, exist_ok=True)

    # Define paths for the logs
    episode_log = os.path.join(save_motion_folder, "output.log")

    # Ensure unique logger name
    logger_name = f"{scene_name}_{seq_name}_{pidx}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Clear previous handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a file handler
    file_handler = logging.FileHandler(episode_log, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Set format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Attach handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Backup original stdout
    original_stdout = sys.stdout

    # Redirect stdout
    log_file = open(episode_log, 'w', encoding='utf-8')
    sys.stdout = DualOutput(original_stdout, log_file)

    logger.info(f"Started logging for task: {scene_name} {seq_name} {pidx}")

    # Return logger, log path, and a cleanup function
    def cleanup():
        sys.stdout = original_stdout
        log_file.close()
        logger.info(f"Finished logging for task: {scene_name} {seq_name} {pidx}")

    return logger, episode_log, cleanup
