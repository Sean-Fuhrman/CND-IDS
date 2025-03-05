import logging
import os
import time 
def sanitize_filename(filename):
    # Replace spaces with underscores
    filename = filename.replace(" ", "_")
    # Remove or replace invalid characters
    invalid_chars = ['[', ']', ',', ':', '/', '\\', '=', "'", '"']
    for char in invalid_chars:
        filename = filename.replace(char, '')
    return filename
def init_logs(experiment):
    #Create log file based on config
    if not os.path.exists("logs"):
        os.makedirs("logs")

    filename = time.strftime("%d-%m-%Y-%I-%M-%S")
    for key, value in experiment.items():
        if value is not None:
            if key == "metrics":
                continue
            filename += f"-{key}={value}"
            #check if filename is too long
            if len(filename) > 200:
                filename = filename[:200]
                break
    filename = sanitize_filename(filename)
    logging.basicConfig(filename=f"logs/{filename}.log", level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    #Set up logging to also print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    logging.info("Logging initialized with filename: %s", filename)
    