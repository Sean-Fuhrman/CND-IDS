#!/bin/bash

VOLUME_DIR=/home/jovyan/sean-vol

# Copy current folder to root
echo "Copying current folder to root"
mkdir -p /root/running_folder

# Define the list of files/folders to exclude
EXCLUDE_LIST=("data/","logs/", "results/", ".git", ".ipynb_checkpoints", ".vscode","copy_then_run.sh")

# Build the rsync exclude parameters
EXCLUDE_PARAMS=()
for item in "${EXCLUDE_LIST[@]}"; do
    EXCLUDE_PARAMS+=("--exclude=$item")
done
rsync -av "${EXCLUDE_PARAMS[@]}" "/home/jovyan/sean-vol/" "/root/running_folder/"


# Run the script
echo "Running the script"
cd /root/running_folder/sean-vol
python run_experiments.py

# Copy the results back
echo "Copying the results back"
rsync -av "/root/running_folder/sean-vol/results/" "/home/jovyan/sean-vol/results/"

# Copy the logs back
echo "Copying the logs back"
rsync -av "/root/running_folder/sean-vol/logs/" "/home/jovyan/sean-vol/logs/"

# Clean up
echo "Cleaning up"
rm -rf /root/running_folder




echo "Done"