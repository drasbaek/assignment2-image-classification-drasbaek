#!/usr/bin/env bash
# create virtual environment called img_classification
python3 -m venv img_classification_env

# activate virtual environment
source ./img_classification_env/bin/activate

# install requirements
echo "[INFO] Installing requirements..."
python3 -m pip install -r requirements.txt