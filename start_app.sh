#!/usr/bin/env bash

# 1. Move to the script's directory
cd "$(dirname "$0")"

# 2. Activate conda environment (replace with the correct path if needed)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate reditbot

# 3. Run the app in the background
python app.py &

# 4. Loading animation (10 seconds)
for i in {1..10}; do
    printf "."
    sleep 1
done
echo

# 5. Open the app in the default browser
open http://localhost:8080/