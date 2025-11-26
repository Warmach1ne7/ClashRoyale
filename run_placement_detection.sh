#!/bin/bash
# Example workflow for creating troop placement dataset

# Your data structure:
# /home/ostikar/MyProjects/CS541/ClashRoyale/data/
#   ├── arena_01/
#   │   ├── game_01/images/
#   │   └── game_02/images/
#   ├── arena_02/
#   │   └── game_01/images/
#   ...

DATA_DIR="/home/ostikar/MyProjects/CS541/ClashRoyale/data"

echo "=========================================="
echo "CLASH ROYALE PLACEMENT DATASET CREATOR"
echo "=========================================="

# OPTION 1: Color-based detection (RECOMMENDED for blue clock)
# Fast, no templates needed, works great for blue circular clock
echo -e "\n[OPTION 1] Color-based detection (blue clock)"
python create_placement_dataset.py "$DATA_DIR" \
    --method color \
    --output troop_placements_color.csv \
    --filter

echo -e "\nResult: troop_placements_color.csv"
echo "Format: troop,x,y,arena,frame"

# OPTION 2: Template-based detection (if you have templates)
# More customizable, good if color detection has false positives
# echo -e "\n[OPTION 2] Template-based detection"
# python create_placement_dataset.py "$DATA_DIR" \
#     --method template \
#     --templates clock_templates \
#     --output troop_placements_template.csv \
#     --filter

# Process specific arenas only
# echo -e "\n[OPTION 3] Process specific arenas"
# python create_placement_dataset.py "$DATA_DIR" \
#     --method color \
#     --arenas arena_01 arena_02 arena_03 \
#     --output troop_placements_subset.csv

# Visualize detections on a sample frame
# python create_placement_dataset.py "$DATA_DIR" \
#     --method color \
#     --output troop_placements.csv \
#     --visualize sample_detection.png

echo -e "\n=========================================="
echo "DONE!"
echo "=========================================="
