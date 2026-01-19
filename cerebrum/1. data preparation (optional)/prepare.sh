#!/bin/bash

echo "NAME: prepare.sh"
echo "AUTHOR: David Carteau, France, November 2025"
echo "LICENSE: MIT (see 'license.txt' file content)"
echo "PURPOSE: Prepare training data"

echo "Starting training data preparation..."

python3 games_select.py

echo "[TIP] add --gamelimit 1024"
echo "[TIP] add '-Tda2024' if you want to only keep games played after 2024"
echo "[TIP] add '-Tdb2025' if you want to only keep games played before 2025"

echo "Note: pgn-extract needs to be installed on Linux"
echo "You can install it with: sudo apt-get install pgn-extract (on Debian/Ubuntu)"
echo "or download from: https://www.cs.kent.ac.uk/people/staff/djb/pgn-extract/"
./pgn-extract -s -Wepd games.pgn | python3 positions_split.py

python3 positions_merge.py
python3 positions_select.py
python3 positions_shuffle.py

echo "Preparation completed!"
echo "Press Enter to continue..."
read -p ""

