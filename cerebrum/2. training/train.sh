#!/bin/bash

echo "NAME: train.sh"
echo "AUTHOR: David Carteau, France, November 2025"
echo "LICENSE: MIT (see 'license.txt' file content)"
echo "PURPOSE: Train neural network"

echo "Starting neural network training..."

python3 train.py

echo "Training completed!"
echo "Press Enter to continue..."
read -p "Press Enter to continue, or Ctrl+C to exit..."

# 以下是询问用户是否要关机的代码，已被注释掉
# 如果需要启用关机功能，请取消注释以下代码段
# read -p "Do you want to shutdown the system? (y/N): " choice
# if [[ "$choice" =~ ^[Yy]$ ]]; then
#     echo "Shutting down in 1 minute... Press Ctrl+C to cancel."
#     sudo shutdown -h +1
# else
#     echo "Shutdown cancelled."
# fi
