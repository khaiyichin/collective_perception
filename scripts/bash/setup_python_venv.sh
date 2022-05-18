#!/bin/bash

# Copy required python scripts to working directory
CURR_DIR=$(pwd)
echo "Changing to working directory: $1";
pushd $1

# Copy scripts
cp $CURR_DIR/../python/requirements.txt .

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install --upgrade pip && pip install -r requirements.txt

# Integrate graph-tool
pushd .venv/lib/python*/site-packages/
touch dist-packages.pth
echo "/usr/lib/python3/dist-packages" >> dist-packages.pth
popd

# Exit environment
deactivate