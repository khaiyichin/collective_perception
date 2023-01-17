#!/bin/bash

# Install static simulator
pushd collective_perception_static
../scripts/bash/setup_python_venv.sh . requirements.txt # setup virtual environment that can find apt installed packages
source .venv/bin/activate
pip install .
deactivate
popd

# Install dynamic simulator
pushd collective_perception_dynamic
mkdir build
cd build
cmake ..
make -j$(nproc)
popd
