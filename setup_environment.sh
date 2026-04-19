#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $ROOT_DIR

echo "==================================================="
echo " Starting Automated Setup for Artifact"
echo "==================================================="

echo "[1/5] Setting up Conda virtual environment and VTK..."
# module load conda
CONDA_ENV_DIR="./fofpz_conda_env"
if [ ! -d "$CONDA_ENV_DIR" ]; then
	echo "Creating local Conda environment in $CONDA_ENV_DIR..."
	conda create -p $CONDA_ENV_DIR python=3 vtk=9.4.1 -c conda-forge -y
else
	echo "Conda environment already exists in $CONDA_ENV_DIR. Skipping creation..."
fi

echo "[2/5] Building our method, Google Draco, and LCP..."
cd src/GPU
mkdir build && cd build
cmake ..
make -j
cd ../../CPU
mkdir build && cd build
cmake ..
make -j

cd ../../../third_party/draco
mkdir build && cd build
cmake ..
make -j

cd ../../LCP
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX="$(pwd)" ..
make lcp
make install
cd ../..

echo "[3/5] Downloading and building SZ3 v3.1.8..."
if [ ! -d "SZ3" ]; then
	git clone -c advice.detachedHead=false --branch v3.1.8 --depth 1 https://github.com/szcompressor/SZ3.git
	cd SZ3
	mkdir build && cd build
	cmake ..
	make -j
	cd ../..
else
	echo "SZ3 already exists. Skipping..."
fi

echo "[4/5] Downloading and building ZFP v1.0.1..."
if [ ! -d "zfp" ]; then
	git clone -c advice.detachedHead=false --branch 1.0.1 --depth 1 https://github.com/llnl/zfp.git
	cd zfp
	mkdir build && cd build
	cmake ..
	make -j
	cd ../..
else
	echo "ZFP already exists. Skipping..."
fi

echo "[5/5] Downloading and building cuSZp2..."
if [ ! -d "cuSZp" ]; then
	git clone https://github.com/szcompressor/cuSZp.git
	cd cuSZp
	mkdir build && cd build
	cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install/ ..
	make -j
	make install
	cd ../..
else
	echo "cuSZp2 already exists. Skipping..."
fi

echo "==================================================="
echo " Setup Complete! You are ready to run experiments."
echo "==================================================="
