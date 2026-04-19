#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -A DATASETS
# DATASETS["HACC_hi"]=""
DATASETS["HACC_low"]="https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/EXASKY/HACC/EXASKY-HACC-data-medium-size.tar.gz"
DATASETS["EXAALT"]="https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/EXAALT/SDRBENCH-EXAALT-2869440.tar.gz"
DATASETS["FPM_high"]="https://cloud.sdsc.edu/v1/AUTH_sciviscontest/2016/smoothinglength_0.20/020_run19_1.tar.bz2"
DATASETS["FPM_mid"]="https://cloud.sdsc.edu/v1/AUTH_sciviscontest/2016/smoothinglength_0.30/030_run24.tar.bz2"
DATASETS["FPM_low"]="https://cloud.sdsc.edu/v1/AUTH_sciviscontest/2016/smoothinglength_0.44/044_run40.tar.bz2"

mkdir -p $ROOT_DIR/datasets

# Check argument counts
if [ $# -gt 1 ]; then
	echo "Error: Too many arguments."
	echo "Usage: bash download_data.sh [dataset_name]"
	exit 1
fi

# Function to handle download and extraction
download_data() {
	local key=$1
	local url=${DATASETS[$key]}

	if [ -z "$url" ]; then
		echo "Error: Dataset '$key' not found in configuration."
		return
	fi

	local OUTPUT_DIR=$ROOT_DIR/datasets/$key

	# Download
	echo "--- Downloading $key dataset ---"
	wget "$url" -P "$OUTPUT_DIR/"

	# Extraction
	echo "--- Extracting $key dataset ---"
	local filename=$(basename "$url")
	local zip_path="$OUTPUT_DIR/$filename"
	local strip_depth=$(tar -tf "$zip_path" | grep -v '/$' | head -1 | tr -cd '/' | wc -c)
	tar -xf "$zip_path" -C "$OUTPUT_DIR" --strip-components="$strip_depth"
	rm "$zip_path"

	# Remove / rename attributes
	if [ -f "$OUTPUT_DIR/xx.dat2" ]; then # EXAALT
		mv "$OUTPUT_DIR/xx.dat2" "$OUTPUT_DIR/xx.f32"
		mv "$OUTPUT_DIR/yy.dat2" "$OUTPUT_DIR/yy.f32"
		mv "$OUTPUT_DIR/zz.dat2" "$OUTPUT_DIR/zz.f32"
		find "$OUTPUT_DIR" -type f ! -name "xx*.f32" ! -name "yy*.f32" ! -name "zz*.f32" -delete
	elif [ -f "$OUTPUT_DIR/000.vtu" ]; then # FPM
		python3 $ROOT_DIR/scripts/analysis/vtu2bin.py --d $key
		find "$OUTPUT_DIR" -type f ! -name "xx*.f32" ! -name "yy*.f32" ! -name "zz*.f32" -delete
	elif [ -f "$OUTPUT_DIR/xx.f32" ]; then # HACC_low
		find "$OUTPUT_DIR" -type f ! -name "xx*.f32" ! -name "yy*.f32" ! -name "zz*.f32" -delete
	fi
}

if [ $# -eq 1 ]; then
	# Download specific dataset
	download_data "$1"
else
	# No arguments provided: download all
	echo "Downloading all datasets..."
	for key in "${!DATASETS[@]}"; do
		download_data "$key"
	done
fi
