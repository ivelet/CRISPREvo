#!/bin/bash
# Script to download all necessary data files for the CRISPREvo project

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Dataset configuration
ZENODO_URL="https://zenodo.org/records/18323772/files/data.zip?download=1"
OUTPUT_FILE="data.zip"

# Fine-tuned model configuration
MODEL_URL="https://zenodo.org/records/18328270/files/crispr_evo.zip?download=1"
MODEL_OUTPUT_FILE="crispr_evo.zip"

echo -e "${GREEN}Starting downloads...${NC}"

# Check if wget is installed
if ! command -v wget &> /dev/null; then
    echo -e "${RED}Error: wget is not installed. Install it with: sudo apt-get install wget${NC}"
    exit 1
fi

# Check if unzip is installed
if ! command -v unzip &> /dev/null; then
    echo -e "${RED}Error: unzip is not installed. Install it with: sudo apt-get install unzip${NC}"
    exit 1
fi

# Download the dataset
# echo -e "${YELLOW}Downloading dataset from Zenodo...${NC}"
# wget -O "$OUTPUT_FILE" "$ZENODO_URL"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Dataset download failed${NC}"
    exit 1
fi
echo -e "${GREEN}Dataset download complete!${NC}"

# Download the model
echo -e "${YELLOW}Downloading model from Zenodo...${NC}"
wget -O "$MODEL_OUTPUT_FILE" "$MODEL_URL"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Model download failed${NC}"
    exit 1
fi
echo -e "${GREEN}Model download complete!${NC}"

# Extract the dataset
echo -e "${YELLOW}Extracting dataset...${NC}"
unzip -o "$OUTPUT_FILE"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Dataset extraction failed${NC}"
    exit 1
fi
echo -e "${GREEN}Dataset extraction complete!${NC}"

# Extract the model
echo -e "${YELLOW}Extracting model...${NC}"
unzip -o "$MODEL_OUTPUT_FILE"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Model extraction failed${NC}"
    exit 1
fi
echo -e "${GREEN}Model extraction complete!${NC}"

# Cleanup
read -p "Remove downloaded zip files? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm "$OUTPUT_FILE" "$MODEL_OUTPUT_FILE"
    echo -e "${GREEN}Zip files removed${NC}"
fi

echo -e "${GREEN}All data setup complete!${NC}"