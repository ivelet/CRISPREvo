#!/bin/bash
# Script to download all necessary data files for the CRISPREvo project
ZENODO_URL="https://zenodo.org/records/18323772/files/data.zip?download=1"
OUTPUT_FILE="data.zip"
EXTRACT_DIR="data"

echo -e "${GREEN}Starting data download...${NC}"

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

# Download the file
echo -e "${YELLOW}Downloading data from Zenodo...${NC}"
wget -O "$OUTPUT_FILE" "$ZENODO_URL"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Download failed${NC}"
    exit 1
fi

echo -e "${GREEN}Download complete!${NC}"

# Extract the archive
echo -e "${YELLOW}Extracting data...${NC}"
unzip -o "$OUTPUT_FILE"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Extraction failed${NC}"
    exit 1
fi

echo -e "${GREEN}Extraction complete!${NC}"

# Remove the zip file after extraction
read -p "Remove the zip file? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm "$OUTPUT_FILE"
    echo -e "${GREEN}Zip file removed${NC}"
fi

echo -e "${GREEN}Data setup complete!${NC}"