#!/bin/bash

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[1;34m'
NC='\033[0m' # No Color

clear
echo -e "${BLUE}===================================================="
echo -e "             STREAMLIT APP LAUNCHER"
echo -e "====================================================${NC}\n"

echo -e "${YELLOW}[1/3] Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python3 not found! Please install Python first.${NC}"
    exit 1
fi
python3 --version
echo

echo -e "${YELLOW}[2/3] Installing dependencies...${NC}"
python3 -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${RED}Error installing dependencies!${NC}"
    exit 1
fi
echo

echo -e "${YELLOW}[3/3] Starting Streamlit app...${NC}"
echo -e "${BLUE}----------------------------------------------------${NC}"
streamlit run app.py
echo -e "${BLUE}----------------------------------------------------${NC}"

echo -e "${GREEN}App closed. Have a nice day!${NC}"