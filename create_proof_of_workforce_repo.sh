#!/bin/bash

# Script to create the proof_of_workforce repository
# Run this from your home directory or wherever you want the repo

set -e

REPO_NAME="proof_of_workforce"
REPO_URL="https://github.com/danindiana/proof_of_workforce.git"

echo "Creating $REPO_NAME repository..."

# Create directory structure
mkdir -p "$REPO_NAME"/{docs,specs,examples}
cd "$REPO_NAME"

# Initialize git
git init
git config user.name "danindiana"
git config user.email "danindiana@users.noreply.github.com"
git branch -M main

# Get the files from the bundle
echo "Please place the proof_of_workforce_repo.bundle file in the same directory as this script"
echo "Then run: git clone proof_of_workforce_repo.bundle $REPO_NAME"
echo "cd $REPO_NAME"
echo "git remote set-url origin $REPO_URL"
echo "git push -u origin main"

exit 0
