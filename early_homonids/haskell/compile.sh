#!/bin/bash

# Compilation script for phylogenetic analysis Haskell programs
# This script checks for GHC and compiles all modules with full type checking

set -e  # Exit on error

echo "========================================="
echo "  Phylogenetic Analysis Compilation"
echo "========================================="
echo ""

# Check for GHC
if ! command -v ghc &> /dev/null; then
    echo "ERROR: GHC (Glasgow Haskell Compiler) not found!"
    echo ""
    echo "Please install GHC using one of these methods:"
    echo ""
    echo "Ubuntu/Debian:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install ghc"
    echo ""
    echo "macOS (using Homebrew):"
    echo "  brew install ghc"
    echo ""
    echo "Using GHCup (recommended):"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh"
    echo ""
    exit 1
fi

# Display GHC version
echo "GHC version:"
ghc --version
echo ""

# Type check all modules
echo "Type checking modules..."
echo ""

echo "Checking PhyloTree.hs..."
ghc -Wall -fno-code PhyloTree.hs

echo "Checking PhyloAnalysis.hs..."
ghc -Wall -fno-code PhyloAnalysis.hs

echo "Checking TraitEvolution.hs..."
ghc -Wall -fno-code TraitEvolution.hs

echo "Checking PhyloDistance.hs..."
ghc -Wall -fno-code PhyloDistance.hs

echo "Checking Examples.hs..."
ghc -Wall -fno-code Examples.hs

echo ""
echo "All type checks passed! ✓"
echo ""

# Compile with optimizations
echo "Compiling with optimizations (-O2)..."
echo ""

ghc -Wall -O2 -o Examples Examples.hs

echo ""
echo "========================================="
echo "  Compilation successful!"
echo "========================================="
echo ""
echo "Run examples with: ./Examples"
echo "Or load in GHCi with: ghci Examples.hs"
echo ""
