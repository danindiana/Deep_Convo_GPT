# Makefile for Phylogenetic Analysis in Haskell

# Compiler and flags
GHC = ghc
GHCFLAGS = -Wall -O2 -fspec-constr-keen -fno-warn-orphans
GHCI = ghci

# Source files
SOURCES = PhyloTree.hs PhyloAnalysis.hs TraitEvolution.hs PhyloDistance.hs Examples.hs

# Object files (generated during compilation)
OBJECTS = $(SOURCES:.hs=.o)
INTERFACES = $(SOURCES:.hs=.hi)

# Executable
EXECUTABLE = Examples

# Default target
.PHONY: all
all: $(EXECUTABLE)

# Compile the executable
$(EXECUTABLE): $(SOURCES)
	$(GHC) $(GHCFLAGS) -o $@ Examples.hs

# Compile individual modules
PhyloTree.o: PhyloTree.hs
	$(GHC) $(GHCFLAGS) -c PhyloTree.hs

PhyloAnalysis.o: PhyloAnalysis.hs PhyloTree.hs
	$(GHC) $(GHCFLAGS) -c PhyloAnalysis.hs

TraitEvolution.o: TraitEvolution.hs PhyloTree.hs PhyloAnalysis.hs
	$(GHC) $(GHCFLAGS) -c TraitEvolution.hs

PhyloDistance.o: PhyloDistance.hs PhyloTree.hs PhyloAnalysis.hs
	$(GHC) $(GHCFLAGS) -c PhyloDistance.hs

Examples.o: Examples.hs PhyloTree.hs PhyloAnalysis.hs TraitEvolution.hs PhyloDistance.hs
	$(GHC) $(GHCFLAGS) -c Examples.hs

# Run the examples
.PHONY: run
run: $(EXECUTABLE)
	./$(EXECUTABLE)

# Interactive GHCi session
.PHONY: repl
repl:
	$(GHCI) $(SOURCES)

# Type check all modules
.PHONY: check
check:
	$(GHC) $(GHCFLAGS) -fno-code $(SOURCES)

# Clean build artifacts
.PHONY: clean
clean:
	rm -f $(OBJECTS) $(INTERFACES) $(EXECUTABLE)
	rm -f *.hi *.o

# Clean everything including backup files
.PHONY: distclean
distclean: clean
	rm -f *~

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all        - Compile all modules and create executable (default)"
	@echo "  run        - Compile and run examples"
	@echo "  repl       - Start GHCi with all modules loaded"
	@echo "  check      - Type check all modules without generating code"
	@echo "  clean      - Remove compiled objects and executables"
	@echo "  distclean  - Remove all generated files including backups"
	@echo "  help       - Show this help message"

# Documentation generation (requires Haddock)
.PHONY: docs
docs:
	haddock --html --odir=docs $(SOURCES)

# Profiling build
.PHONY: prof
prof:
	$(GHC) $(GHCFLAGS) -prof -fprof-auto -rtsopts -o $(EXECUTABLE) Examples.hs

# Install dependencies (if using cabal)
.PHONY: install-deps
install-deps:
	cabal install --only-dependencies

# Quick compile without optimization (faster for development)
.PHONY: quick
quick:
	$(GHC) -Wall -o $(EXECUTABLE) Examples.hs

# Verbose compilation
.PHONY: verbose
verbose:
	$(GHC) $(GHCFLAGS) -v -o $(EXECUTABLE) Examples.hs
