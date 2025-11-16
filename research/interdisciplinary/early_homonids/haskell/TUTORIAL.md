# Phylogenetic Analysis Tutorial

A step-by-step guide to using the type-safe phylogenetic analysis system.

## Getting Started

### 1. Load the Modules in GHCi

```bash
cd early_homonids/haskell
ghci PhyloTree.hs
```

In GHCi:
```haskell
:load Examples
```

### 2. Build the Phylogenetic Tree

```haskell
let tree = buildHominidTree
```

This creates a complete phylogenetic tree of early hominids with 21 species.

## Basic Queries

### Finding Species

```haskell
-- Find a specific species by name
let maybeSpecies = findSpecies "Homo sapiens" tree

-- Extract from Maybe
case maybeSpecies of
    Just species -> do
        putStrLn $ speciesName species
        putStrLn $ "Genus: " ++ show (genus species)
        putStrLn $ "Location: " ++ location species
        print (traits species)
    Nothing -> putStrLn "Not found"
```

### Finding by Genus

```haskell
-- Find all Homo species
let homoSpecies = findByGenus Homo tree

-- Display them
mapM_ (putStrLn . speciesName) homoSpecies
```

Output:
```
Homo habilis
Homo rudolfensis
Homo ergaster
Homo erectus
Homo heidelbergensis
Homo neanderthalensis
Homo sapiens
```

### Tree Statistics

```haskell
-- Tree depth (longest path from root to leaf)
treeDepth tree

-- Total number of species
treeSize tree

-- Number of species (same as treeSize)
length (dfs tree)

-- Get only leaf species (terminal taxa)
length (getLeaves tree)
```

## Traversals

### Depth-First Search

```haskell
-- Get all species in depth-first order
let species = dfs tree
mapM_ (putStrLn . speciesName) species
```

### Breadth-First Search

```haskell
-- Get species by evolutionary level
let levels = bfs tree
mapM_ print levels
```

### Preorder and Postorder

```haskell
-- Preorder: parent before children
let preorder = preorder tree

-- Postorder: children before parent
let postorder = postorder tree
```

## Phylogenetic Analysis

### Finding Evolutionary Paths

```haskell
-- Find path between two species
let Just afarensis = findSpecies "Australopithecus afarensis" tree
let Just sapiens = findSpecies "Homo sapiens" tree

case findPath afarensis sapiens tree of
    Just path -> do
        putStrLn "Evolutionary path:"
        mapM_ (putStrLn . ("  -> " ++) . speciesName) path
    Nothing -> putStrLn "No path found"
```

### Most Recent Common Ancestor (MRCA)

```haskell
let Just habilis = findSpecies "Homo habilis" tree
let Just sapiens = findSpecies "Homo sapiens" tree

case mostRecentCommonAncestor habilis sapiens tree of
    Just mrca -> putStrLn $ "MRCA: " ++ speciesName mrca
    Nothing -> putStrLn "No common ancestor"
```

### Divergence Time

```haskell
case divergenceTime habilis sapiens tree of
    Just (MYA time) -> putStrLn $ "Diverged " ++ show time ++ " million years ago"
    Nothing -> putStrLn "Cannot calculate"
```

### Lineage

```haskell
-- Get complete lineage from root to species
case lineage sapiens tree of
    Just lin -> do
        putStrLn "Complete lineage:"
        mapM_ (putStrLn . ("  -> " ++) . speciesName) lin
    Nothing -> putStrLn "Lineage not found"
```

## Trait Evolution

### Defining Trait Predicates

```haskell
-- Define helpers for cranial capacity
let isCranialCapacity (CranialCapacity _) = True
    isCranialCapacity _ = False

let extractCC (CranialCapacity cc) = cc
    extractCC _ = 0

-- Define helpers for bipedalism
let isBipedalism (BipedalismDegree _) = True
    isBipedalism _ = False

let extractBD (BipedalismDegree bd) = bd
    extractBD _ = 0
```

### Trait Timeline

```haskell
-- Get cranial capacity over time
let ccTimeline = traitTimeline isCranialCapacity extractCC tree

-- Display first 10 entries
mapM_ (\(MYA time, cc) ->
    putStrLn $ show time ++ " MYA: " ++ show cc ++ " cc")
    (take 10 ccTimeline)
```

### Detecting Evolutionary Trends

```haskell
-- Detect trend in cranial capacity
let trend = detectTrend isCranialCapacity extractCC tree
putStrLn $ "Trend: " ++ show trend
-- Output: Increasing
```

### Trait Statistics

```haskell
-- Calculate statistics for cranial capacity
let stats = traitStatistics isCranialCapacity extractCC tree

putStrLn $ "Count: " ++ show (count stats)
putStrLn $ "Mean: " ++ show (mean stats)
putStrLn $ "Min: " ++ show (minVal stats)
putStrLn $ "Max: " ++ show (maxVal stats)
putStrLn $ "Std Dev: " ++ show (stdDev stats)
```

### Evolutionary Rates

```haskell
let Just erectus = findSpecies "Homo erectus" tree
let Just sapiens = findSpecies "Homo sapiens" tree

-- Calculate evolutionary rate (change per million years)
let rate = evolutionaryRate isCranialCapacity extractCC erectus sapiens
putStrLn $ "Evolutionary rate: " ++ show rate ++ " cc per MY"
```

## Distance Calculations

### Patristic Distance

```haskell
-- Sum of branch lengths along path
let Just habilis = findSpecies "Homo habilis" tree
let Just sapiens = findSpecies "Homo sapiens" tree

case patristicDistance habilis sapiens tree of
    Just dist -> putStrLn $ "Patristic distance: " ++ show dist ++ " MY"
    Nothing -> putStrLn "Cannot calculate"
```

### Morphological Distance

```haskell
-- Distance based on trait differences
let morphDist = morphologicalDistance habilis sapiens
putStrLn $ "Morphological distance: " ++ show morphDist
```

### Temporal Distance

```haskell
-- Time difference
let tempDist = temporalDistance habilis sapiens
putStrLn $ "Temporal distance: " ++ show tempDist ++ " MY"
```

### Distance Matrices

```haskell
-- Build patristic distance matrix
let matrix = buildPatristicMatrix tree

-- Look up distance between two species
let dist = matrixLookup "Homo habilis" "Homo sapiens" matrix
putStrLn $ "Distance: " ++ show dist

-- Convert to list
let distList = matrixToList matrix
putStrLn $ "Matrix entries: " ++ show (length distList)
```

## Selection Analysis

### Directional Selection

```haskell
-- Analyze directional selection on cranial capacity
let selection = directionalSelection isCranialCapacity extractCC tree

putStrLn $ "Selection type: " ++ selectionType selection
putStrLn $ "Trend: " ++ show (trend selection)
putStrLn $ "Rate: " ++ show (selectionRate selection)
putStrLn $ "Strength: " ++ show (strength selection)
```

### Stabilizing Selection

```haskell
-- Analyze stabilizing selection
let selection = stabilizingSelection isBipedalism extractBD tree
putStrLn $ "Strength: " ++ show (strength selection)
```

## Adaptive Radiation

```haskell
-- Check for adaptive radiation in Homo genus
let radiation = adaptiveRadiation Homo tree

putStrLn $ "Species involved: " ++ show (speciesInvolved radiation)
putStrLn $ "Trait diversification: " ++ show (traitDiversification radiation)
putStrLn $ "Duration: " ++ show (temporalDuration radiation) ++ " MY"
putStrLn $ "Is radiation: " ++ show (isRadiation radiation)
```

## Statistical Analysis

### Genus Distribution

```haskell
-- Count species by genus
import qualified Data.Map.Strict as Map
let dist = genusDistribution tree
mapM_ print (Map.toList dist)
```

### Temporal Distribution

```haskell
-- Count species by time period
let tempDist = temporalDistribution tree
mapM_ (\(period, count) ->
    putStrLn $ show period ++ " MYA: " ++ show count ++ " species")
    (Map.toList tempDist)
```

## Comparative Analysis

### Compare Two Species

```haskell
let Just erectus = findSpecies "Homo erectus" tree
let Just neanderthal = findSpecies "Homo neanderthalensis" tree

let comparisons = compareSpecies erectus neanderthal
mapM_ (\(trait, v1, v2) -> do
    putStrLn $ trait ++ ":"
    putStrLn $ "  H. erectus: " ++ show v1
    putStrLn $ "  H. neanderthalensis: " ++ show v2
    putStrLn $ "  Difference: " ++ show (abs (v1 - v2)))
    comparisons
```

### Find Similar Species

```haskell
let Just erectus = findSpecies "Homo erectus" tree

-- Find species with similarity > 0.7
let similar = findSimilar erectus tree 0.7
putStrLn "Similar species:"
mapM_ (putStrLn . speciesName) similar
```

## Evolution Models

### Gradual Change

```haskell
let model = GradualChange 50.0  -- 50 cc per MY

-- Apply to lineage
let Just homoLineage = lineage sapiens tree
let evolved = evolveTraits model (filter ((== Homo) . genus) homoLineage)

mapM_ (\(sp, value) ->
    putStrLn $ speciesName sp ++ ": " ++ show value)
    evolved
```

### Punctuated Equilibrium

```haskell
let model = PunctuatedEquilibrium 100.0 5.0  -- rapid then slow

let evolved = evolveTraits model homoLineage
```

## Running Complete Examples

### Run All Examples

```haskell
:load Examples
main
```

### Run Individual Examples

```haskell
example1_BasicQueries
example2_PhylogeneticAnalysis
example3_TraitEvolution
example4_Distances
example5_SelectionAnalysis
```

## Working with Custom Species

### Creating a New Species

```haskell
let newSpecies = Species
    { speciesName = "Homo hypotheticus"
    , genus = Homo
    , timeRange = (MYA 0.5, MYA 0.1)
    , traits =
        [ CranialCapacity 1600
        , BipedalismDegree 1.0
        , ToolUse True
        , BodyMass 75
        ]
    , location = "Worldwide"
    , confidence = 0.5
    }
```

### Building Custom Trees

```haskell
-- Create a simple tree
let simpleTree = node parent
    [ leaf child1
    , leaf child2
    ]
```

## Type Safety Examples

### Compile-Time Errors

These will fail to compile:

```haskell
-- Wrong type for TimeScale
let bad1 = MYA "string"  -- ERROR: expects Double

-- Incomplete pattern matching
let extract (CranialCapacity cc) = cc  -- WARNING: non-exhaustive

-- Invalid tree structure
let bad2 = Node parent []  -- Creates Leaf automatically
```

### Type-Safe Guarantees

```haskell
-- Compiler ensures:
-- 1. All pattern matches are exhaustive
-- 2. No null pointer errors (Maybe types)
-- 3. Consistent trait types
-- 4. Valid tree structures

-- Validation at runtime
isValid tree  -- True if tree is valid
```

## Performance Tips

### Lazy Evaluation

```haskell
-- Only compute what you need
let allPaths = findAllPaths sp1 sp2 tree
let shortestOnly = take 1 $ sortBy (comparing length) allPaths
```

### Memoization

```haskell
-- Cache expensive computations
let matrix = buildPatristicMatrix tree  -- Build once
let d1 = matrixLookup "sp1" "sp2" matrix  -- Fast lookup
let d2 = matrixLookup "sp3" "sp4" matrix  -- Fast lookup
```

## Troubleshooting

### Species Not Found

```haskell
-- Always check Maybe results
case findSpecies "Unknown" tree of
    Just sp -> -- work with species
    Nothing -> putStrLn "Species not found in tree"
```

### Type Errors

```haskell
-- Use type signatures for clarity
myFunction :: Species -> PhyloTree -> Maybe Double
myFunction sp tree = patristicDistance sp sp tree
```

### Module Loading

```bash
# If imports fail, check module dependencies:
ghci -v PhyloTree.hs  # Verbose output
```

## Next Steps

1. Read the [README.md](README.md) for comprehensive documentation
2. Explore [Examples.hs](Examples.hs) for complete examples
3. Run [Tests.hs](Tests.hs) to verify correctness
4. Extend the tree with new species
5. Implement custom analysis functions

## Resources

- GHCi commands: `:help`
- Type of expression: `:type expression`
- Info about function: `:info function`
- Reload modules: `:reload`
- Browse module: `:browse PhyloTree`
