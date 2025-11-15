{-|
Module      : Examples
Description : Comprehensive examples demonstrating phylogenetic analysis
Copyright   : (c) 2025
License     : MIT
Maintainer  : phylo@example.com

This module provides complete working examples of phylogenetic analysis
using the type-safe Haskell implementation.
-}

module Examples where

import PhyloTree
import PhyloAnalysis
import TraitEvolution
import PhyloDistance
import qualified Data.Map.Strict as Map

-- * Example 1: Basic Tree Construction and Queries

example1_BasicQueries :: IO ()
example1_BasicQueries = do
    putStrLn "=== Example 1: Basic Tree Construction and Queries ==="
    putStrLn ""

    let tree = buildHominidTree

    -- Tree metrics
    putStrLn $ "Tree depth: " ++ show (treeDepth tree)
    putStrLn $ "Tree size: " ++ show (treeSize tree)
    putStrLn $ "Number of species: " ++ show (length (dfs tree))
    putStrLn ""

    -- Find specific species
    case findSpecies "Homo sapiens" tree of
        Nothing -> putStrLn "Homo sapiens not found"
        Just species -> do
            putStrLn $ "Found: " ++ speciesName species
            putStrLn $ "Genus: " ++ show (genus species)
            putStrLn $ "Location: " ++ location species
            putStrLn $ "Traits: " ++ show (traits species)
    putStrLn ""

    -- Find all Homo species
    let homoSpecies = findByGenus Homo tree
    putStrLn $ "Homo species count: " ++ show (length homoSpecies)
    putStrLn "Homo species:"
    mapM_ (putStrLn . ("  - " ++) . speciesName) homoSpecies
    putStrLn ""

-- * Example 2: Phylogenetic Analysis

example2_PhylogeneticAnalysis :: IO ()
example2_PhylogeneticAnalysis = do
    putStrLn "=== Example 2: Phylogenetic Analysis ==="
    putStrLn ""

    let tree = buildHominidTree

    -- Find path between species
    case (findSpecies "Australopithecus afarensis" tree,
          findSpecies "Homo sapiens" tree) of
        (Just afarensis, Just sapiens) -> do
            putStrLn "Finding evolutionary path from A. afarensis to H. sapiens:"
            case findPath afarensis sapiens tree of
                Nothing -> putStrLn "No path found"
                Just path -> do
                    putStrLn $ "Path length: " ++ show (pathLength path)
                    putStrLn "Path:"
                    mapM_ (putStrLn . ("  -> " ++) . speciesName) path
            putStrLn ""

            -- Find MRCA
            case mostRecentCommonAncestor afarensis sapiens tree of
                Nothing -> putStrLn "No common ancestor found"
                Just mrca -> do
                    putStrLn $ "Most recent common ancestor: " ++ speciesName mrca
                    putStrLn ""

            -- Calculate divergence time
            case divergenceTime afarensis sapiens tree of
                Nothing -> putStrLn "Cannot calculate divergence time"
                Just (MYA time) -> do
                    putStrLn $ "Divergence time: " ++ show time ++ " MYA"
                    putStrLn ""

        _ -> putStrLn "Species not found"

-- * Example 3: Trait Evolution Analysis

example3_TraitEvolution :: IO ()
example3_TraitEvolution = do
    putStrLn "=== Example 3: Trait Evolution Analysis ==="
    putStrLn ""

    let tree = buildHominidTree

    -- Trace cranial capacity evolution
    putStrLn "Cranial Capacity Evolution:"
    let ccTimeline = traitTimeline isCranialCapacity extractCC tree
    mapM_ (\(MYA time, cc) ->
        putStrLn $ "  " ++ show time ++ " MYA: " ++ show cc ++ " cc")
        (take 10 ccTimeline)
    putStrLn ""

    -- Detect trend
    let trend = detectTrend isCranialCapacity extractCC tree
    putStrLn $ "Cranial capacity trend: " ++ show trend
    putStrLn ""

    -- Calculate trait statistics
    let ccStats = traitStatistics isCranialCapacity extractCC tree
    putStrLn "Cranial Capacity Statistics:"
    putStrLn $ "  Count: " ++ show (count ccStats)
    putStrLn $ "  Mean: " ++ show (mean ccStats) ++ " cc"
    putStrLn $ "  Min: " ++ show (minVal ccStats) ++ " cc"
    putStrLn $ "  Max: " ++ show (maxVal ccStats) ++ " cc"
    putStrLn $ "  Std Dev: " ++ show (stdDev ccStats) ++ " cc"
    putStrLn ""

    -- Bipedalism evolution
    putStrLn "Bipedalism Evolution:"
    let bdTimeline = traitTimeline isBipedalism extractBD tree
    mapM_ (\(MYA time, bd) ->
        putStrLn $ "  " ++ show time ++ " MYA: " ++ show (bd * 100) ++ "%")
        (take 10 bdTimeline)
    putStrLn ""

  where
    isCranialCapacity (CranialCapacity _) = True
    isCranialCapacity _ = False

    extractCC (CranialCapacity cc) = cc
    extractCC _ = 0

    isBipedalism (BipedalismDegree _) = True
    isBipedalism _ = False

    extractBD (BipedalismDegree bd) = bd
    extractBD _ = 0

-- * Example 4: Distance Calculations

example4_Distances :: IO ()
example4_Distances = do
    putStrLn "=== Example 4: Distance Calculations ==="
    putStrLn ""

    let tree = buildHominidTree

    case (findSpecies "Homo habilis" tree,
          findSpecies "Homo sapiens" tree,
          findSpecies "Australopithecus afarensis" tree) of
        (Just habilis, Just sapiens, Just afarensis) -> do
            -- Patristic distance
            case patristicDistance habilis sapiens tree of
                Nothing -> putStrLn "Cannot calculate patristic distance"
                Just dist -> do
                    putStrLn $ "Patristic distance (H. habilis to H. sapiens): " ++
                              show dist ++ " MY"
            putStrLn ""

            -- Morphological distance
            let morphDist = morphologicalDistance habilis sapiens
            putStrLn $ "Morphological distance (H. habilis to H. sapiens): " ++
                      show morphDist
            putStrLn ""

            -- Temporal distance
            let tempDist = temporalDistance habilis sapiens
            putStrLn $ "Temporal distance (H. habilis to H. sapiens): " ++
                      show tempDist ++ " MY"
            putStrLn ""

            -- Compare with A. afarensis
            case patristicDistance afarensis sapiens tree of
                Nothing -> putStrLn "Cannot calculate patristic distance"
                Just dist -> do
                    putStrLn $ "Patristic distance (A. afarensis to H. sapiens): " ++
                              show dist ++ " MY"
            putStrLn ""

        _ -> putStrLn "Species not found"

    -- Build distance matrix
    let allSpecies = dfs tree
        matrix = buildPatristicMatrix tree
    putStrLn $ "Distance matrix size: " ++ show (Map.size matrix)
    putStrLn ""

-- * Example 5: Selection Pressure Analysis

example5_SelectionAnalysis :: IO ()
example5_SelectionAnalysis = do
    putStrLn "=== Example 5: Selection Pressure Analysis ==="
    putStrLn ""

    let tree = buildHominidTree

    -- Directional selection on cranial capacity
    let dirSelection = directionalSelection isCranialCapacity extractCC tree
    putStrLn "Directional Selection on Cranial Capacity:"
    putStrLn $ "  Type: " ++ selectionType dirSelection
    putStrLn $ "  Trend: " ++ show (trend dirSelection)
    putStrLn $ "  Rate: " ++ show (selectionRate dirSelection)
    putStrLn $ "  Strength: " ++ show (strength dirSelection)
    putStrLn ""

    -- Stabilizing selection on bipedalism
    let stabSelection = stabilizingSelection isBipedalism extractBD tree
    putStrLn "Stabilizing Selection on Bipedalism:"
    putStrLn $ "  Type: " ++ selectionType stabSelection
    putStrLn $ "  Variance: " ++ show (selectionRate stabSelection)
    putStrLn $ "  Strength: " ++ show (strength stabSelection)
    putStrLn ""

  where
    isCranialCapacity (CranialCapacity _) = True
    isCranialCapacity _ = False

    extractCC (CranialCapacity cc) = cc
    extractCC _ = 0

    isBipedalism (BipedalismDegree _) = True
    isBipedalism _ = False

    extractBD (BipedalismDegree bd) = bd
    extractBD _ = 0

-- * Example 6: Adaptive Radiation

example6_AdaptiveRadiation :: IO ()
example6_AdaptiveRadiation = do
    putStrLn "=== Example 6: Adaptive Radiation Analysis ==="
    putStrLn ""

    let tree = buildHominidTree

    -- Analyze each genus for adaptive radiation
    let genera = [minBound .. maxBound] :: [Genus]
    mapM_ analyzeGenus genera
    putStrLn ""

  where
    analyzeGenus :: Genus -> IO ()
    analyzeGenus g = do
        let tree = buildHominidTree
            radiation = adaptiveRadiation g tree
        putStrLn $ show g ++ ":"
        putStrLn $ "  Species involved: " ++ show (speciesInvolved radiation)
        putStrLn $ "  Trait diversification: " ++ show (traitDiversification radiation)
        putStrLn $ "  Temporal duration: " ++ show (temporalDuration radiation) ++ " MY"
        putStrLn $ "  Is radiation event: " ++ show (isRadiation radiation)

-- * Example 7: Genus Distribution

example7_GenusDistribution :: IO ()
example7_GenusDistribution = do
    putStrLn "=== Example 7: Genus Distribution ==="
    putStrLn ""

    let tree = buildHominidTree
        distribution = genusDistribution tree

    putStrLn "Species count by genus:"
    mapM_ (\(g, count) ->
        putStrLn $ "  " ++ show g ++ ": " ++ show count ++ " species")
        (Map.toList distribution)
    putStrLn ""

-- * Example 8: Temporal Analysis

example8_TemporalAnalysis :: IO ()
example8_TemporalAnalysis = do
    putStrLn "=== Example 8: Temporal Distribution ==="
    putStrLn ""

    let tree = buildHominidTree
        distribution = temporalDistribution tree

    putStrLn "Species count by time period:"
    let sorted = take 10 $ reverse $ Map.toList distribution
    mapM_ (\(period, count) ->
        putStrLn $ "  " ++ show period ++ " MYA: " ++ show count ++ " species")
        sorted
    putStrLn ""

-- * Example 9: Comparative Species Analysis

example9_ComparativeAnalysis :: IO ()
example9_ComparativeAnalysis = do
    putStrLn "=== Example 9: Comparative Species Analysis ==="
    putStrLn ""

    let tree = buildHominidTree

    case (findSpecies "Homo erectus" tree,
          findSpecies "Homo neanderthalensis" tree) of
        (Just erectus, Just neanderthal) -> do
            putStrLn "Comparing H. erectus and H. neanderthalensis:"
            putStrLn ""

            let comparisons = compareSpecies erectus neanderthal
            mapM_ (\(traitName, val1, val2) -> do
                putStrLn $ "  " ++ traitName ++ ":"
                putStrLn $ "    H. erectus: " ++ show val1
                putStrLn $ "    H. neanderthalensis: " ++ show val2
                putStrLn $ "    Difference: " ++ show (abs (val1 - val2)))
                comparisons
            putStrLn ""

        _ -> putStrLn "Species not found"

-- * Example 10: Evolutionary Rates

example10_EvolutionaryRates :: IO ()
example10_EvolutionaryRates = do
    putStrLn "=== Example 10: Evolutionary Rates ==="
    putStrLn ""

    let tree = buildHominidTree
        homoSpecies = findByGenus Homo tree

    putStrLn "Cranial capacity evolutionary rates in Homo lineage:"
    mapM_ calculateRate (pairs homoSpecies)
    putStrLn ""

  where
    pairs :: [a] -> [(a, a)]
    pairs [] = []
    pairs [_] = []
    pairs (x:y:rest) = (x, y) : pairs (y:rest)

    calculateRate :: (Species, Species) -> IO ()
    calculateRate (sp1, sp2) = do
        let rate = evolutionaryRate isCranialCapacity extractCC sp1 sp2
        putStrLn $ "  " ++ speciesName sp1 ++ " -> " ++ speciesName sp2 ++
                  ": " ++ show rate ++ " cc/MY"

    isCranialCapacity (CranialCapacity _) = True
    isCranialCapacity _ = False

    extractCC (CranialCapacity cc) = cc
    extractCC _ = 0

-- * Main - Run All Examples

runAllExamples :: IO ()
runAllExamples = do
    putStrLn "========================================="
    putStrLn "  Phylogenetic Analysis Examples"
    putStrLn "  Type-Safe Haskell Implementation"
    putStrLn "========================================="
    putStrLn ""

    example1_BasicQueries
    putStrLn ""
    example2_PhylogeneticAnalysis
    putStrLn ""
    example3_TraitEvolution
    putStrLn ""
    example4_Distances
    putStrLn ""
    example5_SelectionAnalysis
    putStrLn ""
    example6_AdaptiveRadiation
    putStrLn ""
    example7_GenusDistribution
    putStrLn ""
    example8_TemporalAnalysis
    putStrLn ""
    example9_ComparativeAnalysis
    putStrLn ""
    example10_EvolutionaryRates
    putStrLn ""

    putStrLn "========================================="
    putStrLn "  All examples completed successfully!"
    putStrLn "========================================="

-- For GHCi convenience
main :: IO ()
main = runAllExamples
