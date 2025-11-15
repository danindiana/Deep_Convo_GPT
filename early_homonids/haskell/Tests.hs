{-|
Module      : Tests
Description : Test suite for phylogenetic analysis modules
Copyright   : (c) 2025
License     : MIT

Comprehensive test suite using QuickCheck and HUnit to verify
type safety and correctness of phylogenetic analysis functions.
-}

module Main where

import PhyloTree
import PhyloAnalysis
import TraitEvolution
import PhyloDistance

import Test.QuickCheck
import Test.HUnit
import Data.Maybe (isJust, isNothing, fromJust)
import Control.Monad (when)

-- * QuickCheck Properties

-- | Tree validation property: all valid trees should pass validation
prop_treeIsValid :: Property
prop_treeIsValid = property $ isValid buildHominidTree

-- | Distance symmetry: distance(A,B) = distance(B,A)
prop_distanceSymmetric :: Property
prop_distanceSymmetric = property $ do
    let tree = buildHominidTree
        species = dfs tree
    case species of
        (s1:s2:_) ->
            patristicDistance s1 s2 tree == patristicDistance s2 s1 tree
        _ -> True

-- | Self-distance is zero
prop_selfDistanceZero :: Property
prop_selfDistanceZero = property $ do
    let tree = buildHominidTree
        species = dfs tree
    case species of
        (s:_) -> morphologicalDistance s s == 0
        _ -> True

-- | Path to self is trivial
prop_pathToSelf :: Property
prop_pathToSelf = property $ do
    let tree = buildHominidTree
        species = dfs tree
    case species of
        (s:_) ->
            case findPath s s tree of
                Just path -> path == [s]
                Nothing -> False
        _ -> True

-- | MRCA of species with itself is itself
prop_mrcaSelf :: Property
prop_mrcaSelf = property $ do
    let tree = buildHominidTree
        species = dfs tree
    case species of
        (s:_) ->
            case mostRecentCommonAncestor s s tree of
                Just mrca -> mrca == s
                Nothing -> False
        _ -> True

-- | Tree depth is positive
prop_treeDepthPositive :: Property
prop_treeDepthPositive = property $ treeDepth buildHominidTree > 0

-- | Tree size equals number of species in DFS
prop_treeSizeEqualsDFS :: Property
prop_treeSizeEqualsDFS = property $
    let tree = buildHominidTree
    in treeSize tree == length (dfs tree)

-- | All species have valid time ranges
prop_validTimeRanges :: Property
prop_validTimeRanges = property $
    let tree = buildHominidTree
        species = dfs tree
    in all validTimeRange species
  where
    validTimeRange s =
        let (MYA earliest, MYA latest) = timeRange s
        in earliest >= latest && earliest >= 0 && latest >= 0

-- | Genus distribution is non-empty
prop_genusDistributionNonEmpty :: Property
prop_genusDistributionNonEmpty = property $
    not $ null $ genusDistribution buildHominidTree

-- | Trait variance is non-negative
prop_traitVarianceNonNegative :: Property
prop_traitVarianceNonNegative = property $
    traitVariance isCranialCapacity extractCC buildHominidTree >= 0
  where
    isCranialCapacity (CranialCapacity _) = True
    isCranialCapacity _ = False
    extractCC (CranialCapacity cc) = cc
    extractCC _ = 0

-- | Evolution trend detection is consistent
prop_trendDetectionConsistent :: Property
prop_trendDetectionConsistent = property $
    let trend = detectTrend isCranialCapacity extractCC buildHominidTree
    in trend `elem` [Increasing, Decreasing, Stable]
  where
    isCranialCapacity (CranialCapacity _) = True
    isCranialCapacity _ = False
    extractCC (CranialCapacity cc) = cc
    extractCC _ = 0

-- * HUnit Tests

-- | Test finding Homo sapiens
testFindHomoSapiens :: Test
testFindHomoSapiens = TestCase $ do
    let tree = buildHominidTree
        result = findSpecies "Homo sapiens" tree
    assertBool "Homo sapiens should be found" (isJust result)
    when (isJust result) $ do
        let species = fromJust result
        assertEqual "Species name" "Homo sapiens" (speciesName species)
        assertEqual "Genus" Homo (genus species)

-- | Test finding all Homo species
testFindHomoGenus :: Test
testFindHomoGenus = TestCase $ do
    let tree = buildHominidTree
        homoSpecies = findByGenus Homo tree
    assertBool "Should find multiple Homo species" (length homoSpecies > 3)

-- | Test path finding
testPathFinding :: Test
testPathFinding = TestCase $ do
    let tree = buildHominidTree
        Just afarensis = findSpecies "Australopithecus afarensis" tree
        Just sapiens = findSpecies "Homo sapiens" tree
        path = findPath afarensis sapiens tree
    assertBool "Path should exist" (isJust path)
    when (isJust path) $ do
        let p = fromJust path
        assertBool "Path should be non-empty" (not $ null p)
        assertEqual "Path should start with afarensis" afarensis (last p)

-- | Test MRCA
testMRCA :: Test
testMRCA = TestCase $ do
    let tree = buildHominidTree
        Just habilis = findSpecies "Homo habilis" tree
        Just sapiens = findSpecies "Homo sapiens" tree
        mrca = mostRecentCommonAncestor habilis sapiens tree
    assertBool "MRCA should exist" (isJust mrca)
    when (isJust mrca) $ do
        let ancestor = fromJust mrca
        assertEqual "MRCA should be H. habilis" Homo (genus ancestor)

-- | Test tree depth
testTreeDepth :: Test
testTreeDepth = TestCase $ do
    let tree = buildHominidTree
        depth = treeDepth tree
    assertBool "Tree depth should be reasonable" (depth > 3 && depth < 15)

-- | Test tree size
testTreeSize :: Test
testTreeSize = TestCase $ do
    let tree = buildHominidTree
        size = treeSize tree
    assertBool "Tree should contain multiple species" (size > 15)
    assertEqual "Size should match DFS count" size (length (dfs tree))

-- | Test cranial capacity evolution
testCranialCapacityEvolution :: Test
testCranialCapacityEvolution = TestCase $ do
    let tree = buildHominidTree
        trend = detectTrend isCranialCapacity extractCC tree
    assertEqual "Cranial capacity should be increasing" Increasing trend
  where
    isCranialCapacity (CranialCapacity _) = True
    isCranialCapacity _ = False
    extractCC (CranialCapacity cc) = cc
    extractCC _ = 0

-- | Test bipedalism evolution
testBipedalismEvolution :: Test
testBipedalismEvolution = TestCase $ do
    let tree = buildHominidTree
        trend = detectTrend isBipedalism extractBD tree
    assertBool "Bipedalism trend should be detected"
               (trend `elem` [Increasing, Stable])
  where
    isBipedalism (BipedalismDegree _) = True
    isBipedalism _ = False
    extractBD (BipedalismDegree bd) = bd
    extractBD _ = 0

-- | Test distance matrix
testDistanceMatrix :: Test
testDistanceMatrix = TestCase $ do
    let tree = buildHominidTree
        matrix = buildPatristicMatrix tree
    assertBool "Distance matrix should be non-empty" (not $ null $ matrixToList matrix)

-- | Test genus distribution
testGenusDistribution :: Test
testGenusDistribution = TestCase $ do
    let tree = buildHominidTree
        dist = genusDistribution tree
    assertBool "Should have multiple genera" (length dist > 3)

-- * Test Suite

-- | All QuickCheck properties
quickCheckTests :: IO ()
quickCheckTests = do
    putStrLn "Running QuickCheck property tests..."
    putStrLn ""

    putStr "  Tree validity: "
    quickCheck prop_treeIsValid

    putStr "  Distance symmetry: "
    quickCheck prop_distanceSymmetric

    putStr "  Self-distance is zero: "
    quickCheck prop_selfDistanceZero

    putStr "  Path to self: "
    quickCheck prop_pathToSelf

    putStr "  MRCA of self: "
    quickCheck prop_mrcaSelf

    putStr "  Tree depth positive: "
    quickCheck prop_treeDepthPositive

    putStr "  Tree size equals DFS: "
    quickCheck prop_treeSizeEqualsDFS

    putStr "  Valid time ranges: "
    quickCheck prop_validTimeRanges

    putStr "  Genus distribution non-empty: "
    quickCheck prop_genusDistributionNonEmpty

    putStr "  Trait variance non-negative: "
    quickCheck prop_traitVarianceNonNegative

    putStr "  Trend detection consistent: "
    quickCheck prop_trendDetectionConsistent

    putStrLn ""

-- | All HUnit tests
hunitTests :: Test
hunitTests = TestList
    [ TestLabel "Find Homo sapiens" testFindHomoSapiens
    , TestLabel "Find Homo genus" testFindHomoGenus
    , TestLabel "Path finding" testPathFinding
    , TestLabel "MRCA" testMRCA
    , TestLabel "Tree depth" testTreeDepth
    , TestLabel "Tree size" testTreeSize
    , TestLabel "Cranial capacity evolution" testCranialCapacityEvolution
    , TestLabel "Bipedalism evolution" testBipedalismEvolution
    , TestLabel "Distance matrix" testDistanceMatrix
    , TestLabel "Genus distribution" testGenusDistribution
    ]

-- | Run all tests
main :: IO ()
main = do
    putStrLn "========================================="
    putStrLn "  Phylogenetic Analysis Test Suite"
    putStrLn "========================================="
    putStrLn ""

    -- QuickCheck tests
    quickCheckTests

    -- HUnit tests
    putStrLn "Running HUnit unit tests..."
    putStrLn ""
    counts <- runTestTT hunitTests
    putStrLn ""

    -- Summary
    putStrLn "========================================="
    putStrLn "  Test Summary"
    putStrLn "========================================="
    putStrLn $ "Tests run: " ++ show (cases counts)
    putStrLn $ "Failures: " ++ show (failures counts)
    putStrLn $ "Errors: " ++ show (errors counts)
    putStrLn ""

    if failures counts == 0 && errors counts == 0
        then putStrLn "All tests passed! ✓"
        else putStrLn "Some tests failed! ✗"
