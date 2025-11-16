{-# LANGUAGE ScopedTypeVariables #-}

{-|
Module      : PhyloAnalysis
Description : Phylogenetic analysis algorithms and metrics
Copyright   : (c) 2025
License     : MIT
Maintainer  : phylo@example.com

This module provides type-safe algorithms for analyzing phylogenetic trees,
including path finding, distance calculations, and ancestral relationships.
-}

module PhyloAnalysis
    ( -- * Path Finding
      findPath
    , findAllPaths
    , shortestPath
    , pathLength

    -- * Ancestral Relationships
    , isAncestor
    , findCommonAncestor
    , mostRecentCommonAncestor
    , lineage

    -- * Divergence Analysis
    , divergenceTime
    , timeSinceDivergence
    , branchLength

    -- * Tree Metrics
    , treeHeight
    , averageBranchLength
    , phylogeneticDiversity
    , balanceIndex

    -- * Statistical Analysis
    , genusDistribution
    , temporalDistribution
    , traitStatistics

    -- * Comparative Analysis
    , compareSpecies
    , findSimilar
    , traitDifference

    ) where

import PhyloTree
import Data.List (find, minimumBy, sortBy, group, sort, nub)
import Data.Maybe (mapMaybe, listToMaybe, fromMaybe, catMaybes)
import Data.Ord (comparing)
import qualified Data.Map.Strict as Map

-- * Path Finding Algorithms

-- | Find a path between two species in the tree
-- Returns Nothing if no path exists
findPath :: Species -> Species -> PhyloTree -> Maybe [Species]
findPath start end tree
    | start == end = Just [start]
    | otherwise = findPath' start end tree
  where
    findPath' :: Species -> Species -> PhyloTree -> Maybe [Species]
    findPath' s e (Leaf current)
        | current == e = Just [e]
        | otherwise = Nothing
    findPath' s e (Node current children)
        | current == e = Just [e]
        | otherwise =
            case mapMaybe (findPath' s e) children of
                [] -> Nothing
                (path:_) -> Just (current : path)

-- | Find all possible paths between two species
findAllPaths :: Species -> Species -> PhyloTree -> [[Species]]
findAllPaths start end tree = findAllPaths' start end tree
  where
    findAllPaths' :: Species -> Species -> PhyloTree -> [[Species]]
    findAllPaths' s e (Leaf current)
        | current == e = [[e]]
        | otherwise = []
    findAllPaths' s e (Node current children)
        | current == e = [[e]]
        | otherwise =
            let childPaths = concatMap (findAllPaths' s e) children
            in map (current :) childPaths

-- | Find shortest path between two species
shortestPath :: Species -> Species -> PhyloTree -> Maybe [Species]
shortestPath start end tree =
    case findAllPaths start end tree of
        [] -> Nothing
        paths -> Just $ minimumBy (comparing length) paths

-- | Calculate path length (number of edges)
pathLength :: [Species] -> Int
pathLength [] = 0
pathLength [_] = 0
pathLength xs = length xs - 1

-- * Ancestral Relationship Analysis

-- | Check if first species is an ancestor of the second
isAncestor :: Species -> Species -> PhyloTree -> Bool
isAncestor ancestor descendant tree =
    case findPath ancestor descendant tree of
        Nothing -> False
        Just path -> ancestor `elem` path && ancestor /= descendant

-- | Find common ancestors of two species
findCommonAncestor :: Species -> Species -> PhyloTree -> [Species]
findCommonAncestor sp1 sp2 tree =
    let path1 = maybe [] id (findPath (getSpecies tree) sp1 tree)
        path2 = maybe [] id (findPath (getSpecies tree) sp2 tree)
    in commonElements path1 path2
  where
    commonElements :: Eq a => [a] -> [a] -> [a]
    commonElements xs ys = filter (`elem` ys) xs

-- | Find most recent common ancestor (MRCA)
mostRecentCommonAncestor :: Species -> Species -> PhyloTree -> Maybe Species
mostRecentCommonAncestor sp1 sp2 tree =
    case findCommonAncestor sp1 sp2 tree of
        [] -> Nothing
        ancestors -> Just $ last ancestors

-- | Get complete lineage from root to species
lineage :: Species -> PhyloTree -> Maybe [Species]
lineage target tree = findPath (getSpecies tree) target tree

-- * Divergence Analysis

-- | Calculate divergence time between two species
-- Returns the time of their most recent common ancestor
divergenceTime :: Species -> Species -> PhyloTree -> Maybe TimeScale
divergenceTime sp1 sp2 tree = do
    mrca <- mostRecentCommonAncestor sp1 sp2 tree
    let (earliest, _) = timeRange mrca
    return earliest

-- | Calculate time since divergence from MRCA
timeSinceDivergence :: Species -> Species -> PhyloTree -> Maybe TimeScale
timeSinceDivergence sp1 sp2 tree = do
    divTime <- divergenceTime sp1 sp2 tree
    let (MYA dt) = divTime
        (MYA s1, _) = timeRange sp1
        (MYA s2, _) = timeRange sp2
        avgSpeciesTime = (s1 + s2) / 2
    return $ MYA (dt - avgSpeciesTime)

-- | Calculate branch length (temporal distance from parent to child)
branchLength :: Species -> Species -> Double
branchLength parent child =
    let (MYA pEarliest, _) = timeRange parent
        (MYA cEarliest, _) = timeRange child
    in abs (pEarliest - cEarliest)

-- * Tree Metrics

-- | Calculate maximum temporal depth of the tree
treeHeight :: PhyloTree -> Double
treeHeight tree =
    let allSpecies = dfs tree
        times = map (\s -> let (MYA t, _) = timeRange s in t) allSpecies
    in maximum times - minimum times

-- | Calculate average branch length across all parent-child pairs
averageBranchLength :: PhyloTree -> Double
averageBranchLength tree =
    let lengths = branchLengths tree
    in if null lengths then 0 else sum lengths / fromIntegral (length lengths)
  where
    branchLengths :: PhyloTree -> [Double]
    branchLengths (Leaf _) = []
    branchLengths (Node parent children) =
        let parentSp = parent
            childSpecies = map getSpecies children
            lengths = map (branchLength parentSp) childSpecies
            childLengths = concatMap branchLengths children
        in lengths ++ childLengths

-- | Calculate phylogenetic diversity (sum of all branch lengths)
phylogeneticDiversity :: PhyloTree -> Double
phylogeneticDiversity tree = sum (allBranchLengths tree)
  where
    allBranchLengths :: PhyloTree -> [Double]
    allBranchLengths (Leaf _) = []
    allBranchLengths (Node parent children) =
        let parentSp = parent
            childSpecies = map getSpecies children
            lengths = map (branchLength parentSp) childSpecies
        in lengths ++ concatMap allBranchLengths children

-- | Calculate tree balance index (lower = more balanced)
-- Uses Colless' index: sum of |L - R| for each node
balanceIndex :: PhyloTree -> Int
balanceIndex (Leaf _) = 0
balanceIndex (Node _ children) =
    let sizes = map treeSize children
        imbalance = if length sizes >= 2
                    then abs (head sizes - (sizes !! 1))
                    else 0
    in imbalance + sum (map balanceIndex children)

-- * Statistical Analysis

-- | Calculate distribution of species across genera
genusDistribution :: PhyloTree -> Map.Map Genus Int
genusDistribution tree =
    let allSpecies = dfs tree
        genera = map genus allSpecies
    in Map.fromListWith (+) [(g, 1) | g <- genera]

-- | Calculate temporal distribution of species
-- Returns map from time period to count
temporalDistribution :: PhyloTree -> Map.Map Int Int
temporalDistribution tree =
    let allSpecies = dfs tree
        periods = map (\s -> let (MYA t, _) = timeRange s in floor t) allSpecies
    in Map.fromListWith (+) [(p, 1) | p <- periods]

-- | Calculate statistics for a specific trait
traitStatistics :: (Trait -> Bool) -> (Trait -> Double) -> PhyloTree -> TraitStats
traitStatistics predicate extractor tree =
    let allSpecies = dfs tree
        relevantTraits = concatMap (filter predicate . traits) allSpecies
        values = map extractor relevantTraits
    in if null values
       then TraitStats 0 0 0 0 0
       else TraitStats
            { count = length values
            , mean = sum values / fromIntegral (length values)
            , minVal = minimum values
            , maxVal = maximum values
            , stdDev = sqrt $ sum [(x - m) ^ 2 | x <- values] / fromIntegral (length values)
            }
  where
    m = sum values / fromIntegral (length values)
    values = map extractor relevantTraits

data TraitStats = TraitStats
    { count :: Int
    , mean :: Double
    , minVal :: Double
    , maxVal :: Double
    , stdDev :: Double
    } deriving (Eq, Show)

-- * Comparative Analysis

-- | Compare two species across all shared trait types
compareSpecies :: Species -> Species -> [(String, Double, Double)]
compareSpecies sp1 sp2 =
    let traits1 = traits sp1
        traits2 = traits sp2
    in mapMaybe compareTrait (zip traits1 traits2)
  where
    compareTrait :: (Trait, Trait) -> Maybe (String, Double, Double)
    compareTrait (CranialCapacity c1, CranialCapacity c2) =
        Just ("Cranial Capacity", c1, c2)
    compareTrait (BipedalismDegree b1, BipedalismDegree b2) =
        Just ("Bipedalism", b1, b2)
    compareTrait (BodyMass m1, BodyMass m2) =
        Just ("Body Mass", m1, m2)
    compareTrait (BrainToBodyRatio r1, BrainToBodyRatio r2) =
        Just ("Brain/Body Ratio", r1, r2)
    compareTrait _ = Nothing

-- | Find species similar to a given species based on traits
findSimilar :: Species -> PhyloTree -> Double -> [Species]
findSimilar target tree threshold =
    let allSpecies = dfs tree
        similarities = map (\s -> (s, similarity target s)) allSpecies
        similar = filter (\(s, sim) -> s /= target && sim >= threshold) similarities
    in map fst $ sortBy (comparing (negate . snd)) similar
  where
    similarity :: Species -> Species -> Double
    similarity s1 s2 =
        let comparisons = compareSpecies s1 s2
            differences = map (\(_, v1, v2) -> abs (v1 - v2)) comparisons
            normalized = if null differences
                        then 0
                        else sum differences / fromIntegral (length differences)
        in 1.0 / (1.0 + normalized)

-- | Calculate trait difference between two species
traitDifference :: Species -> Species -> String -> Maybe Double
traitDifference sp1 sp2 traitName =
    case traitName of
        "cranial_capacity" -> diffCC
        "bipedalism" -> diffBD
        "body_mass" -> diffBM
        "brain_body_ratio" -> diffBBR
        _ -> Nothing
  where
    diffCC = do
        cc1 <- findCranialCapacity sp1
        cc2 <- findCranialCapacity sp2
        return $ abs (cc1 - cc2)

    diffBD = do
        bd1 <- findBipedalism sp1
        bd2 <- findBipedalism sp2
        return $ abs (bd1 - bd2)

    diffBM = do
        bm1 <- findBodyMass sp1
        bm2 <- findBodyMass sp2
        return $ abs (bm1 - bm2)

    diffBBR = do
        bbr1 <- findBrainBodyRatio sp1
        bbr2 <- findBrainBodyRatio sp2
        return $ abs (bbr1 - bbr2)

    findCranialCapacity :: Species -> Maybe Double
    findCranialCapacity s = listToMaybe [cc | CranialCapacity cc <- traits s]

    findBipedalism :: Species -> Maybe Double
    findBipedalism s = listToMaybe [bd | BipedalismDegree bd <- traits s]

    findBodyMass :: Species -> Maybe Double
    findBodyMass s = listToMaybe [bm | BodyMass bm <- traits s]

    findBrainBodyRatio :: Species -> Maybe Double
    findBrainBodyRatio s = listToMaybe [eq | BrainToBodyRatio eq <- traits s]

-- * Additional Helper Functions

-- | Get all species within a time range
speciesInTimeRange :: TimeScale -> TimeScale -> PhyloTree -> [Species]
speciesInTimeRange minTime maxTime tree =
    filter inRange (dfs tree)
  where
    inRange :: Species -> Bool
    inRange s =
        let (earliest, latest) = timeRange s
        in earliest >= minTime && latest <= maxTime

-- | Count species by time period
speciesCountByPeriod :: PhyloTree -> [(Int, Int)]
speciesCountByPeriod tree =
    let distribution = temporalDistribution tree
    in sortBy (comparing fst) $ Map.toList distribution

-- | Find evolutionary trends in a trait over time
traitTrend :: (Trait -> Bool) -> (Trait -> Double) -> PhyloTree -> [(TimeScale, Double)]
traitTrend predicate extractor tree =
    let allSpecies = dfs tree
        withTrait = [(timeRange s, t) | s <- allSpecies, t <- filter predicate (traits s)]
        avgByTime = map (\((earliest, _), trait) -> (earliest, extractor trait)) withTrait
    in sortBy (comparing (negate . (\(MYA t) -> t) . fst)) avgByTime
