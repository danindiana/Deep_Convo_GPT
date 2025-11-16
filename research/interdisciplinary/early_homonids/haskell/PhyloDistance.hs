{-|
Module      : PhyloDistance
Description : Phylogenetic distance calculations and metrics
Copyright   : (c) 2025
License     : MIT
Maintainer  : phylo@example.com

This module provides type-safe distance calculations between species,
including patristic distance, genetic distance, and morphological distance.
-}

module PhyloDistance
    ( -- * Distance Metrics
      patristicDistance
    , morphologicalDistance
    , temporalDistance
    , phylogeneticDistance

    -- * Distance Matrices
    , DistanceMatrix
    , buildDistanceMatrix
    , buildPatristicMatrix
    , buildMorphologicalMatrix

    -- * Distance-based Analysis
    , nearestNeighbor
    , kNearestNeighbors
    , distanceToMRCA
    , averageDistance

    -- * Clustering
    , upgmaTree
    , neighborJoining
    , ClusterNode(..)

    -- * Distance Matrix Operations
    , matrixLookup
    , matrixToList
    , normalizeMatrix

    ) where

import PhyloTree
import PhyloAnalysis
import Data.List (minimumBy, sortBy, find)
import Data.Maybe (fromMaybe, mapMaybe)
import Data.Ord (comparing)
import qualified Data.Map.Strict as Map

-- * Distance Metrics

-- | Calculate patristic distance (sum of branch lengths along path)
patristicDistance :: Species -> Species -> PhyloTree -> Maybe Double
patristicDistance sp1 sp2 tree = do
    path <- findPath sp1 sp2 tree
    return $ pathDistance path
  where
    pathDistance :: [Species] -> Double
    pathDistance [] = 0
    pathDistance [_] = 0
    pathDistance (x:y:rest) = branchLength x y + pathDistance (y:rest)

-- | Calculate morphological distance based on trait differences
morphologicalDistance :: Species -> Species -> Double
morphologicalDistance sp1 sp2 =
    let ccDist = traitDist getCranialCapacity 1.0
        bdDist = traitDist getBipedalism 1000.0  -- Scale up for visibility
        bmDist = traitDist getBodyMass 10.0
    in sqrt (ccDist^2 + bdDist^2 + bmDist^2)
  where
    traitDist :: (Species -> Maybe Double) -> Double -> Double
    traitDist extractor scale =
        let v1 = fromMaybe 0 (extractor sp1)
            v2 = fromMaybe 0 (extractor sp2)
        in (v1 - v2) * scale

    getCranialCapacity :: Species -> Maybe Double
    getCranialCapacity s = find' (\case CranialCapacity cc -> Just cc; _ -> Nothing) (traits s)

    getBipedalism :: Species -> Maybe Double
    getBipedalism s = find' (\case BipedalismDegree bd -> Just bd; _ -> Nothing) (traits s)

    getBodyMass :: Species -> Maybe Double
    getBodyMass s = find' (\case BodyMass bm -> Just bm; _ -> Nothing) (traits s)

    find' :: (Trait -> Maybe a) -> [Trait] -> Maybe a
    find' _ [] = Nothing
    find' f (x:xs) = case f x of
                       Just v -> Just v
                       Nothing -> find' f xs

-- | Calculate temporal distance (difference in time ranges)
temporalDistance :: Species -> Species -> Double
temporalDistance sp1 sp2 =
    let (MYA t1, _) = timeRange sp1
        (MYA t2, _) = timeRange sp2
    in abs (t1 - t2)

-- | Calculate combined phylogenetic distance
-- Weighted combination of patristic, morphological, and temporal distances
phylogeneticDistance :: Species -> Species -> PhyloTree -> Double
phylogeneticDistance sp1 sp2 tree =
    let patristic = fromMaybe 0 (patristicDistance sp1 sp2 tree)
        morphological = morphologicalDistance sp1 sp2
        temporal = temporalDistance sp1 sp2
    in (patristic * 0.5) + (morphological * 0.0001) + (temporal * 0.3)

-- * Distance Matrix

-- | Distance matrix type - maps species pairs to distances
type DistanceMatrix = Map.Map (String, String) Double

-- | Build general distance matrix using custom distance function
buildDistanceMatrix :: (Species -> Species -> Double) -> [Species] -> DistanceMatrix
buildDistanceMatrix distFunc species =
    Map.fromList [((speciesName s1, speciesName s2), distFunc s1 s2)
                 | s1 <- species, s2 <- species]

-- | Build patristic distance matrix
buildPatristicMatrix :: PhyloTree -> DistanceMatrix
buildPatristicMatrix tree =
    let species = dfs tree
    in Map.fromList [((speciesName s1, speciesName s2),
                      fromMaybe 0 (patristicDistance s1 s2 tree))
                    | s1 <- species, s2 <- species]

-- | Build morphological distance matrix
buildMorphologicalMatrix :: PhyloTree -> DistanceMatrix
buildMorphologicalMatrix tree =
    let species = dfs tree
    in buildDistanceMatrix morphologicalDistance species

-- * Distance-based Analysis

-- | Find nearest neighbor to a species
nearestNeighbor :: Species -> [Species] -> (Species -> Species -> Double) -> Maybe Species
nearestNeighbor target candidates distFunc =
    case filter (/= target) candidates of
        [] -> Nothing
        others -> Just $ minimumBy (comparing (distFunc target)) others

-- | Find k nearest neighbors
kNearestNeighbors :: Int -> Species -> [Species] -> (Species -> Species -> Double) -> [Species]
kNearestNeighbors k target candidates distFunc =
    let others = filter (/= target) candidates
        sorted = sortBy (comparing (distFunc target)) others
    in take k sorted

-- | Calculate distance from species to most recent common ancestor
distanceToMRCA :: Species -> Species -> PhyloTree -> Maybe Double
distanceToMRCA sp1 sp2 tree = do
    mrca <- mostRecentCommonAncestor sp1 sp2 tree
    dist1 <- patristicDistance sp1 mrca tree
    dist2 <- patristicDistance sp2 mrca tree
    return $ (dist1 + dist2) / 2

-- | Calculate average distance from one species to all others
averageDistance :: Species -> [Species] -> (Species -> Species -> Double) -> Double
averageDistance target others distFunc =
    let distances = map (distFunc target) (filter (/= target) others)
    in if null distances
       then 0
       else sum distances / fromIntegral (length distances)

-- * Clustering Algorithms

-- | Cluster node for hierarchical clustering
data ClusterNode
    = ClusterLeaf Species
    | ClusterNode ClusterNode ClusterNode Double  -- left, right, distance
    deriving (Eq, Show)

-- | UPGMA (Unweighted Pair Group Method with Arithmetic Mean) clustering
upgmaTree :: DistanceMatrix -> [Species] -> ClusterNode
upgmaTree matrix species =
    case species of
        [] -> error "Cannot build UPGMA tree from empty species list"
        [s] -> ClusterLeaf s
        _ -> upgma (map ClusterLeaf species)
  where
    upgma :: [ClusterNode] -> ClusterNode
    upgma [] = error "Empty cluster list"
    upgma [node] = node
    upgma nodes =
        let (node1, node2, dist) = findClosestPair nodes
            merged = ClusterNode node1 node2 dist
            remaining = filter (\n -> n /= node1 && n /= node2) nodes
        in upgma (merged : remaining)

    findClosestPair :: [ClusterNode] -> (ClusterNode, ClusterNode, Double)
    findClosestPair nodes =
        let pairs = [(n1, n2, clusterDistance n1 n2)
                    | n1 <- nodes, n2 <- nodes, n1 /= n2]
        in if null pairs
           then error "No pairs to cluster"
           else minimumBy (comparing (\(_, _, d) -> d)) pairs

    clusterDistance :: ClusterNode -> ClusterNode -> Double
    clusterDistance (ClusterLeaf s1) (ClusterLeaf s2) =
        matrixLookup (speciesName s1) (speciesName s2) matrix
    clusterDistance node1 node2 =
        let leaves1 = getClusterLeaves node1
            leaves2 = getClusterLeaves node2
            distances = [matrixLookup (speciesName s1) (speciesName s2) matrix
                        | s1 <- leaves1, s2 <- leaves2]
        in if null distances
           then 0
           else sum distances / fromIntegral (length distances)

-- | Get all leaves from a cluster node
getClusterLeaves :: ClusterNode -> [Species]
getClusterLeaves (ClusterLeaf s) = [s]
getClusterLeaves (ClusterNode left right _) =
    getClusterLeaves left ++ getClusterLeaves right

-- | Neighbor-Joining algorithm (more accurate than UPGMA)
neighborJoining :: DistanceMatrix -> [Species] -> ClusterNode
neighborJoining matrix species =
    case species of
        [] -> error "Cannot build NJ tree from empty species list"
        [s] -> ClusterLeaf s
        [s1, s2] -> ClusterNode (ClusterLeaf s1) (ClusterLeaf s2)
                                 (matrixLookup (speciesName s1) (speciesName s2) matrix)
        _ -> nj (map ClusterLeaf species)
  where
    nj :: [ClusterNode] -> ClusterNode
    nj [] = error "Empty cluster list"
    nj [node] = node
    nj nodes =
        let (node1, node2, dist) = findNJPair nodes
            merged = ClusterNode node1 node2 dist
            remaining = filter (\n -> n /= node1 && n /= node2) nodes
        in nj (merged : remaining)

    findNJPair :: [ClusterNode] -> (ClusterNode, ClusterNode, Double)
    findNJPair nodes =
        let pairs = [(n1, n2, njCriterion n1 n2 nodes)
                    | n1 <- nodes, n2 <- nodes, n1 /= n2]
        in if null pairs
           then error "No pairs to join"
           else minimumBy (comparing (\(_, _, d) -> d)) pairs

    njCriterion :: ClusterNode -> ClusterNode -> [ClusterNode] -> Double
    njCriterion node1 node2 allNodes =
        let leaves1 = getClusterLeaves node1
            leaves2 = getClusterLeaves node2
            s1 = head leaves1
            s2 = head leaves2
            dij = matrixLookup (speciesName s1) (speciesName s2) matrix
            n = fromIntegral (length allNodes)
            ui = sum [matrixLookup (speciesName s1) (speciesName (head (getClusterLeaves k))) matrix
                     | k <- allNodes] / (n - 2)
            uj = sum [matrixLookup (speciesName s2) (speciesName (head (getClusterLeaves k))) matrix
                     | k <- allNodes] / (n - 2)
        in dij - ui - uj

-- * Distance Matrix Operations

-- | Lookup distance in matrix (symmetric)
matrixLookup :: String -> String -> DistanceMatrix -> Double
matrixLookup name1 name2 matrix =
    fromMaybe 0 $ Map.lookup (name1, name2) matrix
                  `mplus` Map.lookup (name2, name1) matrix
  where
    mplus :: Maybe a -> Maybe a -> Maybe a
    mplus (Just x) _ = Just x
    mplus Nothing y = y

-- | Convert distance matrix to list
matrixToList :: DistanceMatrix -> [((String, String), Double)]
matrixToList = Map.toList

-- | Normalize distance matrix to [0, 1] range
normalizeMatrix :: DistanceMatrix -> DistanceMatrix
normalizeMatrix matrix =
    let values = map snd (Map.toList matrix)
        maxDist = if null values then 1 else maximum values
    in if maxDist == 0
       then matrix
       else Map.map (/ maxDist) matrix

-- * Additional Distance Functions

-- | Calculate all pairwise distances
allPairwiseDistances :: [Species] -> PhyloTree -> [((Species, Species), Double)]
allPairwiseDistances species tree =
    [((s1, s2), fromMaybe 0 (patristicDistance s1 s2 tree))
    | s1 <- species, s2 <- species, s1 /= s2]

-- | Find most distant pair of species
mostDistantPair :: [Species] -> PhyloTree -> Maybe ((Species, Species), Double)
mostDistantPair species tree =
    case allPairwiseDistances species tree of
        [] -> Nothing
        distances -> Just $ maximumBy (comparing snd) distances

-- | Calculate diameter of the tree (maximum distance between any two leaves)
treeDiameter :: PhyloTree -> Double
treeDiameter tree =
    let leaves = getLeaves tree
    in case mostDistantPair leaves tree of
         Nothing -> 0
         Just (_, dist) -> dist

-- | Calculate distance from species to root
distanceToRoot :: Species -> PhyloTree -> Maybe Double
distanceToRoot species tree =
    patristicDistance (getSpecies tree) species tree

-- | Calculate sum of distances from species to all others
totalDistance :: Species -> [Species] -> PhyloTree -> Double
totalDistance species others tree =
    sum [fromMaybe 0 (patristicDistance species other tree)
        | other <- others, other /= species]

-- | Robinson-Foulds distance (for comparing tree topologies)
-- Simplified implementation counting symmetric differences in bipartitions
robinsonFouldsDistance :: PhyloTree -> PhyloTree -> Int
robinsonFouldsDistance tree1 tree2 =
    let bipartitions1 = extractBipartitions tree1
        bipartitions2 = extractBipartitions tree2
        symmetric = (bipartitions1 `setDiff` bipartitions2) ++
                   (bipartitions2 `setDiff` bipartitions1)
    in length symmetric
  where
    setDiff xs ys = filter (`notElem` ys) xs

-- | Extract bipartitions from tree (splits defined by each edge)
extractBipartitions :: PhyloTree -> [([String], [String])]
extractBipartitions tree =
    let allLeaves = map speciesName (getLeaves tree)
    in bipartitions tree allLeaves
  where
    bipartitions :: PhyloTree -> [String] -> [([String], [String])]
    bipartitions (Leaf _) _ = []
    bipartitions (Node _ children) allLeaves =
        let childLeaves = map (map speciesName . getLeaves) children
            splits = map (\subset -> (subset, allLeaves `listDiff` subset)) childLeaves
        in splits ++ concatMap (`bipartitions` allLeaves) children

    listDiff xs ys = filter (`notElem` ys) xs

-- | Calculate cophenetic correlation (correlation between distance matrices)
copheneticCorrelation :: DistanceMatrix -> DistanceMatrix -> Double
copheneticCorrelation matrix1 matrix2 =
    let pairs = Map.keys matrix1
        vals1 = [fromMaybe 0 (Map.lookup p matrix1) | p <- pairs]
        vals2 = [fromMaybe 0 (Map.lookup p matrix2) | p <- pairs]
    in correlation vals1 vals2

-- | Calculate Pearson correlation coefficient
correlation :: [Double] -> [Double] -> Double
correlation xs ys =
    let n = fromIntegral (length xs)
        meanX = sum xs / n
        meanY = sum ys / n
        covXY = sum [((x - meanX) * (y - meanY)) | (x, y) <- zip xs ys] / n
        varX = sum [((x - meanX) ^ 2) | x <- xs] / n
        varY = sum [((y - meanY) ^ 2) | y <- ys] / n
    in if varX == 0 || varY == 0
       then 0
       else covXY / (sqrt varX * sqrt varY)

-- | Calculate gamma statistic (correlation of distances with path lengths)
gammaStatistic :: PhyloTree -> Double
gammaStatistic tree =
    let leaves = getLeaves tree
        n = length leaves
        observedVar = varianceOfTerminalBranchLengths tree
        expectedVar = expectedVarianceUnderBrownian n
    in (observedVar - expectedVar) / sqrt expectedVar

-- | Variance of terminal branch lengths
varianceOfTerminalBranchLengths :: PhyloTree -> Double
varianceOfTerminalBranchLengths tree =
    let leaves = getLeaves tree
        root = getSpecies tree
        distances = mapMaybe (\leaf -> patristicDistance root leaf tree) leaves
        mean' = sum distances / fromIntegral (length distances)
        squaredDiffs = map (\d -> (d - mean') ^ 2) distances
    in if null squaredDiffs
       then 0
       else sum squaredDiffs / fromIntegral (length squaredDiffs)

-- | Expected variance under Brownian motion model
expectedVarianceUnderBrownian :: Int -> Double
expectedVarianceUnderBrownian n = fromIntegral n / 12.0

maximumBy :: (a -> a -> Ordering) -> [a] -> a
maximumBy cmp (x:xs) = foldl (\a b -> if cmp a b == GT then a else b) x xs
maximumBy _ [] = error "maximumBy: empty list"
