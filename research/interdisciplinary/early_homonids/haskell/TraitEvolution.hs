{-# LANGUAGE GADTs #-}
{-# LANGUAGE RankNTypes #-}

{-|
Module      : TraitEvolution
Description : Model and analyze trait evolution in phylogenetic trees
Copyright   : (c) 2025
License     : MIT
Maintainer  : phylo@example.com

This module provides type-safe modeling of trait evolution, including
directional selection, adaptive radiation, and evolutionary trends.
-}

module TraitEvolution
    ( -- * Trait Evolution Models
      EvolutionModel(..)
    , applyEvolutionModel
    , evolveTraits

    -- * Trait Tracking
    , traceTraitEvolution
    , traitTimeline
    , ancestralTraitReconstruction

    -- * Evolutionary Patterns
    , detectTrend
    , evolutionaryRate
    , adaptiveRadiation
    , convergentEvolution

    -- * Selection Analysis
    , directionalSelection
    , stabilizingSelection
    , disruptiveSelection

    -- * Trait-based Queries
    , speciesWithTrait
    , traitRange
    , traitMean
    , traitVariance

    -- * Evolutionary Metrics
    , traitDisparity
    , evolutionaryDistance
    , phenotypicDiversity

    ) where

import PhyloTree
import PhyloAnalysis
import Data.List (sortBy, nub, find)
import Data.Maybe (mapMaybe, catMaybes, fromMaybe)
import Data.Ord (comparing)
import qualified Data.Map.Strict as Map

-- * Evolution Models

-- | Type-safe evolution models
data EvolutionModel where
    -- | Gradual change at constant rate
    GradualChange :: Double -> EvolutionModel

    -- | Punctuated equilibrium (rapid change, then stasis)
    PunctuatedEquilibrium :: Double -> Double -> EvolutionModel

    -- | Adaptive radiation (rapid diversification)
    AdaptiveRadiation :: Double -> EvolutionModel

    -- | Directional selection toward optimum
    DirectionalSelection :: Double -> EvolutionModel

    -- | Neutral drift (random walk)
    NeutralDrift :: EvolutionModel

instance Show EvolutionModel where
    show (GradualChange rate) = "Gradual Change (rate: " ++ show rate ++ ")"
    show (PunctuatedEquilibrium rapid stasis) =
        "Punctuated Equilibrium (rapid: " ++ show rapid ++
        ", stasis: " ++ show stasis ++ ")"
    show (AdaptiveRadiation rate) = "Adaptive Radiation (rate: " ++ show rate ++ ")"
    show (DirectionalSelection optimum) =
        "Directional Selection (optimum: " ++ show optimum ++ ")"
    show NeutralDrift = "Neutral Drift"

-- | Apply evolution model to calculate expected trait value
applyEvolutionModel :: EvolutionModel -> Double -> Double -> Double
applyEvolutionModel (GradualChange rate) initialValue time =
    initialValue + rate * time
applyEvolutionModel (PunctuatedEquilibrium rapid stasis) initialValue time =
    if time < 0.5
        then initialValue + rapid * time
        else initialValue + rapid * 0.5 + stasis * (time - 0.5)
applyEvolutionModel (AdaptiveRadiation rate) initialValue time =
    initialValue * (1 + rate) ** time
applyEvolutionModel (DirectionalSelection optimum) initialValue time =
    initialValue + (optimum - initialValue) * (1 - exp (-time))
applyEvolutionModel NeutralDrift initialValue _ =
    initialValue

-- | Evolve traits along a lineage according to a model
evolveTraits :: EvolutionModel -> [Species] -> [(Species, Double)]
evolveTraits model lineage =
    case lineage of
        [] -> []
        (first:rest) ->
            let initialTrait = extractInitialTrait first
            in evolve initialTrait first rest
  where
    extractInitialTrait :: Species -> Double
    extractInitialTrait s =
        case find isCranialCapacity (traits s) of
            Just (CranialCapacity cc) -> cc
            _ -> 400.0  -- default

    isCranialCapacity (CranialCapacity _) = True
    isCranialCapacity _ = False

    evolve :: Double -> Species -> [Species] -> [(Species, Double)]
    evolve current sp [] = [(sp, current)]
    evolve current sp (next:rest) =
        let time = timeBetween sp next
            newValue = applyEvolutionModel model current time
        in (sp, current) : evolve newValue next rest

    timeBetween :: Species -> Species -> Double
    timeBetween s1 s2 =
        let (MYA t1, _) = timeRange s1
            (MYA t2, _) = timeRange s2
        in abs (t1 - t2)

-- * Trait Tracking

-- | Trace evolution of a specific trait through the tree
traceTraitEvolution :: (Trait -> Bool) -> PhyloTree -> [(Species, Trait)]
traceTraitEvolution predicate tree =
    let allSpecies = dfs tree
    in [(s, t) | s <- allSpecies, t <- traits s, predicate t]

-- | Create timeline of trait values
traitTimeline :: (Trait -> Bool) -> (Trait -> Double) -> PhyloTree -> [(TimeScale, Double)]
traitTimeline predicate extractor tree =
    let traced = traceTraitEvolution predicate tree
        timeline = [(earliest, extractor t) | (s, t) <- traced,
                    let (earliest, _) = timeRange s]
    in sortBy (comparing (negate . (\(MYA t) -> t) . fst)) timeline

-- | Reconstruct ancestral trait values using parsimony
ancestralTraitReconstruction :: (Trait -> Bool) -> (Trait -> Double) -> PhyloTree -> Map.Map String Double
ancestralTraitReconstruction predicate extractor tree =
    Map.fromList $ reconstruct tree
  where
    reconstruct :: PhyloTree -> [(String, Double)]
    reconstruct (Leaf s) =
        case find predicate (traits s) of
            Just trait -> [(speciesName s, extractor trait)]
            Nothing -> []
    reconstruct (Node s children) =
        let childValues = concatMap reconstruct children
            childMean = if null childValues
                       then 0
                       else sum (map snd childValues) / fromIntegral (length childValues)
            ancestral = case find predicate (traits s) of
                           Just trait -> extractor trait
                           Nothing -> childMean
        in (speciesName s, ancestral) : childValues

-- * Evolutionary Patterns

-- | Detect evolutionary trend (increasing, decreasing, or stable)
detectTrend :: (Trait -> Bool) -> (Trait -> Double) -> PhyloTree -> EvolutionaryTrend
detectTrend predicate extractor tree =
    let timeline = traitTimeline predicate extractor tree
        values = map snd timeline
    in if null values || length values < 2
       then Stable
       else analyzeTrend values
  where
    analyzeTrend :: [Double] -> EvolutionaryTrend
    analyzeTrend vals =
        let pairs = zip vals (tail vals)
            increases = length $ filter (\(a, b) -> b > a) pairs
            decreases = length $ filter (\(a, b) -> b < a) pairs
            total = length pairs
        in if increases > total `div` 2
           then Increasing
           else if decreases > total `div` 2
           then Decreasing
           else Stable

data EvolutionaryTrend = Increasing | Decreasing | Stable
    deriving (Eq, Show)

-- | Calculate evolutionary rate (trait change per million years)
evolutionaryRate :: (Trait -> Bool) -> (Trait -> Double) -> Species -> Species -> Double
evolutionaryRate predicate extractor ancestor descendant =
    let ancestorValue = findTraitValue predicate extractor ancestor
        descendantValue = findTraitValue predicate extractor descendant
        (MYA aTime, _) = timeRange ancestor
        (MYA dTime, _) = timeRange descendant
        timeDiff = abs (aTime - dTime)
    in if timeDiff == 0
       then 0
       else abs (descendantValue - ancestorValue) / timeDiff

-- | Identify adaptive radiation events (rapid speciation with trait divergence)
adaptiveRadiation :: Genus -> PhyloTree -> AdaptiveRadiationEvent
adaptiveRadiation targetGenus tree =
    let genusSpecies = findByGenus targetGenus tree
        speciesCount = length genusSpecies
        traitVariances = calculateTraitVariances genusSpecies
        timeSpan = calculateTimeSpan genusSpecies
    in AdaptiveRadiationEvent
        { radiationGenus = targetGenus
        , speciesInvolved = speciesCount
        , traitDiversification = average traitVariances
        , temporalDuration = timeSpan
        , isRadiation = speciesCount > 3 && average traitVariances > 100
        }

data AdaptiveRadiationEvent = AdaptiveRadiationEvent
    { radiationGenus :: Genus
    , speciesInvolved :: Int
    , traitDiversification :: Double
    , temporalDuration :: Double
    , isRadiation :: Bool
    } deriving (Eq, Show)

calculateTraitVariances :: [Species] -> [Double]
calculateTraitVariances species =
    let cranialCapacities = mapMaybe getCranialCapacity species
    in if null cranialCapacities
       then []
       else [variance cranialCapacities]
  where
    getCranialCapacity s = listToMaybe [cc | CranialCapacity cc <- traits s]
    listToMaybe [] = Nothing
    listToMaybe (x:_) = Just x

calculateTimeSpan :: [Species] -> Double
calculateTimeSpan [] = 0
calculateTimeSpan species =
    let times = [t | s <- species, let (MYA t, _) = timeRange s]
    in maximum times - minimum times

-- | Detect convergent evolution (similar traits in distantly related species)
convergentEvolution :: (Trait -> Bool) -> (Trait -> Double) -> PhyloTree -> [(Species, Species, Double)]
convergentEvolution predicate extractor tree =
    let allSpecies = dfs tree
        pairs = [(s1, s2) | s1 <- allSpecies, s2 <- allSpecies, s1 /= s2]
        convergent = mapMaybe (checkConvergence predicate extractor tree) pairs
    in convergent

checkConvergence :: (Trait -> Bool) -> (Trait -> Double) -> PhyloTree ->
                   (Species, Species) -> Maybe (Species, Species, Double)
checkConvergence predicate extractor tree (s1, s2) =
    let v1 = findTraitValue predicate extractor s1
        v2 = findTraitValue predicate extractor s2
        similarity = 1.0 / (1.0 + abs (v1 - v2))
        distantlyRelated = case divergenceTime s1 s2 tree of
                             Just (MYA t) -> t > 2.0  -- More than 2 MYA
                             Nothing -> False
    in if similarity > 0.9 && distantlyRelated
       then Just (s1, s2, similarity)
       else Nothing

-- * Selection Analysis

-- | Analyze directional selection on a trait
directionalSelection :: (Trait -> Bool) -> (Trait -> Double) -> PhyloTree -> SelectionPressure
directionalSelection predicate extractor tree =
    let timeline = traitTimeline predicate extractor tree
        trend = detectTrend predicate extractor tree
        rate = if length timeline >= 2
               then let (MYA t1, v1) = head timeline
                        (MYA t2, v2) = last timeline
                    in abs (v2 - v1) / abs (t1 - t2)
               else 0
    in SelectionPressure
        { selectionType = "Directional"
        , trend = trend
        , selectionRate = rate
        , strength = if rate > 100 then Strong else if rate > 50 then Moderate else Weak
        }

-- | Analyze stabilizing selection (selection against extremes)
stabilizingSelection :: (Trait -> Bool) -> (Trait -> Double) -> PhyloTree -> SelectionPressure
stabilizingSelection predicate extractor tree =
    let values = map snd $ traitTimeline predicate extractor tree
        var = if null values then 0 else variance values
        trend = detectTrend predicate extractor tree
    in SelectionPressure
        { selectionType = "Stabilizing"
        , trend = trend
        , selectionRate = var
        , strength = if var < 50 then Strong else if var < 100 then Moderate else Weak
        }

-- | Analyze disruptive selection (selection for extremes)
disruptiveSelection :: (Trait -> Bool) -> (Trait -> Double) -> PhyloTree -> SelectionPressure
disruptiveSelection predicate extractor tree =
    let values = map snd $ traitTimeline predicate extractor tree
        mean' = if null values then 0 else average values
        extremes = filter (\v -> abs (v - mean') > 200) values
        extremeRatio = if null values
                      then 0
                      else fromIntegral (length extremes) / fromIntegral (length values)
    in SelectionPressure
        { selectionType = "Disruptive"
        , trend = Stable
        , selectionRate = extremeRatio
        , strength = if extremeRatio > 0.3 then Strong else if extremeRatio > 0.15 then Moderate else Weak
        }

data SelectionPressure = SelectionPressure
    { selectionType :: String
    , trend :: EvolutionaryTrend
    , selectionRate :: Double
    , strength :: SelectionStrength
    } deriving (Eq, Show)

data SelectionStrength = Weak | Moderate | Strong
    deriving (Eq, Show, Ord)

-- * Trait-based Queries

-- | Find all species possessing a specific trait
speciesWithTrait :: (Trait -> Bool) -> PhyloTree -> [Species]
speciesWithTrait predicate tree =
    filter (any predicate . traits) (dfs tree)

-- | Calculate range of trait values
traitRange :: (Trait -> Bool) -> (Trait -> Double) -> PhyloTree -> (Double, Double)
traitRange predicate extractor tree =
    let values = map snd $ traitTimeline predicate extractor tree
    in if null values
       then (0, 0)
       else (minimum values, maximum values)

-- | Calculate mean trait value
traitMean :: (Trait -> Bool) -> (Trait -> Double) -> PhyloTree -> Double
traitMean predicate extractor tree =
    let values = map snd $ traitTimeline predicate extractor tree
    in if null values then 0 else average values

-- | Calculate trait variance
traitVariance :: (Trait -> Bool) -> (Trait -> Double) -> PhyloTree -> Double
traitVariance predicate extractor tree =
    let values = map snd $ traitTimeline predicate extractor tree
    in variance values

-- * Evolutionary Metrics

-- | Calculate trait disparity (morphological diversity)
traitDisparity :: (Trait -> Bool) -> (Trait -> Double) -> PhyloTree -> Double
traitDisparity predicate extractor tree =
    let values = map snd $ traitTimeline predicate extractor tree
    in variance values

-- | Calculate evolutionary distance based on trait differences
evolutionaryDistance :: (Trait -> Bool) -> (Trait -> Double) -> Species -> Species -> Double
evolutionaryDistance predicate extractor sp1 sp2 =
    let v1 = findTraitValue predicate extractor sp1
        v2 = findTraitValue predicate extractor sp2
    in abs (v1 - v2)

-- | Calculate phenotypic diversity across all traits
phenotypicDiversity :: PhyloTree -> Double
phenotypicDiversity tree =
    let ccVar = traitVariance isCranialCapacity getCranialCapacity tree
        bdVar = traitVariance isBipedalism getBipedalism tree
        bmVar = traitVariance isBodyMass getBodyMass tree
    in (ccVar + bdVar * 1000 + bmVar) / 3
  where
    isCranialCapacity (CranialCapacity _) = True
    isCranialCapacity _ = False

    isBipedalism (BipedalismDegree _) = True
    isBipedalism _ = False

    isBodyMass (BodyMass _) = True
    isBodyMass _ = False

    getCranialCapacity (CranialCapacity cc) = cc
    getCranialCapacity _ = 0

    getBipedalism (BipedalismDegree bd) = bd
    getBipedalism _ = 0

    getBodyMass (BodyMass bm) = bm
    getBodyMass _ = 0

-- * Helper Functions

findTraitValue :: (Trait -> Bool) -> (Trait -> Double) -> Species -> Double
findTraitValue predicate extractor species =
    case find predicate (traits species) of
        Just trait -> extractor trait
        Nothing -> 0.0

variance :: [Double] -> Double
variance [] = 0
variance values =
    let mean' = average values
        squaredDiffs = map (\x -> (x - mean') ** 2) values
    in average squaredDiffs

average :: [Double] -> Double
average [] = 0
average xs = sum xs / fromIntegral (length xs)
