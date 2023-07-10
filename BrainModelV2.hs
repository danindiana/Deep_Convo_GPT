-- File: BrainModel.hs
module BrainModel where

import qualified Data.Map as Map

-- Represent the brain as a Map from Int (synapse) to Double (strength)
type Brain = Map.Map Int Double

-- Initialize an empty brain
emptyBrain :: Brain
emptyBrain = Map.empty

-- Function to encode sensory input with attention
encode :: Brain -> [(Int, Double)] -> Double -> Brain
encode brain inputs attention = foldl encodeOne brain inputs
  where
    encodeOne :: Brain -> (Int, Double) -> Brain
    encodeOne b (i, val) = Map.insertWith (+) i (val * attention) b

-- Function to retrieve information
retrieve :: Brain -> Int -> Maybe Double
retrieve = Map.lookup
