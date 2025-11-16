{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

{-|
Module      : PhyloTree
Description : Type-safe phylogenetic tree data structures for early hominids
Copyright   : (c) 2025
License     : MIT
Maintainer  : phylo@example.com

This module provides core data structures for representing phylogenetic
trees with strong type safety guarantees. The implementation uses algebraic
data types to ensure tree validity at compile time.
-}

module PhyloTree
    ( -- * Core Types
      TimeScale(..)
    , Genus(..)
    , Trait(..)
    , Species(..)
    , PhyloTree(..)

    -- * Tree Construction
    , leaf
    , node
    , buildHominidTree

    -- * Basic Operations
    , getSpecies
    , getChildren
    , treeDepth
    , treeSize

    -- * Traversals
    , dfs
    , bfs
    , preorder
    , postorder

    -- * Queries
    , findSpecies
    , findByGenus
    , getAllSpecies
    , getLeaves

    -- * Validation
    , isValid
    , validateTimeRange

    ) where

import GHC.Generics (Generic)
import Data.List (find, sortBy)
import Data.Maybe (mapMaybe)

-- | Time scale in millions of years ago (MYA)
newtype TimeScale = MYA Double
    deriving (Eq, Ord, Show, Generic)

instance Num TimeScale where
    (MYA a) + (MYA b) = MYA (a + b)
    (MYA a) - (MYA b) = MYA (a - b)
    (MYA a) * (MYA b) = MYA (a * b)
    abs (MYA a) = MYA (abs a)
    signum (MYA a) = MYA (signum a)
    fromInteger n = MYA (fromInteger n)

-- | Taxonomic genus classification
data Genus
    = Sahelanthropus
    | Orrorin
    | Ardipithecus
    | Australopithecus
    | Paranthropus
    | Kenyanthropus
    | Homo
    deriving (Eq, Show, Ord, Enum, Bounded, Generic)

-- | Morphological and behavioral traits
data Trait
    = CranialCapacity Double        -- in cubic centimeters
    | BipedalismDegree Double       -- 0.0 (quadrupedal) to 1.0 (fully bipedal)
    | ToolUse Bool                  -- evidence of tool manufacture/use
    | BodyMass Double               -- in kilograms
    | CanineSize Double             -- relative size
    | BrainToBodyRatio Double       -- encephalization quotient
    deriving (Eq, Show, Generic)

-- | Species with full taxonomic and morphological information
data Species = Species
    { speciesName :: String                    -- Full species name
    , genus :: Genus                           -- Taxonomic genus
    , timeRange :: (TimeScale, TimeScale)      -- (earliest, latest) appearance
    , traits :: [Trait]                        -- Morphological/behavioral traits
    , location :: String                       -- Geographic location
    , confidence :: Double                     -- Classification confidence (0-1)
    } deriving (Eq, Show, Generic)

-- | Phylogenetic tree structure
-- Each node contains a species and may have descendant lineages
data PhyloTree
    = Leaf Species                  -- Terminal node (extinct with no descendants)
    | Node Species [PhyloTree]      -- Internal node with descendants
    deriving (Eq, Show, Generic)

-- * Smart Constructors

-- | Create a leaf node
leaf :: Species -> PhyloTree
leaf = Leaf

-- | Create an internal node with children
-- Validates that children exist
node :: Species -> [PhyloTree] -> PhyloTree
node s [] = Leaf s  -- No children means it's actually a leaf
node s children = Node s children

-- * Tree Accessors

-- | Extract species from any tree node
getSpecies :: PhyloTree -> Species
getSpecies (Leaf s) = s
getSpecies (Node s _) = s

-- | Get immediate children of a node
getChildren :: PhyloTree -> [PhyloTree]
getChildren (Leaf _) = []
getChildren (Node _ children) = children

-- | Calculate maximum depth of the tree
treeDepth :: PhyloTree -> Int
treeDepth (Leaf _) = 1
treeDepth (Node _ children) = 1 + maximum (0 : map treeDepth children)

-- | Count total number of nodes
treeSize :: PhyloTree -> Int
treeSize (Leaf _) = 1
treeSize (Node _ children) = 1 + sum (map treeSize children)

-- * Traversal Algorithms

-- | Depth-first search traversal
dfs :: PhyloTree -> [Species]
dfs (Leaf s) = [s]
dfs (Node s children) = s : concatMap dfs children

-- | Breadth-first search traversal (returns levels)
bfs :: PhyloTree -> [[Species]]
bfs tree = bfs' [tree]
  where
    bfs' :: [PhyloTree] -> [[Species]]
    bfs' [] = []
    bfs' level =
        map getSpecies level : bfs' (concatMap getChildren level)

-- | Preorder traversal (parent before children)
preorder :: PhyloTree -> [Species]
preorder = dfs  -- Same as DFS for trees

-- | Postorder traversal (children before parent)
postorder :: PhyloTree -> [Species]
postorder (Leaf s) = [s]
postorder (Node s children) = concatMap postorder children ++ [s]

-- * Query Functions

-- | Find a species by name
findSpecies :: String -> PhyloTree -> Maybe Species
findSpecies name tree = find (\s -> speciesName s == name) (dfs tree)

-- | Find all species of a given genus
findByGenus :: Genus -> PhyloTree -> [Species]
findByGenus g tree = filter (\s -> genus s == g) (dfs tree)

-- | Get all species in the tree
getAllSpecies :: PhyloTree -> [Species]
getAllSpecies = dfs

-- | Get only leaf species (terminal taxa)
getLeaves :: PhyloTree -> [Species]
getLeaves (Leaf s) = [s]
getLeaves (Node _ children) = concatMap getLeaves children

-- * Validation Functions

-- | Validate tree structure and temporal consistency
isValid :: PhyloTree -> Bool
isValid tree =
    validateTimeRange tree &&
    validateStructure tree &&
    validateTraits tree

-- | Check that time ranges are consistent (ancestors predate descendants)
validateTimeRange :: PhyloTree -> Bool
validateTimeRange (Leaf _) = True
validateTimeRange (Node parent children) =
    all (parentPrecedesChild parent) children &&
    all validateTimeRange children
  where
    parentPrecedesChild :: Species -> PhyloTree -> Bool
    parentPrecedesChild p childTree =
        let child = getSpecies childTree
            (MYA pEarliest, _) = timeRange p
            (MYA cEarliest, _) = timeRange child
        in pEarliest >= cEarliest  -- Parent must appear before or with child

-- | Validate tree structure (no empty children lists for Node)
validateStructure :: PhyloTree -> Bool
validateStructure (Leaf _) = True
validateStructure (Node _ []) = False  -- Nodes must have children
validateStructure (Node _ children) = all validateStructure children

-- | Validate that traits are reasonable
validateTraits :: PhyloTree -> Bool
validateTraits tree = all validSpeciesTraits (dfs tree)
  where
    validSpeciesTraits :: Species -> Bool
    validSpeciesTraits s = all validTrait (traits s)

    validTrait :: Trait -> Bool
    validTrait (CranialCapacity cc) = cc > 0 && cc < 3000
    validTrait (BipedalismDegree bd) = bd >= 0 && bd <= 1
    validTrait (ToolUse _) = True
    validTrait (BodyMass bm) = bm > 0 && bm < 500
    validTrait (CanineSize cs) = cs >= 0
    validTrait (BrainToBodyRatio eq) = eq >= 0

-- * Tree Construction - Early Hominid Phylogeny

-- | Build the complete early hominid phylogenetic tree
buildHominidTree :: PhyloTree
buildHominidTree =
    node earliestHominid
        [ node sahelanthropus []
        , node orrorin []
        , node ardipithecusRamidus
            [ leaf ardipithecusKadabba ]
        , node australopithecusAnamensis
            [ node australopithecusAfarensis
                [ node australopithecusAfricanus
                    [ leaf australopithecusSediba ]
                , leaf australopithecusGarhi
                , node paranthropusAethiopicus
                    [ leaf paranthropusBoisei
                    , leaf paranthropusRobustus
                    ]
                , leaf kenyanthropusPlatyops
                ]
            , leaf australopithecusDeyiremeda
            ]
        , node homoHabilis
            [ leaf homoRudolfensis
            , node homoErgaster
                [ node homoErectus
                    [ node homoHeidelbergensis
                        [ leaf homoNeanderthalensis
                        , leaf homoSapiens
                        ]
                    ]
                ]
            ]
        ]

-- Species Definitions

earliestHominid :: Species
earliestHominid = Species
    { speciesName = "Earliest Hominids"
    , genus = Ardipithecus
    , timeRange = (MYA 7.0, MYA 4.0)
    , traits = [BipedalismDegree 0.3, BodyMass 35]
    , location = "East Africa"
    , confidence = 0.5
    }

sahelanthropus :: Species
sahelanthropus = Species
    { speciesName = "Sahelanthropus tchadensis"
    , genus = Sahelanthropus
    , timeRange = (MYA 7.0, MYA 6.0)
    , traits =
        [ CranialCapacity 360
        , BipedalismDegree 0.4
        , BodyMass 36
        , CanineSize 0.8
        ]
    , location = "Chad"
    , confidence = 0.7
    }

orrorin :: Species
orrorin = Species
    { speciesName = "Orrorin tugenensis"
    , genus = Orrorin
    , timeRange = (MYA 6.0, MYA 5.8)
    , traits =
        [ BipedalismDegree 0.5
        , BodyMass 35
        , CanineSize 0.9
        ]
    , location = "Kenya"
    , confidence = 0.75
    }

ardipithecusRamidus :: Species
ardipithecusRamidus = Species
    { speciesName = "Ardipithecus ramidus"
    , genus = Ardipithecus
    , timeRange = (MYA 4.4, MYA 4.3)
    , traits =
        [ CranialCapacity 350
        , BipedalismDegree 0.6
        , BodyMass 50
        , CanineSize 0.85
        ]
    , location = "Ethiopia"
    , confidence = 0.85
    }

ardipithecusKadabba :: Species
ardipithecusKadabba = Species
    { speciesName = "Ardipithecus kadabba"
    , genus = Ardipithecus
    , timeRange = (MYA 5.8, MYA 5.2)
    , traits =
        [ BipedalismDegree 0.5
        , BodyMass 45
        ]
    , location = "Ethiopia"
    , confidence = 0.65
    }

australopithecusAnamensis :: Species
australopithecusAnamensis = Species
    { speciesName = "Australopithecus anamensis"
    , genus = Australopithecus
    , timeRange = (MYA 4.2, MYA 3.9)
    , traits =
        [ CranialCapacity 370
        , BipedalismDegree 0.7
        , BodyMass 55
        ]
    , location = "Kenya, Ethiopia"
    , confidence = 0.8
    }

australopithecusAfarensis :: Species
australopithecusAfarensis = Species
    { speciesName = "Australopithecus afarensis"
    , genus = Australopithecus
    , timeRange = (MYA 3.9, MYA 2.9)
    , traits =
        [ CranialCapacity 450
        , BipedalismDegree 0.8
        , BodyMass 45
        , CanineSize 0.7
        , BrainToBodyRatio 0.01
        ]
    , location = "Ethiopia, Kenya, Tanzania"
    , confidence = 0.95
    }

australopithecusAfricanus :: Species
australopithecusAfricanus = Species
    { speciesName = "Australopithecus africanus"
    , genus = Australopithecus
    , timeRange = (MYA 3.3, MYA 2.1)
    , traits =
        [ CranialCapacity 480
        , BipedalismDegree 0.85
        , BodyMass 41
        , BrainToBodyRatio 0.012
        ]
    , location = "South Africa"
    , confidence = 0.9
    }

australopithecusSediba :: Species
australopithecusSediba = Species
    { speciesName = "Australopithecus sediba"
    , genus = Australopithecus
    , timeRange = (MYA 2.0, MYA 1.8)
    , traits =
        [ CranialCapacity 420
        , BipedalismDegree 0.8
        , BodyMass 33
        ]
    , location = "South Africa"
    , confidence = 0.85
    }

australopithecusGarhi :: Species
australopithecusGarhi = Species
    { speciesName = "Australopithecus garhi"
    , genus = Australopithecus
    , timeRange = (MYA 2.5, MYA 2.5)
    , traits =
        [ CranialCapacity 450
        , ToolUse True
        , BodyMass 49
        ]
    , location = "Ethiopia"
    , confidence = 0.7
    }

australopithecusDeyiremeda :: Species
australopithecusDeyiremeda = Species
    { speciesName = "Australopithecus deyiremeda"
    , genus = Australopithecus
    , timeRange = (MYA 3.5, MYA 3.3)
    , traits =
        [ BipedalismDegree 0.75
        , BodyMass 47
        ]
    , location = "Ethiopia"
    , confidence = 0.65
    }

paranthropusAethiopicus :: Species
paranthropusAethiopicus = Species
    { speciesName = "Paranthropus aethiopicus"
    , genus = Paranthropus
    , timeRange = (MYA 2.7, MYA 2.3)
    , traits =
        [ CranialCapacity 410
        , BipedalismDegree 0.85
        , BodyMass 47
        ]
    , location = "Ethiopia, Kenya"
    , confidence = 0.75
    }

paranthropusBoisei :: Species
paranthropusBoisei = Species
    { speciesName = "Paranthropus boisei"
    , genus = Paranthropus
    , timeRange = (MYA 2.3, MYA 1.2)
    , traits =
        [ CranialCapacity 510
        , BipedalismDegree 0.9
        , BodyMass 49
        , ToolUse True
        ]
    , location = "Ethiopia, Kenya, Tanzania, Malawi"
    , confidence = 0.9
    }

paranthropusRobustus :: Species
paranthropusRobustus = Species
    { speciesName = "Paranthropus robustus"
    , genus = Paranthropus
    , timeRange = (MYA 2.0, MYA 1.2)
    , traits =
        [ CranialCapacity 530
        , BipedalismDegree 0.9
        , BodyMass 54
        , ToolUse True
        ]
    , location = "South Africa"
    , confidence = 0.85
    }

kenyanthropusPlatyops :: Species
kenyanthropusPlatyops = Species
    { speciesName = "Kenyanthropus platyops"
    , genus = Kenyanthropus
    , timeRange = (MYA 3.5, MYA 3.2)
    , traits =
        [ CranialCapacity 450
        , BipedalismDegree 0.8
        ]
    , location = "Kenya"
    , confidence = 0.6
    }

homoHabilis :: Species
homoHabilis = Species
    { speciesName = "Homo habilis"
    , genus = Homo
    , timeRange = (MYA 2.4, MYA 1.4)
    , traits =
        [ CranialCapacity 640
        , BipedalismDegree 0.9
        , ToolUse True
        , BodyMass 52
        , BrainToBodyRatio 0.018
        ]
    , location = "Tanzania, Kenya, Ethiopia, South Africa"
    , confidence = 0.85
    }

homoRudolfensis :: Species
homoRudolfensis = Species
    { speciesName = "Homo rudolfensis"
    , genus = Homo
    , timeRange = (MYA 2.4, MYA 1.8)
    , traits =
        [ CranialCapacity 750
        , BipedalismDegree 0.9
        , ToolUse True
        , BodyMass 60
        ]
    , location = "Kenya"
    , confidence = 0.7
    }

homoErgaster :: Species
homoErgaster = Species
    { speciesName = "Homo ergaster"
    , genus = Homo
    , timeRange = (MYA 1.9, MYA 1.4)
    , traits =
        [ CranialCapacity 850
        , BipedalismDegree 1.0
        , ToolUse True
        , BodyMass 63
        , BrainToBodyRatio 0.02
        ]
    , location = "Kenya, Tanzania, Ethiopia, South Africa"
    , confidence = 0.85
    }

homoErectus :: Species
homoErectus = Species
    { speciesName = "Homo erectus"
    , genus = Homo
    , timeRange = (MYA 1.89, MYA 0.11)
    , traits =
        [ CranialCapacity 950
        , BipedalismDegree 1.0
        , ToolUse True
        , BodyMass 60
        , BrainToBodyRatio 0.022
        ]
    , location = "Africa, Asia"
    , confidence = 0.95
    }

homoHeidelbergensis :: Species
homoHeidelbergensis = Species
    { speciesName = "Homo heidelbergensis"
    , genus = Homo
    , timeRange = (MYA 0.7, MYA 0.2)
    , traits =
        [ CranialCapacity 1250
        , BipedalismDegree 1.0
        , ToolUse True
        , BodyMass 75
        , BrainToBodyRatio 0.025
        ]
    , location = "Africa, Europe"
    , confidence = 0.9
    }

homoNeanderthalensis :: Species
homoNeanderthalensis = Species
    { speciesName = "Homo neanderthalensis"
    , genus = Homo
    , timeRange = (MYA 0.4, MYA 0.04)
    , traits =
        [ CranialCapacity 1450
        , BipedalismDegree 1.0
        , ToolUse True
        , BodyMass 78
        , BrainToBodyRatio 0.028
        ]
    , location = "Europe, Western Asia"
    , confidence = 0.99
    }

homoSapiens :: Species
homoSapiens = Species
    { speciesName = "Homo sapiens"
    , genus = Homo
    , timeRange = (MYA 0.3, MYA 0.0)
    , traits =
        [ CranialCapacity 1400
        , BipedalismDegree 1.0
        , ToolUse True
        , BodyMass 70
        , BrainToBodyRatio 0.029
        ]
    , location = "Worldwide"
    , confidence = 1.0
    }
