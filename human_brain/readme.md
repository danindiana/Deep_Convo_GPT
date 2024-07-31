```mermaid
flowchart TB
    A[Pulse Spike Train Speed Metrics] --> A1[Common Metrics]
    A1 --> B1[Firing Rate]
    A1 --> B2[Interspike Interval ISI]
    A1 --> B3[Burst Metrics]
    A1 --> B4[Synchrony Metrics]
    A1 --> B5[Spike Train Metrics]

    B1 --> C1[Mean Firing Rate]
    B1 --> C2[Instantaneous Firing Rate]

    B2 --> D1[Mean ISI]
    B2 --> D2[Coefficient of Variation CV of ISI]

    B3 --> E1[Burst Rate]
    B3 --> E2[Burst Duration]
    B3 --> E3[Intraburst Frequency]

    B4 --> F1[Spike-Time Tiling Coefficient STTC]
    B4 --> F2[Cross-Correlation]

    B5 --> G1[Victor-Purpura Distance]
    B5 --> G2[Van Rossum Distance]
    B5 --> G3[Modulus-Metric and Max-Metric]

    A --> H[Choosing the Right Metric]
    H --> I[Interest in Overall Firing Rate --> Mean Firing Rate]
    H --> J[Interest in Temporal Patterns and Variability --> ISI CV or Burst Metrics]
```


Python Dict
```
pulse_spike_train_speed_metrics = {
    "PulseSpikeTrainSpeedMetrics": {
        "CommonMetrics": {
            "FiringRate": ["MeanFiringRate", "InstantaneousFiringRate"],
            "InterspikeInterval": ["MeanISI", "CoefficientOfVariationISI"],
            "BurstMetrics": ["BurstRate", "BurstDuration", "IntraburstFrequency"],
            "SynchronyMetrics": ["SpikeTimeTilingCoefficientSTTC", "CrossCorrelation"],
            "SpikeTrainMetrics": ["VictorPurpuraDistance", "VanRossumDistance", "ModulusMetric", "MaxMetric"]
        },
        "ChoosingTheRightMetric": {
            "InterestInOverallFiringRate": "MeanFiringRate",
            "InterestInTemporalPatternsAndVariability": ["ISICV", "BurstMetrics"]
        }
    }
}
```

# Pulse Spike Train Speed Metrics

## Common Metrics
- Firing Rate
  - Mean Firing Rate
  - Instantaneous Firing Rate
- Interspike Interval (ISI)
  - Mean ISI
  - Coefficient of Variation (CV) of ISI
- Burst Metrics
  - Burst Rate
  - Burst Duration
  - Intraburst Frequency
- Synchrony Metrics
  - Spike-Time Tiling Coefficient (STTC)
  - Cross-Correlation
- Spike Train Metrics
  - Victor-Purpura Distance
  - Van Rossum Distance
  - Modulus-Metric and Max-Metric

## Choosing the Right Metric
- Interest in Overall Firing Rate
  - Mean Firing Rate
- Interest in Temporal Patterns and Variability
  - ISI CV
  - Burst Metrics
