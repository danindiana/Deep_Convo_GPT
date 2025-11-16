```mermaid
flowchart TD
    subgraph S1 [Input]
        P[Prompt p]
        G[Ground Truth g]
    end

    S1 --> R[Rollout Generation<br>via Policy Model π_θ]

    subgraph R
        direction LR
        R1[Generate Completion 1<br>t₁, a₁]
        R2[Generate Completion 2<br>t₂, a₂]
        R3[Generate Completion ...]
        R4[Generate Completion G<br>t_G, a_G]
    end

    R --> PG[Parse into Reasoning Graph<br>G = V, E for each completion]

    PG --> LBR[Lipschitz Bottom-Up Reward Calculation]
    
    subgraph LBR
        direction TB
        SG[For each reasoning step<br>in the graph]
        SG --> PERT[Generate Minimal<br>Semantic Perturbations]
        PERT --> EG[Embed Original &<br>Perturbed Premises ϕv_p]
        EG --> FG[Generate New Conclusion<br>v'_c from Perturbed Premises]
        FG --> EG2[Embed Original &<br>New Conclusion ϕv_c, ϕv'_c]
        EG2 --> DIST[Calculate Distances<br>d_in, d_out]
        DIST --> L[Compute Local<br>Lipschitz Ratio L = d_out / d_in]
        L --> AR[Aggregate per-step L<br>into Trace Robustness Score R_robust]
    end

    LBR --> TR[Top-Down Reward Calculation]
    
    subgraph TR
        direction LR
        PR[Compute Reasoning Perplexity P_r]
        PR --> RR[Compute Reasoning Reward R_r<br>via ranking]
        ER[Compute Efficiency Reward R_e]
        FR[Compute Format Reward R_f]
    end

    TR --> CR[Combine Rewards]
    LBR --> CR

    CR --> FRW[Final Reward Calculation<br>R_total = w_f·R_f + 𝕀R_f·w_r·R_r + w_e·R_e + w_robust·R_robust]

    FRW --> GRPO[GRPO Policy Optimization<br>Calculate Advantages & Update π_θ]

    GRPO --> Sync[Policy-Proxy Synchronization<br>π_p ← α·π_p + 1-α·π_θ<br>every T_sync steps]

    Sync --> Final[Updated Policy Model π_θ]
    
    %% Style Links
    linkStyle 0,1,2,3,4,5,6,7,8,9,10,11,12 stroke:blue,stroke-width:2px;
    linkStyle 13 stroke:red,stroke-width:3px;
```
