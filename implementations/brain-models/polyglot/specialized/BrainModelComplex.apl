∇ Z←Layer;W;B
    W ← ? 10 10 ⍴ 2
    B ← ? 10 ⍴ 2
    Z ← W + B
∇

∇ Z←Activate X;Y
    Y ← 1 ÷ (1 + ¯1 * 2 * X)
    Z ← (Y > 0.5) / Y
∇

∇ Learn W;X;Y;η
    η ← 0.1
    Y ← Activate X
    W ← W + η × Y ∘.× X
∇

∇ Z←Network X;Y;L;I;W
    I ← 0
    L ← Layer ⍬
    W ← Layer ⍬
Loop: 
    I ← I + 1
    X ← Activate L + X
    Learn W X L
    → (I < 10) / Loop
    Z ← X
∇
