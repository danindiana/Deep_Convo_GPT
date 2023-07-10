∇ Z←Layer;W;B
    W ← ? 10 10 ⍴ 2
    B ← ? 10 ⍴ 2
    Z ← W + B
∇

∇ Z←Activate X;Y
    Y ← 1 ÷ (1 + ¯1 * 2 * X)
    Z ← (Y > 0.5) / Y
∇

∇ Z←Network X;Y;L;I
    I ← 0
    L ← Layer ⍬
Loop: 
    I ← I + 1
    X ← Activate L + X
    → (I < 10) / Loop
    Z ← X
∇
