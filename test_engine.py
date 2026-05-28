"""
Quick test of the Elliott Wave engine with synthetic + live data.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from datetime import datetime, timedelta
from engine.elliott import analyze, detect_swings, detect_impulse, detect_zigzag, detect_diagonal


def test_with_synthetic_data():
    """Synthetic BTC-like data with a clear 5-wave impulse."""
    print("=" * 60)
    print("TEST 1: Synthetic 5-Wave Uptrend Impulse")
    print("=" * 60)
    
    n = 200
    dates = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
    
    # Build a price series: L-H-L-H-L-H (3 motive waves up)
    # Wave 1: 100 → 120  (+20%)
    # Wave 2: 120 → 110  (-8.3%)
    # Wave 3: 110 → 150  (+36%, extended)
    # Wave 4: 150 → 140  (-6.7%)
    # Wave 5: 140 → 165  (+17.9%)
    
    price = np.zeros(n)
    noise = np.random.normal(0, 0.3, n)
    
    phases = [
        (0, 30, 100, 120),       # Wave 1 up
        (30, 50, 120, 110),       # Wave 2 down
        (50, 90, 110, 150),       # Wave 3 up
        (90, 110, 150, 140),      # Wave 4 down
        (110, 150, 140, 165),     # Wave 5 up
        (150, 200, 165, 165),     # Consolidation
    ]
    
    for start, end, p_start, p_end in phases:
        for i in range(start, end):
            frac = (i - start) / (end - start)
            price[i] = p_start + (p_end - p_start) * frac + noise[i]
    
    highs = np.maximum(price + np.random.uniform(0.5, 2, n), price)
    lows = np.minimum(price - np.random.uniform(0.5, 2, n), price)
    
    result = analyze(highs, lows, dates, sensitivity=5)
    
    print(f"Pattern: {result.pattern_type.value}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Summary: {result.summary}")
    print(f"Waves detected: {len(result.waves)}")
    for w in result.waves:
        print(f"  {w.label}: {w.start_swing.price:.1f} → {w.end_swing.price:.1f} ({w.pct_change:+.1f}%)")
    print(f"Invalidation: {result.invalidation_level}")
    print(f"Details:")
    for d in result.details:
        print(f"  {d}")
    if result.alternate:
        print(f"Alternate: {result.alternate.pattern_type.value} ({result.alternate.confidence:.0%})")
    print()


def test_with_downtrend():
    """Synthetic downtrend with ABC zigzag."""
    print("=" * 60)
    print("TEST 2: Synthetic Downtrend ABC Zigzag")
    print("=" * 60)
    
    n = 200
    dates = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
    
    price = np.zeros(n)
    noise = np.random.normal(0, 0.2, n)
    
    phases = [
        (0, 50, 100, 80),     # Wave A down
        (50, 70, 80, 90),     # Wave B up (50% retrace)
        (70, 120, 90, 65),    # Wave C down (~= A)
        (120, 200, 65, 70),   # Consolidation
    ]
    
    for start, end, p_start, p_end in phases:
        for i in range(start, end):
            frac = (i - start) / (end - start)
            price[i] = p_start + (p_end - p_start) * frac + noise[i]
    
    highs = np.maximum(price + np.random.uniform(0.3, 1.5, n), price)
    lows = np.minimum(price - np.random.uniform(0.3, 1.5, n), price)
    
    result = analyze(highs, lows, dates, sensitivity=5)
    
    print(f"Pattern: {result.pattern_type.value}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"Summary: {result.summary}")
    print(f"Swings found: {len(result.swings)}")
    for w in result.waves:
        print(f"  {w.label}: {w.start_swing.price:.1f} → {w.end_swing.price:.1f} ({w.pct_change:+.1f}%)")
    print()


if __name__ == "__main__":
    test_with_synthetic_data()
    test_with_downtrend()
    print("✅ Engine tests complete!")
