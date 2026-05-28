"""
Elliott Wave Engine — core pattern detection and wave labeling.

Detects:
  - 5-wave Impulses (1-2-3-4-5)
  - ABC Zigzags (5-3-5)
  - ABC Flat corrections (3-3-5)
  - Ending/Leading Diagonals
  - Contracting/Expanding Triangles
  - WXY Complex Corrections

Rules enforced:
  - Wave 2 never retraces >100% of Wave 1
  - Wave 3 is never the shortest among 1, 3, 5
  - Wave 4 never enters Wave 1's price territory (impulse)
  - Wave 4 DOES enter Wave 1 territory (diagonal)
  - Fib relationships: 38.2%, 50%, 61.8%, 78.6%, 100%, 127.2%, 161.8%
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class WaveDegree(Enum):
    """Elliott Wave degrees."""
    SUBMICRO = "Submicro"
    MICRO = "Micro"
    SUBMINUETTE = "Subminuette"
    MINUETTE = "Minuette"
    MINUTE = "Minute"
    MINOR = "Minor"
    INTERMEDIATE = "Intermediate"
    PRIMARY = "Primary"
    CYCLE = "Cycle"
    SUPER_CYCLE = "Supercycle"


class PatternType(Enum):
    IMPULSE = "5-Wave Impulse"
    DIAGONAL_ENDING = "Ending Diagonal"
    DIAGONAL_LEADING = "Leading Expanding Diagonal"
    ZIGZAG = "ABC Zigzag (5-3-5)"
    FLAT = "ABC Flat (3-3-5)"
    TRIANGLE_CONTRACTING = "Contracting Triangle"
    TRIANGLE_EXPANDING = "Expanding Triangle"
    COMPLEX_WXY = "WXY Complex Correction"
    UNKNOWN = "Pattern Uncertain"


@dataclass
class Swing:
    """A price swing — high or low point."""
    index: int
    price: float
    timestamp: any
    is_high: bool  # True = swing high, False = swing low


@dataclass
class Wave:
    """A single Elliott Wave."""
    label: str
    start_swing: Swing
    end_swing: Swing
    degree: WaveDegree = WaveDegree.MINOR
    price_change: float = 0.0
    pct_change: float = 0.0

    def __post_init__(self):
        self.price_change = self.end_swing.price - self.start_swing.price
        if self.start_swing.price != 0:
            self.pct_change = (self.price_change / self.start_swing.price) * 100


@dataclass
class FibonacciLevels:
    """Fibonacci retracement and extension levels from a swing."""
    high: float
    low: float
    diff: float = 0.0
    retrace_382: float = 0.0
    retrace_500: float = 0.0
    retrace_618: float = 0.0
    retrace_786: float = 0.0
    extension_1272: float = 0.0
    extension_1618: float = 0.0
    extension_2618: float = 0.0

    def __post_init__(self):
        self.diff = self.high - self.low
        if self.high > self.low:  # downtrend fib
            self.retrace_382 = self.low + self.diff * 0.382
            self.retrace_500 = self.low + self.diff * 0.5
            self.retrace_618 = self.low + self.diff * 0.618
            self.retrace_786 = self.low + self.diff * 0.786
            self.extension_1272 = self.low - self.diff * 0.272
            self.extension_1618 = self.low - self.diff * 0.618
            self.extension_2618 = self.low - self.diff * 1.618
        else:  # uptrend fib
            self.retrace_382 = self.high - abs(self.diff) * 0.382
            self.retrace_500 = self.high - abs(self.diff) * 0.5
            self.retrace_618 = self.high - abs(self.diff) * 0.618
            self.retrace_786 = self.high - abs(self.diff) * 0.786
            self.extension_1272 = self.high + abs(self.diff) * 0.272
            self.extension_1618 = self.high + abs(self.diff) * 1.618
            self.extension_2618 = self.high + abs(self.diff) * 2.618


@dataclass
class PatternResult:
    """Complete Elliott Wave pattern analysis result."""
    pattern_type: PatternType
    confidence: float  # 0.0 to 1.0
    waves: list[Wave] = field(default_factory=list)
    swings: list[Swing] = field(default_factory=list)
    fib_levels: Optional[FibonacciLevels] = None
    invalidation_level: Optional[float] = None
    confirmation_level: Optional[float] = None
    alternate: Optional['PatternResult'] = None
    summary: str = ""
    details: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# SWING DETECTION
# ═══════════════════════════════════════════════════════════════════════

def detect_swings(highs, lows, timestamps, sensitivity=5):
    """
    Find swing highs and lows in price data.
    
    sensitivity: how many bars must surround a swing (higher = fewer, larger swings)
    """
    n = len(highs)
    swings = []
    
    for i in range(sensitivity, n - sensitivity):
        # Swing high check
        is_high = True
        for j in range(1, sensitivity + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_high = False
                break
        
        if is_high:
            swings.append(Swing(
                index=i, price=highs[i], timestamp=timestamps[i], is_high=True
            ))
            continue
        
        # Swing low check
        is_low = True
        for j in range(1, sensitivity + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_low = False
                break
        
        if is_low:
            swings.append(Swing(
                index=i, price=lows[i], timestamp=timestamps[i], is_high=False
            ))
    
    # Sort by index
    swings.sort(key=lambda s: s.index)
    return swings


# ═══════════════════════════════════════════════════════════════════════
# PATTERN DETECTION
# ═══════════════════════════════════════════════════════════════════════

def _alternates_high_low(swings: list[Swing]) -> bool:
    """Verify swings alternate high-low-high-low."""
    for i in range(1, len(swings)):
        if swings[i].is_high == swings[i - 1].is_high:
            return False
    return True


def _make_wave(label, s1, s2, degree=WaveDegree.MINOR):
    return Wave(label=label, start_swing=s1, end_swing=s2, degree=degree)


def _make_fib(high, low):
    return FibonacciLevels(high=high, low=low)


def detect_impulse(swings: list[Swing]) -> Optional[PatternResult]:
    """
    Try to find a 5-wave impulse pattern in the swings.
    
    Expected swing sequence: H-L-H-L-H-L  (starting from high, 3 motive waves down)
    or:                      L-H-L-H-L-H  (starting from low, 3 motive waves up)
    
    Rules:
      1. Wave 2 never retraces >100% of Wave 1
      2. Wave 3 is never the shortest
      3. Wave 4 never enters Wave 1's territory
    """
    if len(swings) < 6:
        return None
    
    # Try all possible 6-swing windows
    best = None
    best_confidence = 0.0
    
    for start in range(len(swings) - 5):
        s = swings[start:start + 6]
        if not _alternates_high_low(s):
            continue
        
        # Determine direction: if first is low, it's an uptrend impulse
        uptrend = not s[0].is_high
        
        # Build waves
        waves = []
        labels = ["(1)", "(2)", "(3)", "(4)", "(5)"] if uptrend else ["1", "2", "3", "4", "5"]
        
        for i in range(5):
            waves.append(_make_wave(labels[i], s[i], s[i + 1]))
        
        confidence = 1.0
        details = []
        
        w1, w2, w3, w4, w5 = waves
        
        # Rule 1: Wave 2 never retraces >100% of Wave 1
        w2_retrace = abs(w2.price_change / w1.price_change) if w1.price_change != 0 else 999
        if w2_retrace > 1.0:
            confidence -= 0.4
            details.append(f"⚠ Wave 2 retraces {w2_retrace:.0%} of Wave 1 (max 100%)")
        elif w2_retrace > 0.8:
            details.append(f"✓ Wave 2 retraces {w2_retrace:.0%} of Wave 1 (deep but valid)")
        else:
            details.append(f"✓ Wave 2 retraces {w2_retrace:.0%} of Wave 1")
        
        # Rule 2: Wave 3 is never the shortest among 1, 3, 5
        w_lengths = [abs(w.price_change) for w in [w1, w3, w5]]
        if abs(w3.price_change) <= min(w_lengths) and min(w_lengths) > 0:
            confidence -= 0.5
            details.append("⚠ Wave 3 is the shortest (violates EW rule)")
        elif abs(w3.price_change) >= max(w_lengths):
            details.append("✓ Wave 3 is extended (strongest wave)")
        else:
            details.append("✓ Wave 3 passes length rule")
        
        # Rule 3: Wave 4 never enters Wave 1 territory
        if uptrend:
            # Uptrend: Wave 4 low must stay above Wave 1 high
            wave1_top = s[1].price if s[1].is_high else s[0].price
            wave4_bottom = s[4].price if not s[4].is_high else s[3].price
            if wave4_bottom <= wave1_top:
                confidence -= 0.4
                details.append(f"⚠ Wave 4 enters Wave 1 territory (overlap)")
            else:
                details.append("✓ Wave 4 stays out of Wave 1 territory")
        else:
            # Downtrend: Wave 4 high must stay below Wave 1 low
            wave1_bottom = s[1].price if not s[1].is_high else s[0].price
            wave4_top = s[4].price if s[4].is_high else s[3].price
            if wave4_top >= wave1_bottom:
                confidence -= 0.4
                details.append(f"⚠ Wave 4 enters Wave 1 territory (overlap)")
            else:
                details.append("✓ Wave 4 stays out of Wave 1 territory")
        
        # Fib relationships
        w3_vs_w1 = abs(w3.price_change / w1.price_change) if w1.price_change != 0 else 0
        if 1.5 < w3_vs_w1 < 2.0:
            details.append(f"✓ Wave 3 = {w3_vs_w1:.1f}x Wave 1 (common 1.618 extension)")
            confidence += 0.1
        
        confidence = max(0.0, min(1.0, confidence))
        
        if confidence > best_confidence:
            # Build fib from start to end of wave 3
            if uptrend:
                fib = _make_fib(s[-1].price, s[0].price)
            else:
                fib = _make_fib(s[0].price, s[-1].price)
            
            # Invalidation = Wave 2 extreme
            invalidation = s[2].price if not uptrend else s[1].price
            
            # Confirmation = break of wave 4 extreme
            confirmation = s[4].price if uptrend else s[3].price
            
            best_confidence = confidence
            best = PatternResult(
                pattern_type=PatternType.IMPULSE,
                confidence=confidence,
                waves=waves,
                swings=s,
                fib_levels=fib,
                invalidation_level=invalidation,
                confirmation_level=confirmation,
                details=details,
                summary=f"{'Bullish' if uptrend else 'Bearish'} 5-Wave Impulse "
                        f"({'extended Wave 3' if abs(w3_vs_w1) > 1.5 else 'standard'})"
            )
    
    return best


def detect_zigzag(swings: list[Swing]) -> Optional[PatternResult]:
    """
    Try to find an ABC zigzag correction.
    
    Expected: 5-3-5 structure → A(5 waves down), B(3 waves up), C(5 waves down)
    But at the swing level, we look for: H-L-H-L  (3 swings = A-B-C)
    
    Rules:
      - Wave B retraces 50-78.6% of A
      - Wave C often = Wave A or 1.618 * Wave A
    """
    if len(swings) < 4:
        return None
    
    best = None
    best_confidence = 0.0
    
    for start in range(len(swings) - 3):
        s = swings[start:start + 4]
        if not _alternates_high_low(s):
            continue
        
        # Determine direction
        downtrend = s[0].is_high  # Starts from high, moving down
        
        wA = _make_wave("(A)", s[0], s[1])
        wB = _make_wave("(B)", s[1], s[2])
        wC = _make_wave("(C)", s[2], s[3])
        waves = [wA, wB, wC]
        
        confidence = 0.7
        details = []
        
        # B should retrace 50-78.6% of A
        retrace = abs(wB.price_change / wA.price_change) if wA.price_change != 0 else 0
        if 0.5 <= retrace <= 0.786:
            confidence += 0.15
            details.append(f"✓ Wave B retraces {retrace:.0%} of A (ideal 50-78.6%)")
        elif retrace < 1.0:
            details.append(f"⚠ Wave B retraces {retrace:.0%} of A (outside ideal range)")
            confidence -= 0.1
        else:
            confidence -= 0.3
            details.append(f"⚠ Wave B exceeds 100% of A (might be impulse)")
        
        # C often = A or 1.618 * A
        c_vs_a = abs(wC.price_change / wA.price_change) if wA.price_change != 0 else 0
        if 0.9 <= c_vs_a <= 1.1:
            details.append(f"✓ Wave C ≈ Wave A (C = {c_vs_a:.1f}x A)")
            confidence += 0.1
        elif 1.5 <= c_vs_a <= 1.7:
            details.append(f"✓ Wave C = {c_vs_a:.2f}x A (1.618 extension)")
            confidence += 0.1
        else:
            details.append(f"Wave C = {c_vs_a:.1f}x A")
        
        confidence = max(0.0, min(1.0, confidence))
        
        if confidence > best_confidence:
            if downtrend:
                fib = _make_fib(s[0].price, s[-1].price)
            else:
                fib = _make_fib(s[-1].price, s[0].price)
            
            invalidation = s[0].price  # Beyond start of A
            best_confidence = confidence
            best = PatternResult(
                pattern_type=PatternType.ZIGZAG,
                confidence=confidence,
                waves=waves,
                swings=s,
                fib_levels=fib,
                invalidation_level=invalidation,
                details=details,
                summary=f"ABC Zigzag — Wave C = {c_vs_a:.1f}x Wave A"
            )
    
    return best


def detect_diagonal(swings: list[Swing]) -> Optional[PatternResult]:
    """
    Try to find an ending/leading diagonal.
    
    Diagonals have overlapping waves (wave 4 enters wave 1 territory)
    and a converging or expanding wedge shape.
    
    For a contracting diagonal: each wave is smaller than the previous
    For an expanding diagonal: each wave is larger than the previous
    """
    if len(swings) < 6:
        return None
    
    best = None
    best_confidence = 0.0
    
    for start in range(len(swings) - 5):
        s = swings[start:start + 6]
        if not _alternates_high_low(s):
            continue
        
        uptrend = not s[0].is_high
        
        waves = []
        labels = ["1", "2", "3", "4", "5"]
        for i in range(5):
            waves.append(_make_wave(labels[i], s[i], s[i + 1]))
        
        confidence = 0.5
        details = []
        
        w1, w2, w3, w4, w5 = waves
        lengths = [abs(w.price_change) for w in waves]
        
        # Diagonal key: Wave 4 overlaps Wave 1
        if uptrend:
            wave1_top = s[1].price if s[1].is_high else s[0].price
            wave4_bottom = s[4].price if not s[4].is_high else s[3].price
            overlap = wave4_bottom <= wave1_top
        else:
            wave1_bottom = s[1].price if not s[1].is_high else s[0].price
            wave4_top = s[4].price if s[4].is_high else s[3].price
            overlap = wave4_top >= wave1_bottom
        
        if not overlap:
            confidence -= 0.3
            details.append("⚠ No wave 1/4 overlap (required for diagonal)")
        else:
            details.append("✓ Wave 4 overlaps Wave 1 (diagonal characteristic)")
            confidence += 0.1
        
        # Check if contracting or expanding
        if all(lengths[i] < lengths[i - 1] for i in range(1, 5)):
            details.append("✓ Contracting wedge — waves getting smaller")
            confidence += 0.15
            pattern_type = PatternType.DIAGONAL_ENDING
        elif all(lengths[i] > lengths[i - 1] for i in range(1, 5)):
            details.append("✓ Expanding wedge — waves getting larger")
            confidence += 0.15
            pattern_type = PatternType.DIAGONAL_LEADING
        else:
            details.append("⚠ Waves not clearly contracting or expanding")
            pattern_type = PatternType.DIAGONAL_ENDING
        
        confidence = max(0.0, min(1.0, confidence))
        
        if confidence > best_confidence:
            if uptrend:
                fib = _make_fib(s[-1].price, s[0].price)
            else:
                fib = _make_fib(s[0].price, s[-1].price)
            
            invalidation = s[0].price
            best_confidence = confidence
            best = PatternResult(
                pattern_type=pattern_type,
                confidence=confidence,
                waves=waves,
                swings=s,
                fib_levels=fib,
                invalidation_level=invalidation,
                details=details,
                summary=f"{'Bullish' if uptrend else 'Bearish'} {pattern_type.value}"
            )
    
    return best


def detect_triangle(swings: list[Swing]) -> Optional[PatternResult]:
    """
    Try to find a contracting/expanding triangle (A-B-C-D-E).
    
    Triangles have 5 waves labeled A-B-C-D-E, each subdividing into 3 waves.
    At swing level: 6 consecutive swings that converge.
    """
    if len(swings) < 6:
        return None
    
    best = None
    best_confidence = 0.0
    
    for start in range(len(swings) - 5):
        s = swings[start:start + 6]
        if not _alternates_high_low(s):
            continue
        
        waves = []
        labels = ["A", "B", "C", "D", "E"]
        for i in range(5):
            waves.append(_make_wave(labels[i], s[i], s[i + 1]))
        
        confidence = 0.5
        details = []
        
        lengths = [abs(w.price_change) for w in waves]
        
        # Triangle: price range should contract
        highs = [sw.price for sw in s if sw.is_high]
        lows = [sw.price for sw in s if not sw.is_high]
        
        if len(highs) >= 3 and len(lows) >= 3:
            highs_descending = all(highs[i] >= highs[i + 1] for i in range(min(3, len(highs) - 1)))
            lows_ascending = all(lows[i] <= lows[i + 1] for i in range(min(3, len(lows) - 1)))
            
            if highs_descending and lows_ascending:
                details.append("✓ Converging trendlines (contracting triangle)")
                confidence += 0.25
                pattern_type = PatternType.TRIANGLE_CONTRACTING
            elif not highs_descending and not lows_ascending:
                details.append("✓ Expanding trendlines (expanding triangle)")
                confidence += 0.1
                pattern_type = PatternType.TRIANGLE_EXPANDING
            else:
                details.append("⚠ Trendlines not clearly converging or expanding")
                pattern_type = PatternType.TRIANGLE_CONTRACTING
        else:
            pattern_type = PatternType.TRIANGLE_CONTRACTING
        
        # Check wave length ratios
        if len(lengths) >= 5:
            if lengths[0] > lengths[2] > lengths[4]:
                details.append("✓ Declining wave lengths — supports triangle")
                confidence += 0.1
        
        confidence = max(0.0, min(1.0, confidence))
        
        if confidence > best_confidence:
            best_confidence = confidence
            best = PatternResult(
                pattern_type=pattern_type,
                confidence=confidence,
                waves=waves,
                swings=s,
                fib_levels=_make_fib(s[0].price, s[-1].price),
                invalidation_level=s[0].price,
                details=details,
                summary=f"{'Contracting' if 'Contracting' in str(pattern_type.value) else 'Expanding'} Triangle"
            )
    
    return best


# ═══════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def analyze(highs, lows, timestamps, sensitivity=5) -> PatternResult:
    """
    Full Elliott Wave analysis of price data.
    
    Returns the highest-confidence pattern found, or a PatternResult with UNKNOWN.
    """
    swings = detect_swings(highs, lows, timestamps, sensitivity)
    
    if len(swings) < 4:
        return PatternResult(
            pattern_type=PatternType.UNKNOWN,
            confidence=0.0,
            swings=swings,
            summary="Not enough swing points for pattern detection."
        )
    
    # Run all detectors
    detectors = [
        detect_impulse,
        detect_diagonal,
        detect_zigzag,
        detect_triangle,
    ]
    
    candidates = []
    for detector in detectors:
        result = detector(swings)
        if result and result.confidence >= 0.4:
            candidates.append(result)
    
    if not candidates:
        # Return best swing points with no pattern
        return PatternResult(
            pattern_type=PatternType.UNKNOWN,
            confidence=0.0,
            swings=swings,
            summary=f"Found {len(swings)} swing points but no clear Elliott Wave pattern. "
                     f"Market may be in consolidation or early trend stage."
        )
    
    # Sort by confidence
    candidates.sort(key=lambda r: r.confidence, reverse=True)
    best = candidates[0]
    
    # Add alternate if available
    if len(candidates) > 1:
        best.alternate = candidates[1]
    
    # Add fib to result
    if not best.fib_levels and len(best.swings) >= 2:
        s = best.swings
        best.fib_levels = _make_fib(s[0].price, s[-1].price)
    
    return best
