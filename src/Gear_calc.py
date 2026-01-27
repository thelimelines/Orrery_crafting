"""Optimized Python implementation of the gear train search.

This module ports the core logic from the provided JS, adds an optional Numba
JIT path for compound searches, and emits intermediate progress with ETA by default.
"""

from __future__ import annotations

import math
import time
from functools import lru_cache
from typing import Any, Callable, List, Optional, Tuple, TypedDict

np: Any
njit: Any
try:
    import numpy as np
    from numba import njit as _njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    NUMBA_AVAILABLE = False
    np = None

    def _njit(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

njit = _njit


class ProgressState(TypedDict, total=False):
    progress: float
    best_err: float
    best_a: int
    eta: float
    stage: str


class GearResult(TypedDict, total=False):
    status: str
    reason: str
    gear_string: str
    a_value: int
    error: float
    ratio_target: float
    ratio_achieved: float
    teeth_sum: int
    elapsed: float
    reverted: bool

INIT_ERR = 10_000_000.0
MINT = 3
MAXT = 400


def _js_round(x: float) -> int:
    # Match JS Math.round for positive values
    return int(math.floor(x + 0.5))


def _format_duration(seconds: float) -> str:
    if seconds is None or seconds < 0:
        return "--:--:--"
    s = int(seconds + 0.5)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def largest_prime_factor(n: int) -> int:
    """Find the largest prime factor (port of fac in JS)."""
    if n < 2:
        return 0
    while n % 2 == 0:
        n //= 2
    if n == 1:
        return 2
    oldn = n
    d = 3
    maxd = int(math.isqrt(n))
    while d <= maxd:
        while n % d == 0:
            n //= d
        if n != oldn:
            if n == 1:
                return d
            maxd = int(math.isqrt(n))
            oldn = n
        d += 2
    return n


def _generate_stage_factors(stages: int, minw: int, maxw: int) -> Tuple[List[int], List[int]]:
    # fmin/fmax are 1-indexed in the JS logic; keep slot 0 and 1 = 1.
    fmin = [1, 1]
    fmax = [1, 1]
    k = 1
    h = 1
    for _ in range(1, stages):
        k *= minw
        h *= maxw
        fmin.append(k)
        fmax.append(h)
    return fmin, fmax


def _make_packable(minw: int, maxw: int, fmin: List[int], cache_maxsize: Optional[int] = None):
    @lru_cache(maxsize=cache_maxsize)
    def pkbo(n: int, s: int) -> bool:
        if s == 1:
            return minw <= n <= maxw
        # floor root; JS comment notes ceil could miss perfect roots due to roundoff
        l = max(int(n ** (1.0 / s)), minw)
        t = min(maxw, n // fmin[s])
        if t % 2 == 0:
            t -= 1
        while t >= l:
            if n % t == 0 and pkbo(n // t, s - 1):
                return True
            t -= 2
        return False

    @lru_cache(maxsize=cache_maxsize)
    def pkb(n: int, s: int) -> bool:
        if n % 2 == 1:
            return pkbo(n, s)
        if s == 1:
            return minw <= n <= maxw
        l = max(int(n ** (1.0 / s)), minw)
        t = min(maxw, n // fmin[s]) + 1
        while True:
            t -= 1
            if t < l:
                break
            if n % t == 0 and pkb(n // t, s - 1):
                return True
        return False

    return pkb


def _pack_best(
    n: int, s: int, minw: int, maxw: int, fmin: List[int], fmax: List[int], norm: bool
) -> Tuple[int, List[int]]:
    # Bin packing optimum / balanced - find gears with smallest wheels.
    best_sum = maxw * s + 1
    wheels = [0] * s
    best_wheels: List[int] = []

    def rec(n_: int, s_: int, su: int) -> None:
        nonlocal best_sum, best_wheels
        if s_ == 1:
            if su + n_ < best_sum:
                best_sum = su + n_
                wheels[0] = n_
                best_wheels = wheels.copy()
            return
        if s_ == 2:
            l = math.ceil(math.sqrt(n_))
        else:
            l = max(math.ceil(n_ / fmax[s_]), minw)
        upper = min(n_ // fmin[s_], maxw)
        for t in range(upper, l - 1, -1):
            if n_ % t == 0:
                wheels[s_ - 1] = t
                rec(n_ // t, s_ - 1, su + (t if norm else 0))

    rec(n, s, 0)
    return best_sum, best_wheels


def _gear_string(
    savea: int,
    ratio: float,
    stages: int,
    stagesp: int,
    minw: int,
    maxw: int,
    pawl: bool,
    tally: bool,
    rc: bool,
    fmin: List[int],
    fmax: List[int],
) -> Tuple[str, int]:
    # Pack the denominator side
    sum_a, wheels_a = _pack_best(savea, stagesp, minw, maxw, fmin, fmax, True)

    # Pack the numerator side
    norm = (not pawl) or tally
    num = _js_round(savea * ratio)
    sum_b, wheels_b = _pack_best(num, stages, minw, maxw, fmin, fmax, norm)
    total_sum = sum_a + sum_b

    res = ""
    if pawl:
        res += f"R{wheels_b[-1]},"
    for i in range(stagesp - 1, -1, -1):
        if rc:
            res += f"{wheels_b[i]}:{wheels_a[i]}"
        else:
            res += f"{wheels_a[i]}:{wheels_b[i]}"
        if i:
            res += ","
    return res, total_sum


def _default_progress_callback(state: ProgressState) -> None:
    percent = state.get("progress", 0.0) * 100.0
    best_err = state.get("best_err", INIT_ERR)
    best_a = state.get("best_a", 0)
    eta = _format_duration(state.get("eta", -1))
    stage = state.get("stage", "compound")
    print(
        f"[{stage}] {percent:6.2f}% | best_err={best_err:.6g} | a={best_a} | ETA {eta}",
        flush=True,
    )


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _pkbo_nb(n, s, minw, maxw, fmin):
        if s == 1:
            return minw <= n <= maxw
        l = int(n ** (1.0 / s))
        if l < minw:
            l = minw
        t = n // fmin[s]
        if t > maxw:
            t = maxw
        if t % 2 == 0:
            t -= 1
        while t >= l:
            if n % t == 0 and _pkbo_nb(n // t, s - 1, minw, maxw, fmin):
                return True
            t -= 2
        return False

    @njit(cache=True)
    def _pkb_nb(n, s, minw, maxw, fmin):
        if n % 2 == 1:
            return _pkbo_nb(n, s, minw, maxw, fmin)
        if s == 1:
            return minw <= n <= maxw
        l = int(n ** (1.0 / s))
        if l < minw:
            l = minw
        t = n // fmin[s]
        if t > maxw:
            t = maxw
        t += 1
        while True:
            t -= 1
            if t < l:
                break
            if n % t == 0 and _pkb_nb(n // t, s - 1, minw, maxw, fmin):
                return True
        return False

    @njit(cache=True)
    def _search_chunk_nb(
        start_a,
        end_a,
        ratio,
        tol,
        stages,
        stagesp,
        minw,
        maxw,
        fmin,
        best_err,
        best_a,
    ):
        for a in range(start_a, end_a):
            b = int(a * ratio + 0.5)
            err = ratio - (b / a)
            if err < 0:
                err = -err
            if err < best_err:
                if not _pkb_nb(b, stages, minw, maxw, fmin):
                    continue
                if not _pkb_nb(a, stagesp, minw, maxw, fmin):
                    continue
                best_err = err
                best_a = a
                if err <= tol:
                    return best_err, best_a, True
        return best_err, best_a, False


def compute_compound(
    ratio: float,
    tol: float,
    stages: int,
    minw: int = MINT,
    maxw: int = MAXT,
    pawl: bool = False,
    tally: bool = False,
    batch_size: int = 10_000_000,
    cache_maxsize: Optional[int] = None,
    use_numba: bool = True,
    progress_interval: float = 2.5,
    progress_callback: Optional[Callable[[ProgressState], None]] = None,
) -> GearResult:
    """Search compound gear trains.

    Notes:
    - batch_size controls the chunk size for progress/ETA updates (larger = less overhead).
    - cache_maxsize caps the Python packability cache to limit memory use.
    - use_numba enables a JIT-compiled scan when numba is available.
    """
    if ratio <= 0 or not math.isfinite(ratio):
        raise ValueError("ratio must be positive and finite")
    if stages < 1:
        raise ValueError("stages must be >= 1 for compound trains")
    minw = max(int(minw), MINT)
    maxw = min(max(int(maxw), minw + 1), MAXT)

    stagesp = stages
    if pawl:
        stages += 1

    rc = False
    if ratio < 1:
        ratio = 1 / ratio
        rc = True

    if float(ratio).is_integer() and maxw < largest_prime_factor(int(ratio)):
        return {
            "status": "impossible",
            "reason": "ratio not divisible by any gear size",
        }

    fmin, fmax = _generate_stage_factors(stages, minw, maxw)
    maxc = _js_round((maxw ** stages) / ratio) + 1
    maxc2 = int((maxw ** stagesp)) + 1
    if maxc2 < maxc:
        maxc = maxc2
    minc = _js_round(minw ** stagesp)
    progress_range = max(maxc - minc, 1)

    best_err = INIT_ERR
    best_a = 0
    start_time = time.perf_counter()
    last_report = start_time - progress_interval

    if progress_callback is None:
        progress_callback = _default_progress_callback

    use_numba = bool(use_numba and NUMBA_AVAILABLE)
    if batch_size <= 0:
        batch_size = 1_000_000

    if use_numba:
        fmin_arr = np.asarray(fmin, dtype=np.int64)
        a = minc
        chunk = int(batch_size)
        if chunk < 50_000:
            chunk = 50_000
        while a < maxc:
            end = a + chunk
            if end > maxc:
                end = maxc
            best_err, best_a, hit = _search_chunk_nb(
                a,
                end,
                ratio,
                tol,
                stages,
                stagesp,
                minw,
                maxw,
                fmin_arr,
                best_err,
                best_a,
            )
            a = end
            now = time.perf_counter()
            if now - last_report >= progress_interval:
                progress = (a - minc) / progress_range
                eta = (now - start_time) * (1 / progress - 1) if progress > 0 else -1
                progress_callback(
                    {
                        "progress": progress,
                        "best_err": best_err,
                        "best_a": best_a,
                        "eta": eta,
                        "stage": "compound",
                    }
                )
                last_report = now
            if hit:
                break
    else:
        # Build a single cached packability checker for the entire scan.
        pkb = _make_packable(minw, maxw, fmin, cache_maxsize)

        # Sequential scan preserves the original JS "record low" semantics.
        abs_ = abs
        ratio_ = ratio
        tol_ = tol
        for a in range(minc, maxc):
            b = _js_round(a * ratio_)
            err = abs_(ratio_ - (b / a))
            if err < best_err:
                if not pkb(b, stages):
                    continue
                if not pkb(a, stagesp):
                    continue
                best_err = err
                best_a = a
                if err <= tol_:
                    break

            if (a & 0xFFF) == 0:
                now = time.perf_counter()
                if now - last_report >= progress_interval:
                    progress = (a - minc) / progress_range
                    eta = (now - start_time) * (1 / progress - 1) if progress > 0 else -1
                    progress_callback(
                        {
                            "progress": progress,
                            "best_err": best_err,
                            "best_a": best_a,
                            "eta": eta,
                            "stage": "compound",
                        }
                    )
                    last_report = now

    elapsed = time.perf_counter() - start_time
    if best_err == INIT_ERR:
        return {"status": "impossible", "elapsed": elapsed}

    gear_str, teeth_sum = _gear_string(
        best_a, ratio, stages, stagesp, minw, maxw, pawl, tally, rc, fmin, fmax
    )
    achieved = _js_round(best_a * ratio) / best_a
    if rc:
        achieved = 1.0 / achieved

    return {
        "status": "ok",
        "gear_string": gear_str,
        "a_value": best_a,
        "error": best_err,
        "ratio_target": (1 / ratio) if rc else ratio,
        "ratio_achieved": achieved,
        "teeth_sum": teeth_sum,
        "elapsed": elapsed,
        "reverted": False,
    }


def compute_reverted(
    ratio: float,
    minw: int = MINT,
    maxw: int = MAXT,
    progress_interval: float = 2.5,
    progress_callback: Optional[Callable[[ProgressState], None]] = None,
) -> GearResult:
    """Search reverted gear trains (2-stage) with progress output."""
    if ratio <= 0 or not math.isfinite(ratio):
        raise ValueError("ratio must be positive and finite")
    minw = max(int(minw), MINT)
    maxw = min(max(int(maxw), minw + 1), MAXT)

    if progress_callback is None:
        progress_callback = _default_progress_callback

    start_time = time.perf_counter()
    last_report = start_time - progress_interval

    best_err = INIT_ERR
    save_sum = MAXT * 2
    best = (0, 0, 0, 0)

    total_a = maxw - minw + 1
    for ai, a in enumerate(range(minw, maxw + 1), start=1):
        if minw > _js_round((a + maxw) / (ratio * a / maxw + 1)):
            break
        for b in range(minw, maxw + 1):
            c = _js_round((a + b) / (ratio * a / b + 1))
            if c > maxw:
                break
            if c < minw:
                continue
            d = a + b - c
            if d < minw:
                break
            if d > maxw:
                continue
            err = abs((b * d) / (a * c) - ratio)
            if err > best_err:
                continue
            tot = a + b
            if err < best_err or tot < save_sum:
                best_err = err
                save_sum = tot
                best = (a, b, c, d)

        now = time.perf_counter()
        if now - last_report >= progress_interval:
            progress = ai / total_a
            eta = (now - start_time) * (1 / progress - 1) if progress > 0 else -1
            progress_callback(
                {
                    "progress": progress,
                    "best_err": best_err,
                    "best_a": best[0],
                    "eta": eta,
                    "stage": "reverted",
                }
            )
            last_report = now

    elapsed = time.perf_counter() - start_time
    if best_err == INIT_ERR:
        return {"status": "impossible", "elapsed": elapsed}

    a, b, c, d = best
    if max(b, c) < max(a, d):
        gear_str = f"{a}:{b},{c}:{d}"
    else:
        gear_str = f"{c}:{d},{a}:{b}"

    return {
        "status": "ok",
        "gear_string": gear_str,
        "error": best_err,
        "ratio_target": ratio,
        "ratio_achieved": (b * d) / (a * c),
        "teeth_sum": (a + b) * 2,
        "elapsed": elapsed,
        "reverted": True,
    }


def compute_gear_train(
    ratio: float,
    tol: float,
    stages: int,
    minw: int = MINT,
    maxw: int = MAXT,
    pawl: bool = False,
    tally: bool = False,
    batch_size: int = 10_000_000,
    cache_maxsize: Optional[int] = None,
    use_numba: bool = True,
    progress_interval: float = 2.5,
    progress_callback: Optional[Callable[[ProgressState], None]] = None,
) -> GearResult:
    """Dispatch to compound or reverted based on stages."""
    if stages > 0:
        return compute_compound(
            ratio,
            tol,
            stages,
            minw=minw,
            maxw=maxw,
            pawl=pawl,
            tally=tally,
            batch_size=batch_size,
            cache_maxsize=cache_maxsize,
            use_numba=use_numba,
            progress_interval=progress_interval,
            progress_callback=progress_callback,
        )
    return compute_reverted(
        ratio,
        minw=minw,
        maxw=maxw,
        progress_interval=progress_interval,
        progress_callback=progress_callback,
    )


if __name__ == "__main__":
    # Example usage (adjust to your needs)
    result = compute_gear_train(
        ratio=math.pi,
        tol=1e-15,
        stages=6,
        minw=10,
        maxw=120,
        batch_size=10_000_000,
        progress_interval=5.0,
        use_numba=True,
    )
    print(result)
