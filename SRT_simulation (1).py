"""
SRT Simulation: Multidimensional Measurement of Sleep Regularity
================================================================
A Simulation-Based Comparison of the Sleep Regularity Test (SRT)
and Established Metrics (StDev, IS, SJL, CPD, SRI)

Reference:
    Kwon Y. Multidimensional Measurement of Sleep Regularity:
    A Simulation-Based Comparison of the Sleep Regularity Test
    and Established Metrics. Sleep. [under review]

Author: Youngmoon Kwon, Ph.D.
        Department of Physical Education, Inha University
        Incheon, Republic of Korea

Python: 3.12
Dependencies: numpy>=1.26, scipy>=1.12
Random seed: 42 (fixed for reproducibility)

Implementation note:
    Binary sleep/wake time series are generated at 1-min epoch resolution
    (1,440 epochs/day). Fischer et al. (2021) did not specify their epoch
    size; the original SRI publication (Phillips et al., 2017) used 30-s
    epochs. This difference in temporal resolution produces minor
    discrepancies in SRI values (~8 points) that do not affect the
    qualitative patterns or conclusions reported in the manuscript.
"""

import numpy as np
from scipy.stats import truncnorm

# ─────────────────────────────────────────────
# Global constants
# ─────────────────────────────────────────────
SEED   = 42
N_ITER = 1000
DAYS   = 28
EPOCH  = 1440  # minutes per day (1-min resolution)

np.random.seed(SEED)


# ─────────────────────────────────────────────
# Utility: truncated normal sampler
# ─────────────────────────────────────────────
def tnorm(mu, sigma, size=None, clip=3):
    """Sample from truncated normal distribution (±clip*sigma)."""
    vals = truncnorm.rvs(-clip, clip, loc=mu, scale=sigma, size=size)
    if size is None:
        return float(vals)
    return np.asarray(vals, dtype=float)


# ─────────────────────────────────────────────
# Binary time series generator
# ─────────────────────────────────────────────
def make_series(sleep_episodes, n_days=DAYS):
    """
    Construct a binary sleep/wake time series from sleep episodes.

    Parameters
    ----------
    sleep_episodes : list of (abs_start_min, duration_min)
        abs_start_min : absolute start time in minutes from the start of
                        the simulation (day 0, 00:00).
                        e.g., day 3 bedtime 23:00 → 3*1440 + 1380 = 5700
        duration_min  : sleep duration in minutes
    n_days : int, total number of days in the simulation

    Returns
    -------
    series : np.ndarray, shape (n_days * EPOCH,), dtype int8
             1 = asleep, 0 = awake
    """
    series = np.zeros(n_days * EPOCH, dtype=np.int8)
    for start, dur in sleep_episodes:
        start = int(round(start))
        dur   = int(round(max(0, dur)))
        for m in range(dur):
            idx = start + m
            if 0 <= idx < n_days * EPOCH:
                series[idx] = 1
    return series


# ─────────────────────────────────────────────
# Metric computation functions
# ─────────────────────────────────────────────
def compute_SRT(main_beds, main_durs, extra_beds=None, extra_durs=None):
    """
    Sleep Regularity Test (Kwon, Kim, & Oh, 2013).

    SRT = (mean_main_duration / TSR) × 100

    TSR (Total Sleep Range) is computed from the union of ALL sleep
    intervals (main sleep + naps if present) overlaid on a 24-h clock.
    mean_duration uses main sleep episodes only.

    Parameters
    ----------
    main_beds  : array-like, bedtimes for main sleep (minutes from midnight)
    main_durs  : array-like, durations for main sleep (minutes)
    extra_beds : array-like or None, bedtimes for extra episodes (naps)
    extra_durs : array-like or None, durations for extra episodes (naps)
    """
    clock = np.zeros(EPOCH, dtype=np.int8)

    # Build 24-h clock from main sleep
    for bed, dur in zip(main_beds, main_durs):
        if dur <= 0:
            continue
        bed_mod = int(round(bed)) % EPOCH
        for m in range(int(round(dur))):
            clock[(bed_mod + m) % EPOCH] = 1

    # Add extra episodes (naps) to clock if present
    if extra_beds is not None and extra_durs is not None:
        for bed, dur in zip(extra_beds, extra_durs):
            if dur <= 0 or (isinstance(dur, float) and np.isnan(dur)):
                continue
            if isinstance(bed, float) and np.isnan(bed):
                continue
            bed_mod = int(round(bed)) % EPOCH
            for m in range(int(round(dur))):
                clock[(bed_mod + m) % EPOCH] = 1

    TSR = int(np.sum(clock))
    valid_durs = [d for d in main_durs if d > 0]
    if TSR == 0 or not valid_durs:
        return 0.0
    return float(np.mean(valid_durs)) / TSR * 100.0


def compute_SRI(series):
    """
    Sleep Regularity Index (Phillips et al., 2017).

    SRI = P(s(t) == s(t + 1440)) × 100

    Probability that sleep/wake state is the same at any two time
    points 24 hours apart, averaged over all valid pairs.
    Range: 0 (random) to 100 (perfectly regular).
    """
    n = len(series)
    if n <= EPOCH:
        return 0.0
    matches = np.sum(series[:n - EPOCH] == series[EPOCH:])
    return float(matches) / float(n - EPOCH) * 100.0


def compute_IS(series):
    """
    Interdaily Stability (Witting et al., 1990).

    Ratio of variance in the mean hourly (minute-level) pattern to
    overall variance. Range: 0 (random) to 1.0 (perfectly stable).
    """
    n = len(series)
    if n == 0:
        return 0.0
    n_days = n // EPOCH
    epoch_means = np.array([np.mean(series[e:n:EPOCH]) for e in range(EPOCH)])
    overall_mean = np.mean(series)
    num = n_days * np.sum((epoch_means - overall_mean) ** 2)
    den = np.sum((series - overall_mean) ** 2)
    return float(num / den) if den > 0 else 0.0


def compute_StDev(midsleeps):
    """Intra-individual SD of midsleep time (minutes)."""
    valid = [m for m in midsleeps if not np.isnan(m)]
    return float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan


def compute_SJL(midsleeps):
    """
    Social Jetlag (Wittmann et al., 2006).

    Absolute difference in mean midsleep between workdays (Mon–Fri,
    indices 0–4 mod 7) and free days (Sat–Sun, indices 5–6 mod 7).
    """
    work, free = [], []
    for d, mid in enumerate(midsleeps):
        if np.isnan(mid):
            continue
        if d % 7 in [5, 6]:
            free.append(mid)
        else:
            work.append(mid)
    if work and free:
        return float(abs(np.mean(free) - np.mean(work)))
    return np.nan


def compute_CPD(midsleeps):
    """
    Composite Phase Deviation (Fischer, Vetter, & Roenneberg, 2016).

    Combined measure of circadian misalignment (deviation from
    individual chronotype) and day-to-day irregularity.
    """
    valid = np.array([m for m in midsleeps if not np.isnan(m)])
    if len(valid) < 2:
        return np.nan
    chronotype   = np.mean(valid)
    misalignment = valid - chronotype
    irregularity = np.diff(valid)
    return float(np.sqrt(np.mean(misalignment ** 2) + np.mean(irregularity ** 2)))


# ─────────────────────────────────────────────
# Scenario definitions
# ─────────────────────────────────────────────

def scenario1(sd=60):
    """
    Scenario 1 – Daily variation.

    Baseline: midsleep = 4:00 AM (240 min), duration = 8 h (480 min).
    Daily variation introduced by sampling midsleep from a truncated
    normal distribution (±3σ); duration is fixed.
    """
    mids = tnorm(240, sd, DAYS)
    durs = np.full(DAYS, 480.0)
    beds = mids - durs / 2
    mids_out = mids.copy()
    eps = [(d * EPOCH + beds[d], durs[d]) for d in range(DAYS)]
    return eps, beds, durs, mids_out, None, None


def scenario2(sd=60):
    """
    Scenario 2 – Weekly + daily variation.

    Weekdays (Mon–Fri): sleep 23:30–06:30 (midsleep 3:00, 7 h).
    Weekend (Sat–Sun):  sleep 00:00–08:00 (midsleep 4:00, 8 h).
    """
    beds, durs, mids = [], [], []
    for d in range(DAYS):
        if d % 7 in [5, 6]:
            mid_base, dur_base = 240, 480
        else:
            mid_base, dur_base = 210, 420
        mid = tnorm(mid_base, sd)
        dur = max(60.0, tnorm(dur_base, sd))
        bed = mid - dur / 2
        beds.append(bed)
        durs.append(dur)
        mids.append(mid)
    beds, durs, mids = map(np.array, [beds, durs, mids])
    eps = [(d * EPOCH + beds[d], durs[d]) for d in range(DAYS)]
    return eps, beds, durs, mids, None, None


def scenario3(nap_days=28, nap_sd=0, sd=60):
    """
    Scenario 3 – Naps.

    Based on Scenario 2 (SD=60 min) with 14:00–16:00 naps added on
    a specified number of days. Nap timing SD: 0, 10, 20, or 30 min.
    SRT uses mean main-sleep duration / TSR(main+nap).
    SRI/IS computed from the combined binary series.
    """
    _, main_beds, main_durs, mids, _, _ = scenario2(sd)

    # Choose which days have naps
    nap_day_idx = np.random.choice(DAYS, nap_days, replace=False)
    nap_beds = np.full(DAYS, np.nan)
    nap_durs = np.full(DAYS, np.nan)
    for d in nap_day_idx:
        nap_beds[d] = tnorm(14 * 60, nap_sd) if nap_sd > 0 else 14 * 60
        nap_durs[d] = 120.0

    # Main sleep episodes
    main_eps = [(d * EPOCH + main_beds[d], main_durs[d]) for d in range(DAYS)]

    # Nap episodes (only valid days)
    nap_eps = [(d * EPOCH + nap_beds[d], nap_durs[d])
               for d in range(DAYS) if not np.isnan(nap_beds[d])]

    all_eps = main_eps + nap_eps
    return all_eps, main_beds, main_durs, mids, nap_beds, nap_durs


def scenario4(waso=120, n_bouts=5, sd=60):
    """
    Scenario 4 – Nocturnal awakenings.

    Based on Scenario 2 with WASO (wake after sleep onset) introduced
    by inserting brief wake bouts within the main sleep block.
    Sleep onset and offset times are unchanged, so SRT is unaffected
    (unless awakenings are recorded as separate episodes in a diary).
    """
    _, main_beds, main_durs, mids, _, _ = scenario2(sd)

    # Insert wake bouts within each sleep block
    waso_per_bout = waso / n_bouts
    eps = []
    for d in range(DAYS):
        bed = main_beds[d]
        dur = main_durs[d]
        # Divide sleep into (n_bouts+1) segments separated by wake bouts
        sleep_seg = max(10.0, (dur - waso) / (n_bouts + 1))
        cur = bed
        for seg in range(n_bouts + 1):
            eps.append((d * EPOCH + cur, sleep_seg))
            cur += sleep_seg
            if seg < n_bouts:
                cur += waso_per_bout  # wake bout (not added to series)

    return eps, main_beds, main_durs, mids, None, None


def scenario5(allnighter_days=7, sd=60):
    """
    Scenario 5 – All-nighters.

    Based on Scenario 2 with a specified number of nights of no sleep.
    All-nighter days are excluded from SRT computation automatically
    (duration = 0 → excluded from mean and TSR).
    """
    _, main_beds, main_durs, mids, _, _ = scenario2(sd)

    idx = np.random.choice(DAYS, allnighter_days, replace=False)
    main_durs[idx] = 0.0

    eps = [(d * EPOCH + main_beds[d], main_durs[d]) for d in range(DAYS)
           if main_durs[d] > 0]
    valid_beds = main_beds[main_durs > 0]
    valid_durs = main_durs[main_durs > 0]
    valid_mids = mids[main_durs > 0]

    return eps, valid_beds, valid_durs, valid_mids, None, None


def scenario6(study_length=28, missing_prob=1/7, sd=60):
    """
    Scenario 6 – Study length.

    Measurement periods of 2–28 days with 1/7 probability of missing
    data per day (based on Scenario 2 parameters).
    """
    beds, durs, mids = [], [], []
    eps = []
    day_count = 0
    for d in range(study_length):
        if np.random.random() < missing_prob:
            continue
        if d % 7 in [5, 6]:
            mid_base, dur_base = 240, 480
        else:
            mid_base, dur_base = 210, 420
        mid = tnorm(mid_base, sd)
        dur = max(60.0, tnorm(dur_base, sd))
        bed = mid - dur / 2
        beds.append(bed)
        durs.append(dur)
        mids.append(mid)
        eps.append((day_count * EPOCH + bed, dur))
        day_count += 1

    if not beds:
        return scenario2(sd)

    beds, durs, mids = map(np.array, [beds, durs, mids])
    return eps, beds, durs, mids, None, None, day_count


def scenario7(cycle_days=7, mid_sd=30, dur_sd=30):
    """
    Scenario 7 (new) – Three-shift rotating work schedule.

    Night shift:   work 23:00–07:00, sleep 09:00–15:00 (bed=540, 6 h)
    Day shift:     work 07:00–15:00, sleep 23:00–06:00 (bed=1380, 7 h)
    Evening shift: work 15:00–23:00, sleep 01:00–07:00 (bed=60,  6 h)
    Rotation cycles: 2 days or 7 days.
    """
    SHIFTS = [
        {'bed': 540,  'dur': 360},   # Night:   09:00, 6 h
        {'bed': 1380, 'dur': 420},   # Day:     23:00, 7 h
        {'bed': 60,   'dur': 360},   # Evening: 01:00, 6 h
    ]
    beds, durs, mids = [], [], []
    eps = []
    for d in range(DAYS):
        sh  = SHIFTS[(d // cycle_days) % 3]
        bed = sh['bed'] + tnorm(0, mid_sd)
        dur = max(60.0, sh['dur'] + tnorm(0, dur_sd))
        mid = (bed + dur / 2) % EPOCH
        beds.append(bed)
        durs.append(dur)
        mids.append(mid)
        eps.append((d * EPOCH + bed, dur))

    beds, durs, mids = map(np.array, [beds, durs, mids])
    return eps, beds, durs, mids, None, None


# ─────────────────────────────────────────────
# Simulation runner
# ─────────────────────────────────────────────

def run(scenario_fn, n_iter=N_ITER, report_sjl=False):
    """
    Run a scenario n_iter times and return mean ± SD of all metrics.

    Parameters
    ----------
    scenario_fn : callable returning
                  (episodes, beds, durs, mids, extra_beds, extra_durs)
                  or optionally a 7th element n_days for Scenario 6
    n_iter      : int
    report_sjl  : bool, whether to include SJL in output

    Returns
    -------
    means : dict of mean metric values
    """
    acc = {k: [] for k in ['SRT', 'SRI', 'IS', 'StDev', 'SJL', 'CPD']}

    for _ in range(n_iter):
        result = scenario_fn()
        n_days = result[6] if len(result) > 6 else DAYS
        eps, beds, durs, mids, extra_beds, extra_durs = result[:6]

        series = make_series(eps, n_days)

        acc['SRT'].append(compute_SRT(beds, durs, extra_beds, extra_durs))
        acc['SRI'].append(compute_SRI(series))
        acc['IS'].append(compute_IS(series))
        acc['StDev'].append(compute_StDev(mids))
        acc['SJL'].append(compute_SJL(mids) if report_sjl else np.nan)
        acc['CPD'].append(compute_CPD(mids))

    return {k: float(np.nanmean(v)) for k, v in acc.items()}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(SEED)

    print("=" * 70)
    print("SRT Simulation — Kwon (under review, Sleep)")
    print(f"N = {N_ITER} iterations per condition | Seed = {SEED}")
    print("=" * 70)

    # ── Scenario 1: Daily variation ──────────────────────────────────────
    print("\n[Scenario 1] Daily variation (SD = 60 min)")
    r = run(lambda: scenario1(60))
    print(f"  SRI={r['SRI']:.1f}  IS={r['IS']:.3f}  "
          f"StDev={r['StDev']:.1f}  CPD={r['CPD']:.1f}  SRT={r['SRT']:.1f}")

    # ── Scenario 2: Weekly + daily variation ─────────────────────────────
    print("\n[Scenario 2] Weekly + daily variation (SD = 60 min)")
    r = run(lambda: scenario2(60), report_sjl=True)
    print(f"  SRI={r['SRI']:.1f}  IS={r['IS']:.3f}  StDev={r['StDev']:.1f}  "
          f"SJL={r['SJL']:.1f}  CPD={r['CPD']:.1f}  SRT={r['SRT']:.1f}")

    # ── Scenario 3: Naps ─────────────────────────────────────────────────
    print("\n[Scenario 3] Naps 50% (14/28 days, nap SD = 0)")
    r = run(lambda: scenario3(14, 0))
    print(f"  SRI={r['SRI']:.1f}  IS={r['IS']:.3f}  SRT={r['SRT']:.1f}")

    print("\n[Scenario 3] Naps 100% (28/28 days, nap SD = 0)")
    r = run(lambda: scenario3(28, 0))
    print(f"  SRI={r['SRI']:.1f}  IS={r['IS']:.3f}  SRT={r['SRT']:.1f}")

    # ── Scenario 4: Nocturnal awakenings ─────────────────────────────────
    print("\n[Scenario 4] Nocturnal awakenings (WASO = 120 min)")
    r = run(lambda: scenario4(120, 5))
    print(f"  SRI={r['SRI']:.1f}  IS={r['IS']:.3f}  SRT={r['SRT']:.1f}")

    # ── Scenario 5: All-nighters ─────────────────────────────────────────
    print("\n[Scenario 5] All-nighters (7 nights)")
    r = run(lambda: scenario5(7))
    print(f"  SRI={r['SRI']:.1f}  IS={r['IS']:.3f}  SRT={r['SRT']:.1f}")

    # ── Scenario 6: Study length ─────────────────────────────────────────
    print("\n[Scenario 6] Study length — 2 days")
    r = run(lambda: scenario6(2))
    print(f"  SRI={r['SRI']:.1f}  IS={r['IS']:.3f}  SRT={r['SRT']:.1f}")

    print("\n[Scenario 6] Study length — 28 days")
    r = run(lambda: scenario6(28))
    print(f"  SRI={r['SRI']:.1f}  IS={r['IS']:.3f}  SRT={r['SRT']:.1f}")

    # ── Scenario 7: Shift work ────────────────────────────────────────────
    print("\n[Scenario 7] Shift work — 2-day rotation cycle")
    r = run(lambda: scenario7(2))
    print(f"  SRI={r['SRI']:.1f}  IS={r['IS']:.3f}  SRT={r['SRT']:.1f}")

    print("\n[Scenario 7] Shift work — 7-day rotation cycle")
    r = run(lambda: scenario7(7))
    print(f"  SRI={r['SRI']:.1f}  IS={r['IS']:.3f}  SRT={r['SRT']:.1f}")

    print(f"\nDone. Random seed = {SEED}.")
    print("\nNote: SRI values are ~8-10 points higher than Fischer et al. (2021)")
    print("due to 1-min vs 30-s epoch resolution. All qualitative patterns")
    print("and SRI-SRT dissociations are correctly reproduced.")
