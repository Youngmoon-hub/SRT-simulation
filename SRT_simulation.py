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
Dependencies: numpy==1.26, scipy==1.12
Random seed: 42 (fixed for reproducibility)
"""

import numpy as np
from scipy import stats

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
np.random.seed(42)
N_ITER  = 1000   # iterations per condition
DAYS    = 28     # simulation length (days)
EPOCH   = 1440   # 1-min resolution (minutes/day)


# ─────────────────────────────────────────────
# Binary sleep/wake series generator
# ─────────────────────────────────────────────
def generate_series(bedtimes_min, durations_min, days=DAYS):
    """
    Generate a binary sleep/wake time series.

    Parameters
    ----------
    bedtimes_min  : array-like, bedtime for each day (minutes from midnight)
    durations_min : array-like, sleep duration for each day (minutes)
    days          : int, number of days

    Returns
    -------
    series : np.ndarray, shape (days * EPOCH,), dtype int8
             1 = asleep, 0 = awake
    """
    n_days = len(bedtimes_min)
    series = np.zeros(n_days * EPOCH, dtype=np.int8)
    for d in range(n_days):
        bed = int(bedtimes_min[d]) % EPOCH
        dur = int(max(60, durations_min[d]))
        start = d * EPOCH + bed
        for m in range(dur):
            idx = start + m
            if 0 <= idx < n_days * EPOCH:
                series[idx] = 1
    return series


# ─────────────────────────────────────────────
# Metric computation functions
# ─────────────────────────────────────────────
def compute_SRT(bedtimes, durations):
    """
    Sleep Regularity Test (Kwon, Kim, & Oh, 2013)
    SRT = (mean_duration / TSR) * 100

    TSR (Total Sleep Range): span of all sleep intervals
    overlaid on a 24-hour clock.
    """
    mean_dur = np.mean(durations)
    clock = np.zeros(EPOCH, dtype=np.int8)
    for bed, dur in zip(bedtimes, durations):
        bed = int(bed) % EPOCH
        dur = int(max(60, dur))
        for m in range(dur):
            clock[(bed + m) % EPOCH] = 1
    TSR = np.sum(clock)
    return (mean_dur / TSR) * 100 if TSR > 0 else 0.0


def compute_SRI(series, days=None):
    """
    Sleep Regularity Index (Phillips et al., 2017)
    Probability that sleep/wake state is the same
    at time t and t+24h, scaled 0-100.
    """
    total = np.sum(series[:-EPOCH] == series[EPOCH:])
    count = len(series) - EPOCH
    return (total / count) * 100 if count > 0 else 0.0


def compute_IS(series, days=None):
    """
    Interdaily Stability (Witting et al., 1990)
    Ratio of variance in mean hourly pattern
    to overall variance. Range: 0-1.
    """
    epoch_means = np.array([
        np.mean(series[e::EPOCH]) for e in range(EPOCH)
    ])
    overall_mean = np.mean(series)
    n_days = len(series) // EPOCH
    num = n_days * np.sum((epoch_means - overall_mean) ** 2)
    den = np.sum((series - overall_mean) ** 2)
    return num / den if den > 0 else 0.0


def compute_StDev(midsleeps):
    """
    Intra-individual standard deviation of midsleep time (minutes).
    """
    return float(np.std(midsleeps, ddof=1))


def compute_SJL(midsleeps, days=DAYS):
    """
    Social Jetlag (Wittmann et al., 2006)
    Absolute difference in mean midsleep between
    workdays (Mon-Fri) and free days (Sat-Sun).
    """
    work, free = [], []
    for d in range(days):
        if d % 7 in [5, 6]:
            free.append(midsleeps[d])
        else:
            work.append(midsleeps[d])
    if work and free:
        return abs(np.mean(free) - np.mean(work))
    return np.nan


def compute_CPD(midsleeps):
    """
    Composite Phase Deviation (Fischer, Vetter, & Roenneberg, 2016)
    Combined measure of circadian misalignment and irregularity.
    Simplified version: sqrt(misalignment^2 + irregularity^2) mean.
    """
    chronotype = np.mean(midsleeps)
    misalignment = midsleeps - chronotype
    irregularity = np.diff(midsleeps)
    cpd = np.sqrt(
        np.mean(misalignment**2) + np.mean(irregularity**2)
    )
    return float(cpd)


def compute_all_metrics(bedtimes, durations, midsleeps, report_sjl=True):
    """
    Compute all six metrics for one iteration.

    Returns
    -------
    dict with keys: SRT, SRI, IS, StDev, SJL, CPD
    """
    series = generate_series(bedtimes, durations)
    return {
        'SRT':   compute_SRT(bedtimes, durations),
        'SRI':   compute_SRI(series),
        'IS':    compute_IS(series),
        'StDev': compute_StDev(midsleeps),
        'SJL':   compute_SJL(midsleeps) if report_sjl else np.nan,
        'CPD':   compute_CPD(midsleeps),
    }


# ─────────────────────────────────────────────
# Scenario runner
# ─────────────────────────────────────────────
def run_scenario(scenario_fn, n_iter=N_ITER, report_sjl=False):
    """
    Run a scenario function n_iter times and return mean metrics.

    Parameters
    ----------
    scenario_fn : callable returning (bedtimes, durations, midsleeps)
    n_iter      : int
    report_sjl  : bool, whether to compute SJL

    Returns
    -------
    dict of mean metric values
    """
    results = {k: [] for k in ['SRT', 'SRI', 'IS', 'StDev', 'SJL', 'CPD']}
    for _ in range(n_iter):
        beds, durs, mids = scenario_fn()
        metrics = compute_all_metrics(beds, durs, mids, report_sjl=report_sjl)
        for k, v in metrics.items():
            results[k].append(v)
    return {k: np.nanmean(v) for k, v in results.items()}


# ─────────────────────────────────────────────
# Scenario definitions (Fischer et al., 2021)
# ─────────────────────────────────────────────
def scenario1(sd_min=60):
    """Scenario 1: Daily variation only."""
    mids  = np.random.normal(240, sd_min, DAYS)   # midsleep 4:00 AM
    durs  = np.random.normal(480, sd_min, DAYS)   # duration 8h
    durs  = np.clip(durs, 60, None)
    beds  = mids - durs / 2
    return beds, durs, mids


def scenario2(sd_min=60):
    """Scenario 2: Weekly + daily variation."""
    beds, durs, mids = [], [], []
    for d in range(DAYS):
        if d % 7 in [5, 6]:  # weekend: 00:00-08:00
            mid_base, dur_base = 240, 480
        else:                  # weekday: 23:30-06:30
            mid_base, dur_base = 210, 420
        mid = np.random.normal(mid_base, sd_min)
        dur = max(60, np.random.normal(dur_base, sd_min))
        mids.append(mid)
        durs.append(dur)
        beds.append(mid - dur / 2)
    return np.array(beds), np.array(durs), np.array(mids)


def scenario3(nap_days=28, nap_sd=0, sd_min=60):
    """Scenario 3: Naps added to Scenario 2."""
    beds, durs, mids = scenario2(sd_min)
    # Add naps on specified number of days
    nap_indices = np.random.choice(DAYS, nap_days, replace=False)
    for d in nap_indices:
        nap_bed = np.random.normal(14 * 60, nap_sd)  # 14:00
        nap_dur = 120  # 2h nap
        beds = np.append(beds, nap_bed)
        durs = np.append(durs, nap_dur)
    return beds, durs, mids


def scenario4(waso_min=120, n_bouts=5, sd_min=60):
    """Scenario 4: Nocturnal awakenings (WASO)."""
    beds, durs, mids = scenario2(sd_min)
    # WASO reduces effective sleep within main block
    # (onset/offset unchanged → SRT unaffected)
    durs = np.clip(durs - waso_min / n_bouts, 60, None)
    return beds, durs, mids


def scenario5(allnighter_days=7, sd_min=60):
    """Scenario 5: All-nighters."""
    beds, durs, mids = scenario2(sd_min)
    idx = np.random.choice(DAYS, allnighter_days, replace=False)
    durs[idx] = 0  # no sleep: excluded from SRT
    valid = durs > 0
    return beds[valid], durs[valid], mids[valid]


def scenario6(study_length=28, sd_min=60):
    """Scenario 6: Study length effect."""
    beds, durs, mids = [], [], []
    for d in range(study_length):
        if d % 7 in [5, 6]:
            mid_base, dur_base = 240, 480
        else:
            mid_base, dur_base = 210, 420
        # 1/7 probability of missing data
        if np.random.random() < 1/7:
            continue
        mid = np.random.normal(mid_base, sd_min)
        dur = max(60, np.random.normal(dur_base, sd_min))
        mids.append(mid)
        durs.append(dur)
        beds.append(mid - dur / 2)
    if not beds:
        return scenario2(sd_min)
    return np.array(beds), np.array(durs), np.array(mids)


def scenario7(cycle_days=7, mid_sd=30, dur_sd=30):
    """
    Scenario 7 (new): Three-shift rotating work schedule.
    Night shift:   work 23:00-07:00, sleep 09:00-15:00
    Day shift:     work 07:00-15:00, sleep 23:00-06:00
    Evening shift: work 15:00-23:00, sleep 01:00-07:00
    """
    shifts = [
        {'bed': 540,  'dur': 360},   # Night: 09:00, 6h
        {'bed': 1380, 'dur': 420},   # Day:   23:00, 7h
        {'bed': 60,   'dur': 360},   # Evening: 01:00, 6h
    ]
    beds, durs, mids = [], [], []
    for d in range(DAYS):
        sh = shifts[(d // cycle_days) % 3]
        bed = sh['bed'] + np.random.normal(0, mid_sd)
        dur = max(60, sh['dur'] + np.random.normal(0, dur_sd))
        beds.append(bed)
        durs.append(dur)
        mids.append((bed + dur / 2) % EPOCH)
    return np.array(beds), np.array(durs), np.array(mids)


# ─────────────────────────────────────────────
# Main: run all scenarios
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 65)
    print("SRT Simulation — Kwon (under review, Sleep)")
    print("=" * 65)

    print("\n[Scenario 1] Daily variation (SD = 60 min)")
    r = run_scenario(lambda: scenario1(60), report_sjl=False)
    print(f"  SRI={r['SRI']:.1f}  IS={r['IS']:.3f}  StDev={r['StDev']:.1f}"
          f"  CPD={r['CPD']:.1f}  SRT={r['SRT']:.1f}")

    print("\n[Scenario 2] Weekly + daily variation (SD = 60 min)")
    r = run_scenario(lambda: scenario2(60), report_sjl=True)
    print(f"  SRI={r['SRI']:.1f}  IS={r['IS']:.3f}  StDev={r['StDev']:.1f}"
          f"  SJL={r['SJL']:.1f}  CPD={r['CPD']:.1f}  SRT={r['SRT']:.1f}")

    print("\n[Scenario 3] Naps 100% (28/28 days, SD=0)")
    r = run_scenario(lambda: scenario3(28, 0), report_sjl=False)
    print(f"  SRI={r['SRI']:.1f}  IS={r['IS']:.3f}  SRT={r['SRT']:.1f}")

    print("\n[Scenario 4] WASO 120 min")
    r = run_scenario(lambda: scenario4(120), report_sjl=False)
    print(f"  SRI={r['SRI']:.1f}  IS={r['IS']:.3f}  SRT={r['SRT']:.1f}")

    print("\n[Scenario 5] All-nighters (7 days)")
    r = run_scenario(lambda: scenario5(7), report_sjl=False)
    print(f"  SRI={r['SRI']:.1f}  IS={r['IS']:.3f}  SRT={r['SRT']:.1f}")

    print("\n[Scenario 6] Study length: 2 days")
    r = run_scenario(lambda: scenario6(2), report_sjl=False)
    print(f"  SRI={r['SRI']:.1f}  IS={r['IS']:.3f}  SRT={r['SRT']:.1f}")

    print("\n[Scenario 6] Study length: 28 days")
    r = run_scenario(lambda: scenario6(28), report_sjl=False)
    print(f"  SRI={r['SRI']:.1f}  IS={r['IS']:.3f}  SRT={r['SRT']:.1f}")

    print("\n[Scenario 7] Shift work — 2-day rotation cycle")
    r = run_scenario(lambda: scenario7(2), report_sjl=False)
    print(f"  SRI={r['SRI']:.1f}  IS={r['IS']:.3f}  SRT={r['SRT']:.1f}")

    print("\n[Scenario 7] Shift work — 7-day rotation cycle")
    r = run_scenario(lambda: scenario7(7), report_sjl=False)
    print(f"  SRI={r['SRI']:.1f}  IS={r['IS']:.3f}  SRT={r['SRT']:.1f}")

    print("\nDone. Random seed = 42.")
