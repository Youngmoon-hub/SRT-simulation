# SRT-simulation# SRT-simulation

**Simulation code for:**

> Kwon Y. Multidimensional Measurement of Sleep Regularity: A Simulation-Based Comparison of the Sleep Regularity Test and Established Metrics. *Sleep*. [under review]

---

## Overview

This repository contains the Python simulation code used in the above manuscript. The code replicates the Monte Carlo simulation framework of Fischer, Klerman, and Phillips (2021, *Sleep*) and extends it by:

1. Adding the **Sleep Regularity Test (SRT)** as a sixth metric
2. Adding a novel **shift-work scenario** (Scenario 7)

Six sleep regularity metrics are simultaneously computed across seven scenarios:

| Metric | Reference |
|--------|-----------|
| SRT (Sleep Regularity Test) | Kwon, Kim, & Oh (2013) |
| SRI (Sleep Regularity Index) | Phillips et al. (2017) |
| IS (Interdaily Stability) | Witting et al. (1990) |
| StDev (Intra-individual SD) | Bei et al. (2016) |
| SJL (Social Jetlag) | Wittmann et al. (2006) |
| CPD (Composite Phase Deviation) | Fischer, Vetter, & Roenneberg (2016) |

---

## Requirements

```
Python >= 3.12
numpy >= 1.26
scipy >= 1.12
```

Install dependencies:

```bash
pip install numpy scipy
```

---

## Usage

```bash
python SRT_simulation.py
```

All scenarios run automatically. Output shows mean metric values per scenario (1,000 iterations each).

**Random seed is fixed at 42** to ensure full reproducibility.

---

## Scenarios

| Scenario | Description | Source |
|----------|-------------|--------|
| 1 | Daily midsleep variation (SD = 0–120 min) | Fischer et al. (2021) |
| 2 | Weekly + daily variation (workday vs weekend) | Fischer et al. (2021) |
| 3 | Naps added on 0–28 days | Fischer et al. (2021) |
| 4 | Nocturnal awakenings (WASO = 0–240 min) | Fischer et al. (2021) |
| 5 | All-nighters (0–28 nights) | Fischer et al. (2021) |
| 6 | Study length effect (2–28 days) | Fischer et al. (2021) |
| 7 | Three-shift rotating work schedule (**new**) | Kwon (under review) |

---

## SRT Formula

```
SRT = (Mean Sleep Duration / TSR) × 100
```

- **TSR (Total Sleep Range)**: span of all sleep intervals overlaid on a 24-hour clock
- Higher SRT = more temporally concentrated sleep = greater regularity
- Range: 0 (fully dispersed) to 100 (fully concentrated)

---

## Author

**Youngmoon Kwon, Ph.D.**  
Department of Physical Education, Inha University  
Incheon, Republic of Korea

---

## License

MIT License. See [LICENSE](LICENSE) for details.
