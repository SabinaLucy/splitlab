"""
core/simulator.py

A hand-written, seeded simulator for SplitLab experiments.

Produces event-level, multi-day panel data: one row per user, per day that
user was active, for a simulated short-video ranking experiment. Every number
in here is a plant. The whole point of this file is that every later phase
(design, analysis, uplift, bandits) gets checked against a known ground
truth instead of just trusted.

What's planted, on purpose:
    - A baseline treatment effect on watch time (default +2%, per the master
      plan), which also lifts click / long_view / like / follow probabilities
      proportionally.
    - A segment effect: newer users (day_new, new_active) get a bigger lift
      than long-tenure users (30day_retention), matching the real spread
      found in notebooks/00_data_quality.ipynb ("Baseline rate by segment").
    - Five traps, each off by default and turned on one at a time via
      TrapConfig: SRM, novelty decay, a retention-drop guardrail break,
      spillover/interference, and (implicitly) peeking, which needs no
      special code since it's a property of how multi-day data gets
      analyzed, not how it's generated.

Calibration: the baseline_* defaults below were set from the real numbers in
docs/tables/data_quality_report.csv (log_random, KuaiRand-Pure). Latency is
not in the real data, so it stays a simulated value.

This module is pure core/ Python: numpy and pandas only. No FastAPI, no
dashboard code, per CLAUDE.md.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# These labels match the `user_active_degree` categories found in the real KuaiRand `user_features_pure.csv`. Using the same labels here means the segment effect this simulator plants can be compared directly against
# whatever Phase 4's uplift model recovers from simulated data.

SEGMENTS = (
    "day_new",
    "new_active",
    "low_active",
    "middle_active",
    "high_active",
    "full_active",
    "30day_retention",
)

# Rough relative size of each segment in the simulated population. Not fit to the real distribution, edit this if you want to match it exactly.
DEFAULT_SEGMENT_WEIGHTS = {
    "day_new": 0.05,
    "new_active": 0.10,
    "low_active": 0.15,
    "middle_active": 0.20,
    "high_active": 0.20,
    "full_active": 0.20,
    "30day_retention": 0.10,
}

# How much extra lift each segment gets from the treatment, as a multiplier on the base treatment effect. 1.0 = no extra lift, below 1.0 = less lift than average, above 1.0 = more. This is what Phase 4's uplift model is
# supposed to recover. Shaped qualitatively off the real finding in 00_data_quality.ipynb: day_new had the highest baseline click rate,
# 30day_retention the lowest, so newer users are given more room to benefit.
DEFAULT_SEGMENT_EFFECT_MULTIPLIER = {
    "day_new": 2.5,
    "new_active": 1.8,
    "low_active": 1.2,
    "middle_active": 1.0,
    "high_active": 0.6,
    "full_active": 0.4,
    "30day_retention": 0.1,
}

# Used to give segments different baseline watch-time levels (both in the
# pre-period and during the experiment), independent of any treatment effect.
DEFAULT_SEGMENT_LEVEL_MULTIPLIER = {
    "day_new": 0.6,
    "new_active": 0.8,
    "low_active": 0.9,
    "middle_active": 1.0,
    "high_active": 1.2,
    "full_active": 1.4,
    "30day_retention": 1.6,
}


@dataclass
class TrapConfig:
    """Which traps are planted, and how strong each one is.

    Every trap defaults to off. Turn on exactly the trap you're testing a
    detector against, one at a time, so you always know which planted
    problem a given method is supposed to catch.
    """

    # Sample ratio mismatch: skew assignment away from 50/50 so a chi-square
    # SRM check should fail. 0.5 = no skew (the default, healthy split).
    # 0.55 means treatment gets 55% of users instead of 50%.
    srm_treatment_share: float = 0.5

    # Novelty effect: the treatment effect decays toward zero over the course
    # of the experiment. 0.0 = constant effect every day (the default). 1.0 =
    # the effect has fully decayed to zero by the last day.
    novelty_decay: float = 0.0

    # Broken guardrail: treatment users become less likely to return the next
    # day, even though their watch time while active still improves. This is
    # the retention-drop trap Phase 3B's decision system is supposed to catch
    # and return as HOLD rather than SHIP. Expressed as an absolute drop in
    # daily active probability for treatment users, e.g. 0.05 = 5 percentage
    # points lower than control.
    retention_drop: float = 0.0

    # Spillover / interference: this fraction of control users are actually
    # contaminated by treatment (e.g. they have a friend in treatment) and
    # receive spillover_strength of the full treatment effect, even though
    # they're nominally in control. 0.0 = no spillover (the default).
    spillover_fraction: float = 0.0
    spillover_strength: float = 0.0

    # Peeking is not a data-generation trap: it's a property of how someone
    # *analyzes* multi-day data (checking cumulative significance every day
    # instead of once, at a pre-committed sample size). As long as
    # experiment_days > 1, the output of this simulator is already enough to
    # demonstrate peeking inflation in Phase 3. Nothing to configure here.


@dataclass
class SimulatorConfig:
    """Everything needed to generate one reproducible simulated experiment.

    The baseline_* fields were calibrated against the real numbers in
    docs/tables/data_quality_report.csv. Latency is simulated only.
    """

    seed: int = 42
    n_users: int = 20_000
    experiment_days: int = 14

    # Baseline rates, calibrated from docs/tables/data_quality_report.csv.
    baseline_click_rate: float = 0.176
    # Long view is only drawn for clicked rows, so this is P(long_view | click).
    baseline_long_view_rate: float = 0.48
    baseline_like_rate: float = 0.0108
    baseline_follow_rate: float = 0.00074
    # Median of the lognormal for a middle_active user day, in seconds.
    baseline_watch_time_s: float = 10.0
    baseline_daily_active_prob: float = 0.35  # chance a user returns on a given day
    baseline_latency_ms: float = 120.0  # simulated only, not present in real data

    # Spread of user day watch time on the log scale. The default keeps the
    # original simulator behavior. The real log_random data measured about
    # 1.56, so use that when realistic noise matters (power, CUPED gain).
    watch_time_log_sd: float = 0.5

    # The planted treatment effect on the primary metric (watch time), and
    # the click / long_view / like / follow probabilities, which move with it
    # proportionally.
    true_effect_pct: float = 0.02  # +2%, per the master plan

    segment_weights: dict = field(default_factory=lambda: dict(DEFAULT_SEGMENT_WEIGHTS))
    segment_effect_multiplier: dict = field(
        default_factory=lambda: dict(DEFAULT_SEGMENT_EFFECT_MULTIPLIER)
    )
    segment_level_multiplier: dict = field(
        default_factory=lambda: dict(DEFAULT_SEGMENT_LEVEL_MULTIPLIER)
    )

    traps: TrapConfig = field(default_factory=TrapConfig)

    experiment_id: str = "exp_ranking_v1"
    assignment_date: str = "2026-01-01"

    def aa_mode(self) -> "SimulatorConfig":
        """Return a copy of this config with the treatment effect zeroed out.

        Use this to generate an A/A test: real randomization, real sampling
        noise, no real effect. Any "winner" Phase 3's analysis finds on data
        from this config is a false positive by construction, which is
        exactly what notebooks/03_coverage_and_aa.ipynb checks for.
        """
        cfg = copy.deepcopy(self)
        cfg.true_effect_pct = 0.0
        return cfg


def _assign_segments(rng: np.random.Generator, n_users: int, weights: dict) -> np.ndarray:
    segs = list(weights.keys())
    probs = np.array(list(weights.values()), dtype=float)
    probs = probs / probs.sum()
    return rng.choice(segs, size=n_users, p=probs)


def _assign_variant(rng: np.random.Generator, n_users: int, treatment_share: float) -> np.ndarray:
    draws = rng.random(n_users)
    return np.where(draws < treatment_share, "treatment", "control")


def simulate_experiment(config: SimulatorConfig) -> tuple[pd.DataFrame, dict]:
    """Generate one reproducible simulated experiment.

    Returns
    -------
    events : pd.DataFrame
        Event-level, multi-day panel data. One row per user, per day that
        user was active. Columns: experiment_id, user_id, variant,
        assignment_date, date, is_click, long_view, is_like, is_follow,
        watch_time, pre_period_watch_time, user_segment, latency_ms.
    ground_truth : dict
        Every planted value used to generate this data: the true effect
        size, the segment multipliers, which traps were active and how
        strong, and the realized treatment/control counts. Every later
        phase that checks "did my method recover the right answer" checks
        against this dict, not against the config object directly.
    """
    rng = np.random.default_rng(config.seed)

    user_ids = np.arange(1, config.n_users + 1)
    segments = _assign_segments(rng, config.n_users, config.segment_weights)
    variant = _assign_variant(rng, config.n_users, config.traps.srm_treatment_share)

    # Spillover: a fraction of control users are secretly contaminated.
    is_spillover = (variant == "control") & (
        rng.random(config.n_users) < config.traps.spillover_fraction
    )

    assignment_date = pd.Timestamp(config.assignment_date)

    # Pre-period watch time: correlated with segment, no treatment effect at
    # all, since this happens before assignment exists. This is the column
    # CUPED regresses on in Phase 2.
    level_mult = np.array([config.segment_level_multiplier[s] for s in segments])
    pre_period_watch_time = rng.lognormal(
        mean=np.log(config.baseline_watch_time_s * level_mult),
        sigma=config.watch_time_log_sd,
    )

    rows: list[pd.DataFrame] = []

    for day_offset in range(config.experiment_days):
        date = assignment_date + pd.Timedelta(days=day_offset)

        # Novelty decay: the treatment effect shrinks over the experiment.
        if config.experiment_days > 1:
            decay_progress = day_offset / (config.experiment_days - 1)
        else:
            decay_progress = 0.0
        novelty_mult = 1.0 - config.traps.novelty_decay * decay_progress

        # Per-user daily active probability. This is what retention measures.
        active_prob = np.full(config.n_users, config.baseline_daily_active_prob)
        treat_mask = variant == "treatment"
        active_prob[treat_mask] -= config.traps.retention_drop
        active_prob = np.clip(active_prob, 0.01, 0.99)

        is_active = rng.random(config.n_users) < active_prob
        if not is_active.any():
            continue

        active_idx = np.where(is_active)[0]
        n_active = active_idx.size

        seg_active = segments[active_idx]
        var_active = variant[active_idx]
        spill_active = is_spillover[active_idx]
        level_mult_active = level_mult[active_idx]

        # Effective treatment effect per active user on this day.
        seg_effect_mult = np.array(
            [config.segment_effect_multiplier[s] for s in seg_active]
        )
        receives_full_effect = var_active == "treatment"
        receives_spillover_effect = spill_active

        effect = np.zeros(n_active)
        effect[receives_full_effect] = (
            config.true_effect_pct * seg_effect_mult[receives_full_effect] * novelty_mult
        )
        effect[receives_spillover_effect] = (
            config.true_effect_pct
            * seg_effect_mult[receives_spillover_effect]
            * novelty_mult
            * config.traps.spillover_strength
        )

        # Watch time: lognormal around a segment and effect adjusted mean.
        mean_watch_time = config.baseline_watch_time_s * level_mult_active * (1.0 + effect)
        watch_time = rng.lognormal(
            mean=np.log(mean_watch_time), sigma=config.watch_time_log_sd
        )

        # Click, long_view, like and follow, lifted by the same effect.
        click_prob = np.clip(config.baseline_click_rate * (1.0 + effect), 0.001, 0.999)
        is_click = rng.random(n_active) < click_prob

        long_view_prob = np.clip(
            config.baseline_long_view_rate * (1.0 + effect), 0.001, 0.999
        )
        long_view = (rng.random(n_active) < long_view_prob) & is_click

        like_prob = np.where(
            long_view, config.baseline_like_rate * 2, config.baseline_like_rate * 0.3
        )
        is_like = rng.random(n_active) < like_prob

        follow_prob = np.where(
            long_view, config.baseline_follow_rate * 2, config.baseline_follow_rate * 0.2
        )
        is_follow = rng.random(n_active) < follow_prob

        # Latency is simulated only, never present in the real data.
        latency_ms = rng.normal(config.baseline_latency_ms, 15, size=n_active)
        if config.traps.retention_drop > 0:
            # A plausible co-symptom of the same broken release that hurt
            # retention. Mild, and only shows up when that trap is on.
            latency_ms = latency_ms + np.where(var_active == "treatment", 10.0, 0.0)
        latency_ms = np.clip(latency_ms, 1, None)

        rows.append(
            pd.DataFrame(
                {
                    "experiment_id": config.experiment_id,
                    "user_id": user_ids[active_idx],
                    "variant": var_active,
                    "assignment_date": assignment_date.strftime("%Y-%m-%d"),
                    "date": date.strftime("%Y-%m-%d"),
                    "is_click": is_click.astype(int),
                    "long_view": long_view.astype(int),
                    "is_like": is_like.astype(int),
                    "is_follow": is_follow.astype(int),
                    "watch_time": watch_time,
                    "pre_period_watch_time": pre_period_watch_time[active_idx],
                    "user_segment": seg_active,
                    "latency_ms": latency_ms,
                }
            )
        )

    events = (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(
            columns=[
                "experiment_id", "user_id", "variant", "assignment_date", "date",
                "is_click", "long_view", "is_like", "is_follow", "watch_time",
                "pre_period_watch_time", "user_segment", "latency_ms",
            ]
        )
    )

    ground_truth = {
        "seed": config.seed,
        "n_users": config.n_users,
        "experiment_days": config.experiment_days,
        "true_effect_pct": config.true_effect_pct,
        "segment_effect_multiplier": dict(config.segment_effect_multiplier),
        "srm_treatment_share": config.traps.srm_treatment_share,
        "novelty_decay": config.traps.novelty_decay,
        "retention_drop": config.traps.retention_drop,
        "spillover_fraction": config.traps.spillover_fraction,
        "spillover_strength": config.traps.spillover_strength,
        "n_treatment_users": int((variant == "treatment").sum()),
        "n_control_users": int((variant == "control").sum()),
        "n_spillover_contaminated_control_users": int(is_spillover.sum()),
    }

    return events, ground_truth


def simulate_aa_experiment(config: SimulatorConfig) -> tuple[pd.DataFrame, dict]:
    """Convenience wrapper: simulate_experiment(config.aa_mode())."""
    return simulate_experiment(config.aa_mode())


if __name__ == "__main__":
    # Quick manual smoke test: python -m core.simulator
    cfg = SimulatorConfig(n_users=2_000, experiment_days=7)
    events_df, truth = simulate_experiment(cfg)
    print("events shape:", events_df.shape)
    print("columns:", list(events_df.columns))
    print("ground truth:", truth)
    print(events_df.head())
