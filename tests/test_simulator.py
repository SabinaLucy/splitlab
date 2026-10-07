"""
tests/test_simulator.py

Tests for core/simulator.py. Covers:
    - seed reproducibility
    - each planted trap (SRM, novelty, broken guardrail / retention drop,
      spillover) actually triggers under the right config
    - peeking: not a trap to trigger, but a test that the simulator produces
      the multi-day, growing-sample-size structure that makes peeking
      possible to demonstrate in Phase 3
    - output matches the pandera event schema (simulator_event_schema)
    - the output is a multi-day panel: one row per user, per day that user
      was active, never a duplicate (user_id, date) pair

Statistical thresholds below (margins, tolerances) were calibrated by running
the simulator directly at several sample sizes and picking values comfortably
inside the observed signal, not right at the noise floor, so these tests
should be stable in CI and not flaky, while still catching a real
regression in core/simulator.py.
"""

import numpy as np
import pandas as pd
import pytest

from core.simulator import SimulatorConfig, simulate_experiment, simulate_aa_experiment
from core.validation import simulator_event_schema, validate_simulator_output



# Helpers shared across tests


def _daily_retention_by_variant(events: pd.DataFrame) -> pd.DataFrame:
    """Day N -> day N+1 return rate per variant, same logic as
    notebooks/01_simulator.ipynb."""
    dates = sorted(events["date"].unique())
    rows = []
    for variant in ["control", "treatment"]:
        for i in range(len(dates) - 1):
            today_users = set(
                events.loc[
                    (events["date"] == dates[i]) & (events["variant"] == variant),
                    "user_id",
                ]
            )
            tomorrow_users = set(events.loc[events["date"] == dates[i + 1], "user_id"])
            if not today_users:
                continue
            retained = len(today_users & tomorrow_users) / len(today_users)
            rows.append({"date": dates[i], "variant": variant, "retention": retained})
    return pd.DataFrame(rows)


def _average_daily_lift(novelty_decay: float, n_reps: int, n_users: int, days: int) -> np.ndarray:
    """Average the daily treatment/control lift across several seeds, same
    approach as notebooks/01_simulator.ipynb, since a single run is too
    noisy at a small sample size to see the decay shape reliably."""
    lifts = []
    for seed in range(n_reps):
        cfg = SimulatorConfig(n_users=n_users, experiment_days=days, seed=seed)
        cfg.traps.novelty_decay = novelty_decay
        ev, _ = simulate_experiment(cfg)
        daily = ev.groupby(["date", "variant"])["watch_time"].mean().unstack("variant")
        daily["lift"] = daily["treatment"] / daily["control"] - 1
        lifts.append(daily["lift"].values)
    return np.mean(lifts, axis=0)



# Reproducibility

def test_same_seed_reproduces_identical_output():
    config = SimulatorConfig(n_users=2_000, experiment_days=5, seed=42)
    events_a, truth_a = simulate_experiment(config)
    events_b, truth_b = simulate_experiment(config)

    pd.testing.assert_frame_equal(events_a, events_b)
    assert truth_a == truth_b


def test_different_seed_gives_different_output():
    config_a = SimulatorConfig(n_users=2_000, experiment_days=5, seed=1)
    config_b = SimulatorConfig(n_users=2_000, experiment_days=5, seed=2)

    events_a, _ = simulate_experiment(config_a)
    events_b, _ = simulate_experiment(config_b)

    assert not events_a.equals(events_b)



# A/A mode

def test_aa_mode_zeroes_true_effect_without_mutating_original_config():
    config = SimulatorConfig(n_users=1_000, experiment_days=3, true_effect_pct=0.02)
    aa_config = config.aa_mode()

    assert aa_config.true_effect_pct == 0.0
    # the original config must be untouched (aa_mode returns a copy)
    assert config.true_effect_pct == 0.02


def test_simulate_aa_experiment_matches_manual_aa_mode():
    config = SimulatorConfig(n_users=1_000, experiment_days=3, seed=7, true_effect_pct=0.02)
    events_a, truth_a = simulate_aa_experiment(config)
    events_b, truth_b = simulate_experiment(config.aa_mode())

    pd.testing.assert_frame_equal(events_a, events_b)
    assert truth_a == truth_b
    assert truth_a["true_effect_pct"] == 0.0



# Trap: SRM


def test_default_assignment_is_close_to_fifty_fifty():
    config = SimulatorConfig(n_users=5_000, experiment_days=1, seed=1)
    _, truth = simulate_experiment(config)

    realized_share = truth["n_treatment_users"] / config.n_users
    assert abs(realized_share - 0.5) < 0.05


def test_srm_trap_skews_assignment_away_from_fifty_fifty():
    config = SimulatorConfig(n_users=5_000, experiment_days=1, seed=1)
    config.traps.srm_treatment_share = 0.55
    _, truth = simulate_experiment(config)

    realized_share = truth["n_treatment_users"] / config.n_users
    # should land near 0.55, and clearly away from the healthy 0.50 split
    assert abs(realized_share - 0.55) < 0.05
    assert realized_share > 0.52



# Trap: novelty decay

def test_novelty_decay_shrinks_the_effect_toward_zero_over_the_experiment():
    n_reps, n_users, days = 8, 4_000, 10

    no_decay_lift = _average_daily_lift(novelty_decay=0.0, n_reps=n_reps, n_users=n_users, days=days)
    decay_lift = _average_daily_lift(novelty_decay=1.0, n_reps=n_reps, n_users=n_users, days=days)

    no_decay_drop = no_decay_lift[0] - no_decay_lift[-1]
    decay_drop = decay_lift[0] - decay_lift[-1]

    # Both start at roughly the same place, since day 0 has no decay applied
    # yet either way (decay_progress = 0 on day 0 regardless of the decay
    # setting).
    assert no_decay_lift[0] == pytest.approx(decay_lift[0], abs=1e-9)

    # With decay on, the lift should fall off noticeably more than it does
    # under ordinary day-to-day noise alone.
    assert decay_drop > no_decay_drop + 0.008



# Trap: broken guardrail (retention drop)
def test_retention_drop_trap_lowers_next_day_return_rate_in_treatment():
    config = SimulatorConfig(n_users=8_000, experiment_days=5, seed=1)
    config.traps.retention_drop = 0.10
    events, _ = simulate_experiment(config)

    retention = _daily_retention_by_variant(events)
    mean_retention = retention.groupby("variant")["retention"].mean()

    assert mean_retention["treatment"] < mean_retention["control"] - 0.03


def test_without_the_trap_retention_is_not_meaningfully_different_by_variant():
    config = SimulatorConfig(n_users=8_000, experiment_days=5, seed=1)
    # retention_drop defaults to 0.0, trap is off
    events, _ = simulate_experiment(config)

    retention = _daily_retention_by_variant(events)
    mean_retention = retention.groupby("variant")["retention"].mean()

    assert abs(mean_retention["treatment"] - mean_retention["control"]) < 0.03


def test_retention_drop_trap_does_not_hide_the_primary_metric_win():
    """This is the whole point of the trap: watch time still looks like a
    win even though retention broke, which is why a decision rule that only
    checks the primary metric would wrongly ship it."""
    config = SimulatorConfig(n_users=8_000, experiment_days=5, seed=1)
    config.traps.retention_drop = 0.10
    events, _ = simulate_experiment(config)

    mean_watch_time = events.groupby("variant")["watch_time"].mean()
    assert mean_watch_time["treatment"] > mean_watch_time["control"]



# Trap: spillover
def test_zero_spillover_fraction_contaminates_nobody():
    config = SimulatorConfig(n_users=3_000, experiment_days=3, seed=1)
    # spillover_fraction defaults to 0.0
    _, truth = simulate_experiment(config)

    assert truth["n_spillover_contaminated_control_users"] == 0


def test_spillover_fraction_contaminates_roughly_the_configured_share():
    config = SimulatorConfig(n_users=5_000, experiment_days=3, seed=1)
    config.traps.spillover_fraction = 0.20
    config.traps.spillover_strength = 0.5
    _, truth = simulate_experiment(config)

    realized_fraction = truth["n_spillover_contaminated_control_users"] / truth["n_control_users"]
    assert abs(realized_fraction - 0.20) < 0.05


def test_spillover_only_changes_contaminated_control_rows_everything_else_is_identical():
    """With the same seed, turning spillover on should not change a single
    treatment row, and should leave every non-contaminated control row
    bit-identical to the no-spillover run. Only the contaminated control
    rows should actually change, since that's the only thing spillover is
    supposed to touch."""
    clean_config = SimulatorConfig(n_users=3_000, experiment_days=4, seed=7)
    clean_config.traps.spillover_fraction = 0.0
    clean_events, _ = simulate_experiment(clean_config)

    spillover_config = SimulatorConfig(n_users=3_000, experiment_days=4, seed=7)
    spillover_config.traps.spillover_fraction = 0.25
    spillover_config.traps.spillover_strength = 0.6
    spillover_events, spillover_truth = simulate_experiment(spillover_config)

    merged = clean_events.merge(
        spillover_events, on=["user_id", "date"], suffixes=("_clean", "_spillover")
    )

    treatment_rows = merged[merged["variant_clean"] == "treatment"]
    assert np.allclose(treatment_rows["watch_time_clean"], treatment_rows["watch_time_spillover"])

    control_rows = merged[merged["variant_clean"] == "control"]
    differs = ~np.isclose(control_rows["watch_time_clean"], control_rows["watch_time_spillover"])
    n_contaminated_rows_seen = control_rows.loc[differs, "user_id"].nunique()

    # Every contaminated user shows up as "different" on at least one of
    # their active days, but a contaminated user who happened to be
    # inactive on every single day in this short window wouldn't produce a
    # visible diff. So this should be close to, but can be a little below,
    # the ground truth contaminated count, never above it.
    assert 0 < n_contaminated_rows_seen <= spillover_truth["n_spillover_contaminated_control_users"]
    assert n_contaminated_rows_seen > 0.7 * spillover_truth["n_spillover_contaminated_control_users"]



# Peeking: not a trap to trigger, but the structural property that makes peeking possible

def test_output_supports_peeking_style_cumulative_analysis():
    """Peeking isn't something the simulator plants, it's a property of how
    multi-day data gets analyzed. This test confirms the simulator actually
    produces the thing peeking exploits: a sample size that grows day by
    day, which is what Phase 3's sequential testing module is tested
    against."""
    config = SimulatorConfig(n_users=3_000, experiment_days=10, seed=1)
    events, _ = simulate_experiment(config)

    distinct_dates = sorted(events["date"].unique())
    assert len(distinct_dates) == config.experiment_days

    cumulative_users_seen = []
    seen = set()
    for date in distinct_dates:
        seen |= set(events.loc[events["date"] == date, "user_id"])
        cumulative_users_seen.append(len(seen))

    # non-decreasing every day, and strictly more by the end than the start,
    # which is exactly the "sample size keeps growing" situation that makes
    # checking significance every day tempting, and dangerous
    assert all(
        cumulative_users_seen[i] <= cumulative_users_seen[i + 1]
        for i in range(len(cumulative_users_seen) - 1)
    )
    assert cumulative_users_seen[-1] > cumulative_users_seen[0]



# Schema validation

def test_default_simulator_output_passes_schema_validation():
    config = SimulatorConfig(n_users=1_000, experiment_days=3, seed=1)
    events, _ = simulate_experiment(config)

    validated = validate_simulator_output(events)
    assert len(validated) == len(events)


def test_schema_rejects_an_invalid_variant_value():
    config = SimulatorConfig(n_users=200, experiment_days=1, seed=1)
    events, _ = simulate_experiment(config)

    broken = events.copy()
    broken.loc[broken.index[0], "variant"] = "not_a_real_variant"

    with pytest.raises(Exception):
        simulator_event_schema.validate(broken, lazy=True)


def test_schema_rejects_a_negative_watch_time():
    config = SimulatorConfig(n_users=200, experiment_days=1, seed=1)
    events, _ = simulate_experiment(config)

    broken = events.copy()
    broken.loc[broken.index[0], "watch_time"] = -5.0

    with pytest.raises(Exception):
        simulator_event_schema.validate(broken, lazy=True)



# Panel shape
def test_output_is_a_multi_day_panel_with_no_duplicate_user_day_rows():
    config = SimulatorConfig(n_users=2_000, experiment_days=7, seed=1)
    events, _ = simulate_experiment(config)

    assert events["date"].nunique() > 1

    duplicate_user_days = events.duplicated(subset=["user_id", "date"]).sum()
    assert duplicate_user_days == 0


def test_every_expected_column_is_present():
    config = SimulatorConfig(n_users=200, experiment_days=1, seed=1)
    events, _ = simulate_experiment(config)

    expected_columns = {
        "experiment_id", "user_id", "variant", "assignment_date", "date",
        "is_click", "long_view", "is_like", "is_follow", "watch_time",
        "pre_period_watch_time", "user_segment", "latency_ms",
    }
    assert expected_columns.issubset(set(events.columns))


def test_pre_period_watch_time_is_constant_per_user():
    """pre_period_watch_time happens before assignment, so it should be the
    same value on every row for a given user, regardless of which day of
    the experiment that row is for."""
    config = SimulatorConfig(n_users=1_000, experiment_days=5, seed=1)
    events, _ = simulate_experiment(config)

    per_user_unique_values = events.groupby("user_id")["pre_period_watch_time"].nunique()
    assert (per_user_unique_values == 1).all()
