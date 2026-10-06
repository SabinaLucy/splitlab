"""
core/validation.py

Pandera schemas that validate the shape of data moving between phases.

Phase 1: the simulator's event-level panel output. Every later phase that
reads simulator output should validate against this schema first, so a
silent shape change in core/simulator.py gets caught here instead of
surfacing as a confusing downstream bug in design.py or analysis.py.
"""

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

simulator_event_schema = DataFrameSchema(
    {
        "experiment_id": Column(str),
        "user_id": Column(int, Check.ge(1)),
        "variant": Column(str, Check.isin(["control", "treatment"])),
        "assignment_date": Column(str),
        "date": Column(str),
        "is_click": Column(int, Check.isin([0, 1])),
        "long_view": Column(int, Check.isin([0, 1])),
        "is_like": Column(int, Check.isin([0, 1])),
        "is_follow": Column(int, Check.isin([0, 1])),
        "watch_time": Column(float, Check.ge(0)),
        "pre_period_watch_time": Column(float, Check.ge(0)),
        "user_segment": Column(str),
        "latency_ms": Column(float, Check.gt(0)),
    },
    # strict=False: later phases are allowed to add columns (e.g. a
    # feature-engineered column in Phase 4) without breaking this check.
    # What matters here is that these core columns exist and behave.
    strict=False,
    coerce=True,
)


def validate_simulator_output(events: pd.DataFrame) -> pd.DataFrame:
    """Validate simulator event output against simulator_event_schema.

    Returns the validated (and coerced) DataFrame on success. Raises
    pandera.errors.SchemaError on failure, with a readable description of
    which column and which row broke the schema.
    """
    return simulator_event_schema.validate(events, lazy=True)
