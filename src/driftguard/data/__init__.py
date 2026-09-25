"""Loading and temporal alignment of network-flow datasets.

Every dataset is mapped onto one common flow schema (COMMON_SCHEMA) so that
drift statistics and models are comparable across sources. The mapping is
explicit per dataset: nothing is inferred by column position.
"""

from driftguard.data.schema import (
    CATEGORICAL_FIELDS,
    COMMON_SCHEMA,
    FEATURE_GROUPS,
    FlowFrame,
    add_derived_features,
    load_common_frame,
)

__all__ = [
    "CATEGORICAL_FIELDS",
    "COMMON_SCHEMA",
    "FEATURE_GROUPS",
    "FlowFrame",
    "add_derived_features",
    "load_common_frame",
]
