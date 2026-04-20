"""Adapter modules for asymmetric query-side alignment."""

from .identical_text_alignment import (
    AlignmentConfig,
    train_identical_text_alignment,
    LinearAdapter,
)

from .label_training import (
    LabelTrainingConfig,
    train_with_labels,
    split_queries_by_id,
    create_training_pairs,
)

from .era_training import (
    ERAConfig,
    train_era,
    TrainingMode,
)

from .adapted_embedder import AdaptedQueryEmbedder

__all__ = [
    # Alignment
    "AlignmentConfig",
    "train_identical_text_alignment",
    "LinearAdapter",
    # Label training
    "LabelTrainingConfig",
    "train_with_labels",
    "split_queries_by_id",
    "create_training_pairs",
    # ERA training
    "ERAConfig",
    "train_era",
    "TrainingMode",
    # Embedder
    "AdaptedQueryEmbedder",
]
