"""
Configuration for behavior direction ablation pipeline.

Unlike refusal_direction which uses harmful/harmless splits,
this pipeline uses CAA-style A/B datasets for specific behaviors.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple, List

@dataclass
class Config:
    model_alias: str
    model_path: str
    
    # Behavior to study (must match CAA dataset folder names)
    behavior: str = "survival-instinct"
    
    # Dataset sizes
    n_generate: int = 500  # Number of A/B pairs for direction generation
    n_test: int = 50       # Number of A/B pairs for testing (uses full test set if smaller)
    
    # Direction generation settings
    batch_size: int = 8
    
    # Generation settings for open-ended evaluation
    max_new_tokens: int = 512
    
    # Steering multipliers to test
    steering_multipliers: Tuple[float, ...] = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
    
    # Layer selection settings
    prune_layer_percentage: float = 0.2  # Discard directions from last 20% of layers
    
    def artifact_path(self) -> str:
        """Path to save all artifacts for this model."""
        return os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 
            "runs", 
            self.model_alias,
            self.behavior
        )
