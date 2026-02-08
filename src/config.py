"""Sampling defaults and constants for the hierarchical MMM demo."""

# Sampling
DRAWS = 500
TUNE = 500
CHAINS = 4
TARGET_ACCEPT_UNPOOLED = 0.8
TARGET_ACCEPT_CENTERED = 0.9
TARGET_ACCEPT_NONCENTERED = 0.95
PRIOR_SAMPLES = 1000

# HDI
HDI_PROB = 0.94

# Tuned non-centered model: multiplier for c10 (smaller = more pooling)
TUNED_MULTIPLIER = 0.05
