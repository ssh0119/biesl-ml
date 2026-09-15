"""Configuration for the Phase 2 CNN using the original 250 Hz signals."""

from ..plan_c.config import CLASSES

CHANNELS = ("ecg", "respiration", "microphone_energy")
SOURCE_FS = 250
TARGET_FS = 250
DOWNSAMPLE_FACTOR = SOURCE_FS // TARGET_FS
WINDOW_SECONDS = 40
WINDOW_SAMPLES = TARGET_FS * WINDOW_SECONDS

SEED = 0
BATCH_SIZE = 32
MAX_EPOCHS = 24
PATIENCE = 5
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
