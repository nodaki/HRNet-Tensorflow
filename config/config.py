from omegaconf import OmegaConf
import os

cfg = OmegaConf.load(os.path.join(os.getenv("PROJECT_DIR"), "config/config.yaml"))
