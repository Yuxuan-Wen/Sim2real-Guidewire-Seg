from box import Box
from configs.base_config import base_config


config = {
    "gpu_ids": "0",
    "batch_size": 1,
    "val_batchsize": 16,
    "num_workers": 4,
    "num_iters": 10,
    # "num_iters": 1250,
    "max_nums": 40,
    "num_points": 5,
    "valid_step": 250,
    "dataset": "MSD",
    "prompt": "point",
    "out_dir": "output/prompt_lr",
    "name": "train_1000_samples",
    "augment": True,
    "corrupt": None,
    "visual": False,
    "opt": {
        "learning_rate": 1e-4,
    },
    "model": {
        "type": "vit_b",
    },
}

cfg = Box(base_config)
cfg.merge_update(config)
