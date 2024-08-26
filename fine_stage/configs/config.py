from box import Box
from configs.base_config import base_config


config = {
    "gpu_ids": "0",
    "batch_size": 2,
    "val_batchsize": 16,
    "num_workers": 4,
    "num_iters": 10,
    "warm_up_epoch": 2,
    "max_nums": 40,
    "num_points": 5,
    "valid_step": 250,
    "dataset": "phantom",
    "prompt": "box",
    "out_dir": "output/$YOUR OUTPUT DIR$r",
    "name": "$YOUR OUTPUT NAME$",
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
