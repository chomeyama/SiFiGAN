# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Count number of parameters in Generator."""

from logging import getLogger

import hydra
from omegaconf import DictConfig

# A logger for this file
logger = getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="param_count")
def main(config: DictConfig) -> None:
    """Count number of model parameters."""

    model = hydra.utils.instantiate(config.generator)
    model.remove_weight_norm()

    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()

    logger.info(f"Number of params of {model.__class__.__name__} is : {params}")


if __name__ == "__main__":
    main()
