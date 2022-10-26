# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Snake Activation Function Module.

References:
    - BigVGAN: A Universal Neural Vocoder with Large-Scale Training
        https://arxiv.org/pdf/2206.04658.pdf
    - Reproducibility Report: Neural Networks Fail to Learn Periodic Functions and How to Fix It
        https://openreview.net/pdf?id=ysFCiXtCOj

"""

import torch
import torch.nn as nn


class Snake(nn.Module):
    """Snake activation function module."""

    def __init__(self, channels):
        """Initialize Snake module."""
        super(Snake, self).__init__()
        # learnable parameter
        self.alpha = nn.Parameter(torch.ones((1, channels, 1)))

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, channels, T).

        Returns:
            Tensor: Output tensor (B, channels, T).

        """
        return x + torch.sin(self.alpha * x) ** 2 / self.alpha
