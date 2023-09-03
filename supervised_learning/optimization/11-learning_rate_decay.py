#!/usr/bin/env python3
"""Task 11 Optimization"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy
    """
    epoch = global_step // decay_step
    new_alpha = alpha / (1 + decay_rate * epoch)

    return new_alpha
