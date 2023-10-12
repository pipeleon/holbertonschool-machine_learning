#!/usr/bin/env python3
"""Task 0 Keras"""
import numpy as np
import tensorflow as tf


class NST():
    """NST"""
    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        self.attr = 0
