#!/usr/bin/env python

import tensorflow as tf

class Augmenter(object):
	def __init__(self):
		pass

	def build(self):
		"""
		Return tf.keras.layers.Models for data augmentation
		which needs to be used in the first stages of the fit method
		"""
		return tf.keras.models.Sequential([
			tf.keras.layers.RandomFlip("horizontal"),
        	tf.keras.layers.RandomRotation(0.1)	]) # add some more
    

