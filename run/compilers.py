#!/usr/bin/env python
import tensorflow as tf
#import tensorflow_addons as tfa


class Compiler(object):
	def __init__(self, alpha, doesDecay:bool=False, decay_steps:int=None,
				decay_rate:float=None, staircase:bool=False):
		self.alpha = alpha
		self.doesDecay = doesDecay
		self.decay_steps = decay_steps
		self.decay_rate = decay_rate
		self.staircase = staircase
		self.loss, self.metrics = None,None  # will be defined later in sub-classes

		if self.doesDecay:
			if not self.decay_steps and not self.decay_rate:
				raise Exception('You need to add both decay parameters!')
			elif not self.decay_steps:
				raise Exception('decay_steps is missing!')
			elif not self.decay_rate:
				raise Exception('decay_rate is missing')

			assert type(self.decay_steps)==int and \
				type(self.decay_rate)==float, 'Check input type!'

			exp_lr = tf.keras.optimizers.schedules.ExponentialDecay(
			 	self.alpha, 
			 	decay_steps=self.decay_steps, 
			 	decay_rate=self.decay_rate, 
			 	staircase=self.staircase)
			self.optimizer = tf.keras.optimizers.Adam(learning_rate=exp_lr)
		else:
			self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha)

	def getOptimizer(self):
		""" Return the current Compiler's optimizer """
		return self.optimizer

	def getLoss(self):
		""" Return the current Compiler's loss function """
		return self.loss

	def getMetrics(self):
		""" Return the current Compiler's evaluation metrics """
		return self.metrics

	def __str__(self):
		return f'Compiler features:\n \
			learning rate:  {self.alpha}\n \
			learning decay: {self.doesDecay}\n \
			Optimizer:      {self.optimizer.name}\n \
			Loss:           {self.loss.name}\n \
			Metrics:        {str(self.metrics)}\n'
	
		
class CrossEntropy(Compiler):
	def __init__(self, alpha, doesDecay:bool=False, decay_steps:int=None,
				decay_rate:float=None, staircase:bool=False):
		super().__init__(alpha, doesDecay, decay_steps,
				decay_rate, staircase)
		self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
		self.metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]


class TripletLoss(Compiler):
	def __init__(self, alpha, doesDecay:bool=False, decay_steps:int=None,
				decay_rate:float=None, staircase:bool=False):
		super().__init__(alpha, doesDecay, decay_steps,
				decay_rate, staircase)
		self.loss = tfa.losses.TripletSemiHardLoss()
		#self.metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]