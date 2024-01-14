#!/usr/bin/env python

import os
import tensorflow as tf
from run.augmenters import Augmenter

class Loader(object):
	def __init__(self, mainPath:str):
		self.mainPath = mainPath
		self.isDatasetProcessed = False
		self.isDLbuilt = False

	def _preprocess(self):
		""" 
		Internal function
		Scan the dataset in order to check if some file is corrupted: if so it's deleted
		Return None
		"""
		num_skipped = 0
		for folder_name in os.listdir(self.mainPath):
		    folder_path = os.path.join(self.mainPath, folder_name)
		    for fname in os.listdir(folder_path):
		        fpath = os.path.join(folder_path, fname)
		        try:
		            fobj = open(fpath, "rb")
		            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
		        finally:
		            fobj.close()
		        if not is_jfif:
		            num_skipped += 1
		            os.system(f'sudo rm {fpath}')  
		print(f'{num_skipped} images have been deleted')
		self.isDatasetProcessed = True

	
	def buildDataLoader(self, validation_split:float, resized_image_size:tuple, batch_size:int):
		""" Main process
		Executes the data loading from files gathered from directories:
		validation_split: (float), probability to store an image in the validation set
		resized_image_size: (tuple), resizing images at that size (default: 256, 256)
		batch_size: (int), size of data batching (default: 32)
		"""
		if not self.isDatasetProcessed:
			self._preprocess()
		train, validation = tf.keras.utils.image_dataset_from_directory(
		    directory=self.mainPath,
		    validation_split=validation_split,
		    subset="both",
		    seed=1111,
		    shuffle=False,
		    image_size=resized_image_size,
		    batch_size=batch_size)

		self.isDLbuilt = True

		data_augmentation = Augmenter().build()
		train = train.map(lambda img, label: (data_augmentation(img), label), 
    		num_parallel_calls=tf.data.AUTOTUNE)

		train = train.prefetch(tf.data.AUTOTUNE)
		validation = validation.prefetch(tf.data.AUTOTUNE)

		raise NotImplementedError 
		return train, validation