#!/usr/bin/env python
import os
import cv2
import time
from PIL import Image
import numpy as np
import tensorflow as tf


def normalize_input(img, normalization):
	"""
	Normalize input image.
	(credits: DeepFace)
	(github: https://github.com/serengil/deepface/blob/master/deepface/commons/functions.py#L238)
	Args:
		img (numpy array): the input image.
		normalization: (str), which method to use in order to standardize the input image.
							allowed ('base','raw','Facenet','Facenet2018',
										'VGGFace','VGGFace2','ArcFace','SFace')
							SFace allowed ('_v1', '_v2', ..., '_v5')
	Returns:
		numpy array: the normalized image.
	"""
	if normalization == "base":
		return img

	if normalization == "raw":
		pass  # return just restored pixels

	elif normalization == "Facenet":
		tmp_mean, tmp_std = img.mean(), img.std()
		img = (img - tmp_mean) / tmp_std

	elif normalization == "Facenet2018":
		# simply / 127.5 - 1 (similar to facenet 2018 model preprocessing step as @iamrishab posted)
		img /= 127.5
		img -= 1

	elif normalization == "VGGFace":
		# mean subtraction based on VGGFace1 training data
		img[..., 0] -= 93.5940
		img[..., 1] -= 104.7624
		img[..., 2] -= 129.1863

	elif normalization == "VGGFace2":
		# mean subtraction based on VGGFace2 training data
		img[..., 0] -= 91.4953
		img[..., 1] -= 103.8827
		img[..., 2] -= 131.0912

	elif normalization == "ArcFace":
		# Reference study: The faces are cropped and resized to 112×112,
		# and each pixel (ranged between [0, 255]) in RGB images is normalised
		# by subtracting 127.5 then divided by 128.
		img -= 127.5
		img /= 128
		
	elif normalization == "ArcFace2":
		img /= 255

	elif normalization.startswith("SFace"):
		if normalization.endswith('v0')
			return img
		elif normalization.endswith('v1'): 
			img /= 255
		elif normalization.endswith('v2'): # as in ArcFace
			img -= 127.5
			img /= 128
		elif normalization.endswith('v4'): # as in Facenet
			tmp_mean, tmp_std = img.mean(), img.std()
			img = (img - tmp_mean) / tmp_std
		elif normalization.endswith('v5'): # as in Facenet2018
			img /= 127.5
			img -= 1

	else:
		raise ValueError(f"unimplemented normalization type - {normalization}")

	return img


def processImage(image, outputDim, normalization):
	""" 
	Transform an image into a standardized and resized one.
	These are the required steps in order to build the Query/Test matrix dataset
	Args:
		image: (str), filepath to image
		outputDim: (list), of output dimension
		normalization: (str), which method to use in order to standardize the input image.
							allowed ('base','raw','Facenet','Facenet2018',
										'VGGFace','VGGFace2','ArcFace', SFace')
	Return:
		standardized and rescaled tf.tensor according to the model input dimension
	"""
	# loading image
	img = np.asarray(Image.open(image).convert('RGB'))

	# set new type ( not that sure to leave it or not )
	img = img.astype('float32') 	
	
	# normalized according to the parameter specified (check it out for FaceNet/ArcFace)
	processed = normalize_input(img, normalization)
	
	# expanding dimension to match the bath dimension
	processed = tf.expand_dims(processed, axis=0)

	# resizing according to the model 
	processed = tf.image.resize(processed, outputDim)

	return processed


def generateDataset(folder, modelInputShape, normalizationOpt):
	"""
	Args:
		folder: (str), where to load images
		modelInputShape: (tuple) of (ImageSize, ImageSize) output resizing dimension
		normalizationOpt: (str), which method to use in order to standardize the input image.
							allowed ('base','raw','Facenet','Facenet2018',
										'VGGFace','VGGFace2','ArcFace')
	Return a dataset matrix of shape (nImage, ImageSize, ImageSize, Nchannel)
	and the list of original names
	"""
	# assert normalizationOpt in ('base','raw','Facenet','Facenet2018','VGGFace','VGGFace2','ArcFace'), "Not a valid normalization Option!"
	print(f'Loading datasets from folder: {folder} ...')
	print(f'Testing normalization: {normalizationOpt} ...')
	filelistdir = sorted(os.listdir(folder))
	dataset, orignames = [], []	
	for i,image in enumerate(filelistdir):
		orignames.append(get_orig_filename(image))
		standardized = processImage(image = os.path.join(folder,image), 
									outputDim = list(modelInputShape),
									normalization = normalizationOpt)
		dataset.append(standardized)
	tmp = np.asarray(dataset).astype("float32")
	if len(tmp.shape)>4:
		shapes = tmp.shape
		tmp.resize(int(shapes[0]), int(shapes[2]), int(shapes[3]), int(shapes[4]))
	return tmp, orignames


def load_dataset_from_folder(foldername, modelShape, batch_size=32):
	"""
	Args:
		foldername: (str) name of the folder (e.g. 'query_set' or 'test_set'),
			or directory
	Return: 
		tf.data.Dataset object
		sorted list of filenames (alphanumeric)
	Structure should be organized as follows
	main_directory/
	...query_set/
	......images/
	.........a_image_1.jpg
	.........a_image_2.jpg
	...test_set/
	......images/
	.........b_image_1.jpg
	.........b_image_2.jpg

	"""
	filelist = os.listdir(os.path.join(foldername,'images'))
	dataset = tf.keras.utils.image_dataset_from_directory(
		directory=foldername,
		shuffle=False,
		image_size=modelShape, 
		batch_size=batch_size)
	return dataset, sorted(filelist)


def get_orig_filename(filename):
	"""
	This function takes as input any filename and it checks whether the name is 
	in original or "cropped format" regardless the extension, like 
		original:       imagefilename.jpg
		cropped format: imagefilename_crop_?.jpg 
	where ? is the i-th cropped face found.

	Return: 
		original: imagefilename.jpg
	"""
	extension = filename.split('.')[-1]
	if len(filename.split('_crop_')) > 1:
		return f"{str(filename.split('_crop_')[0])}.{str(extension)}"
	return filename


def compute_distance(query_set, query_filelist, test_set, test_filelist,
	distance_metric='euclidean'):
	"""
	Args:
		query_set: (np.array) of embedded features shape (n_image, feature_dim)
		query_filelist: (list) of sorted query file names
		test_set:  (np.array) of embedded features shape (n_image, feature_dim)
		test_filelist: (list) of sorted test file names
		distance_metric: (str)  available ('euclidean', 'cosine', ... )
			(default: 'euclidean')
	Return
		dictionary mapping each query image as key to a (distance, filename) with 
		respect to the test_filename
	"""
	assert distance_metric in ("euclidean", "cosine"), "Not valid distance metric!"

	result = dict()

	test_set_norm = np.linalg.norm(test_set, axis=1)

	for i in range(query_set.shape[0]):

		if distance_metric == "euclidean":
			tmp_distance = np.linalg.norm(query_set[i,:]-test_set, axis=1) 

		elif distance_metric == "cosine":
			tmp_distance = 1 - query_set[i,:].dot(test_set.T)/np.outer(np.linalg.norm(query_set[i,:]),test_set_norm)
			if tmp_distance.shape[0] == 1: 
				tmp_distance.resize(tmp_distance.shape[1],)
				
		data_handler = HandlingDistances(tmp_distance, test_filelist)
		result[get_orig_filename(query_filelist[i])] = data_handler.getDistances()
	return result


def sort_by_distance(distanceDict):
	"""
	Args:
		distanceDict: (dict) mapping query file name to a list of tuples 
			[(dist.value, filename)-1st, (dist.value, filename)-2n, ...]
	Return
		sorted dictionary according to the value
	"""
	sorted_distanceDict = dict() 
	for query in distanceDict:
		sorted_distanceDict[query] = sorted(distanceDict[query], 
											key=lambda x: x.value )
	return sorted_distanceDict


def filtering_distances(distanceDict, k):
	"""
	Notes: to be exectued after "sort_by_distance"
	Args:
		distanceDict: (dict) mapping query file name to a list of tuples 
			[(dist.value, filename)-1st, (dist.value, filename)-2n, ...]
		k: (int), number of item to return
	Return
		sorted dictionary according to the value and k
	"""
	filtered_distanceDict = dict()
	for query in distanceDict:
		tmp_re = []
		tmp_list = distanceDict[query]
		first_k_items = tmp_list[0:k]  # list of tuple
		for item in first_k_items:
			tmp_re.append(item.name)   # since is a DistanceObj (you know)
		filtered_distanceDict[query] = tmp_re

	return filtered_distanceDict


### ---------------------------------------------------------------------------------------
### classes to handle distance computation  
### ---------------------------------------------------------------------------------------

class DistanceObj(object):

	def __init__(self, value, name):
		self.value = value
		self.name = name
	def __repr__(self):
		return repr((self.value, self.name))

class HandlingDistances(object):

	def __init__(self, distances, filenames):
		assert distances.shape[0]==len(filenames) or \
		 distances.shape[1]==len(filenames), "Input dimension mismatch!"
		self.distances = distances
		self.filenames = filenames
		self._processed = []
	def _process(self):
		for i in range(self.distances.shape[0]):
			self._processed.append( DistanceObj(self.distances[i], self.filenames[i]) )
	def getDistances(self):
		self._process()
		return self._processed
