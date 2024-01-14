#!/usr/bin/env python
import os
import json
import requests
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from models.models import DeepFaceWrapper
from run.utils import *
from cropper.common import Yolov8Face

###------------------------------------------------------------------------------------
### Adapt according to your path

### Face detection. Allowed: (Yolov8)
engine = 'Yolov8'

### Face embedding extraction. Allowed: (Facenet / Facenet512 / ArcFace)
model_name = 'Facenet'

### file path of the query, test images
to_crop_query     = ''  # filepath to query images
to_crop_test      = ''  # filepath to gallery images
output_crop_query = ''	# output folder for cropped query images
output_crop_test  = ''	# output folder for cropped gallery images
###------------------------------------------------------------------------------------


def runSFace(output_crop_query, output_crop_test, versions):
	s = time.time()
	print('Now Testing: SFace !')
	model = SFaceLite()
	model.build()
	modelInputShape = model.getInputShape()
	modelOutputShape = model.getOutputShape()
	print(f'Input shape:  {modelInputShape}')
	print(f'Output shape: {modelOutputShape}')

	queryfolder = output_crop_query+'/images'
	testfolder = output_crop_test+'/images'

	for version in versions:
		print(version, '*****')
		normalization_option = f"SFace_{version}"
		output_txt_filename = f"results_{model.model_name}_norm_{normalization_option}_cosine_sub.txt"
		
		print(f'Normalization option: {normalization_option} ...')
		print(f'Loading datasets from folder: {queryfolder} ...')
		query_dataset, query_orignames = generateDataset(folder=queryfolder,
														modelInputShape=modelInputShape,
														normalizationOpt=normalization_option)
		
		print(f'Loading datasets from folder: {testfolder} ...')
		test_dataset, test_orignames = generateDataset(folder=testfolder,
														modelInputShape=modelInputShape,
														normalizationOpt=normalization_option)

		print('Extract embedding features: ...')
		query_predictions = model.predict(query_dataset)
		test_predictions  = model.predict(test_dataset)

		distMetric = "cosine"
		print(f'Distance estimation: {distMetric} ...')
		re = compute_distance(query_set    = query_predictions,
							query_filelist = query_orignames,
							test_set       = test_predictions,
							test_filelist  = test_orignames,
							distance_metric= distMetric)

		print('Now sorting the resulting embeddings: ...')
		sorted_re = sort_by_distance(re)

		print('Filtering top k results: ...')
		filtered_re = filtering_distances(sorted_re, k = 10)

		with open(output_txt_filename, 'w') as f:
			for k in filtered_re:
				f.write(f'{str(k)}:{str(filtered_re[k])}\n')

		e = time.time()
		print(f'Time elapsed: {round((e-s)/60, 2)} min.')
	
		### submission !!!! ####
		mydata = dict()
		mydata["groupname"] = "my-group-name"
		mydata["images"] = filtered_re
		#submit(mydata)

	return filtered_re

		
def runModels(modelName, output_crop_query, output_crop_test):

	s = time.time()
	print(f'Now Testing: {modelName} !')

	model = DeepFaceWrapper(modelName)  
	model.build()

	modelInputShape = model.getInputShape()
	modelOutputShape = model.getOutputShape()
	print(f'Input shape:  {modelInputShape}')
	print(f'Output shape: {modelOutputShape}')

	queryfolder = output_crop_query +'/images'
	testfolder = output_crop_test+'/images'


	if modelName in ("Facenet", "Facenet512"):
		normalization_option = "Facenet"
		distMetrics = ["cosine", "euclidean"]

	elif modelName == "ArcFace":
		normalization_option = "ArcFace"
		distMetrics = ["cosine"]

	elif modelName == "VGG-Face":
		normalization_option = "VGGFace"
		distMetrics = ["cosine"]

	elif modelName == "DeepFace":
		normalization_option = "VGGFace2"
		distMetrics = ["cosine", "euclidean"]

	elif modelName == "OpenFace":
		normalization_option = "SFace_v1"
		distMetrics = ["cosine", "euclidean"]


	print(f'Normalization: {normalization_option} ...')
	print(f'Evaluating distance: {distMetrics}')
	print(f'Loading datasets from folder: {queryfolder} ...')
	query_dataset, query_orignames = generateDataset(folder=queryfolder,
													modelInputShape=modelInputShape,
													normalizationOpt=normalization_option)
	
	print(f'Loading datasets from folder: {testfolder} ...')	
	test_dataset, test_orignames = generateDataset(folder=testfolder,
													modelInputShape=modelInputShape,
													normalizationOpt=normalization_option)

	
	print('Extract embedding features: ...')
	query_predictions = model.model.predict(query_dataset, batch_size=4)
	test_predictions  = model.model.predict(test_dataset, batch_size=4)


	# hardcoding testing all images vs all images in the test set
	for distMetric in distMetrics:
		print(f'Distance estimation: {distMetric} ...')
		output_txt_filename = f'results_{model.model_name}_{distMetric}_sub.txt'

		re = compute_distance(query_set    = query_predictions,
							query_filelist = query_orignames,
							test_set       = test_predictions,
							test_filelist  = test_orignames,
							distance_metric= distMetric)

		print('Now sorting the resulting embeddings: ...')
		sorted_re = sort_by_distance(re)

		print('Filtering top k results: ...')
		filtered_re = filtering_distances(sorted_re, k = 10)

		with open(output_txt_filename, 'w') as f:
			for k in filtered_re:
				f.write(f'{str(k)}:{str(filtered_re[k])}\n')

		### submission !!!! ####
		mydata = dict()
		mydata["groupname"] = "my_group_name"
		mydata["images"] = filtered_re
		#submit(mydata)

	e = time.time()
	print(f'Time elapsed: {round((e-s)/60, 2)} min.')
	return filtered_re


def cropAll(engine, to_crop_query, to_crop_test, output_crop_query, output_crop_test):
		
		s = time.time()

		print('Images are going to be stored in the /images/ of:\n',output_crop_query,'\n',output_crop_test)
		
		if engine == 'Yolov8':
			yaml_file = 'yolov8n_face.yaml'
			pt_file   = 'yolov8n-face.pt'
			cropper = Yolov8Face(yaml_file, pt_file)
			cropper.cropFolder(to_crop_query, output_crop_query, confidence=0.10)
			cropper.cropFolder(to_crop_test, output_crop_test, confidence=0.10)

		else:
			raise Exception("The model has not been implemented (yet)!")

		e = time.time()
		print(f'Time elapsed: {round((e-s)/60, 2)} min.')


def submit(results):
	url=" " 
	res = json.dumps(results)
	response = requests.post(url, res)
	try:
		result = json.loads(response.text)
		print(f"accuracy is {result['results']}")
		return result
	except json.JSONDecodeError:
		print(f"ERROR: {response.text}")
		return None




if __name__=='__main__':

	### folder to store weights
	if not os.path.exists(os.path.join(os.path.expanduser('~'),'.deepface')):
		print('Creating: ~/.deepface/weights  in which models weights will be stored')
		os.mkdir(os.path.join(os.path.expanduser('~')),'.deepface')
		os.mkdir(os.path.join(os.path.expanduser('~')),'.deepface','weights')

	### run cropping
	cropAll(engine, to_crop_query, to_crop_test, output_crop_query, output_crop_test)

	### Extract embeddings and compute similarities:
	### Allowed: face Wrapper allowed pretrained models: Facenet / Facenet512 / ArcFace
	tmp = runModels(model_name, output_crop_query, output_crop_test)

	### model wrapper for SFace
	tmp = runSFace(output_crop_query, output_crop_test, versions=['v0','v1','v2'])

