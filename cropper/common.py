#!/usr/bin/env python
import os
import cv2
import time
import PIL
from termcolor import colored
import numpy as np
import tensorflow as tf
from ultralytics import YOLO


class Yolov8Face(object):

	def __init__(self, yaml_file, pt_file):
		self.yaml_file = yaml_file
		self.pt_file = pt_file
		self.iscroppingdone = False
		self.acceptedFormat = set(('jpg','png','jpeg','png','bmp','gif'))

		# initialize yolo model
		self.model = YOLO(self.yaml_file)
		self.model = YOLO(self.pt_file)


	def _checkFormat(self, filename):
		extension = filename.split('.')[-1]
		return extension in self.acceptedFormat


	def _initialCheck(self, folder):
		filelist = os.listdir(folder)
		extension = []
		for item in filelist:
			if not item.split('.')[-1] in extension:
				extension.append(item.split('.')[-1])
		return len(set(extension).intersection(self.acceptedFormat)) == len(set(extension))
		

	def _cropSingle(self, filename, confidence, device='cpu'):
		"""
		Args:
			filename: (str), name of the image
			output: (str), where to 
			confidence: (float), [0,1] threshold for object detection (default=0.20)
		Return: Ultralitics YOLO boxes class
		"""
		results = self.model(source=filename, conf=confidence, iou=0.70, device=device, 
			visualize=False, save_crop=False, box=False, verbose=True)
		return results[0]


	def _saveCroppedImage(self, bbobj, filename, output, verbose=True):
		""" Save cropped images according to bounding boxes
		Args:
			bbobj: bounding box object from Ultralytics yoloModel
			filename: (str) assuming just a single filename (pics.jpg)
				or other format (i.e. absolute path) (it can handle both)
			yaml_file: (str), path to model's yaml
			pt_file: (str), path to model's .pt file
		Return: None
		"""
		if not os.path.isdir(output):
			os.mkdir(os.path.join(output))
			os.mkdir(os.path.join(output,'images'))

		# checking for image name format (either absolute or not)
		if len(filename.split('/')) > 1:
			splitted = str(filename.split('/')[-1])
		else:
			splitted = filename
		outName = str(splitted.split('.')[0])

		img = np.array(PIL.Image.open(filename).convert('RGB'))
		someBoxes = False

		for i, bb in enumerate(bbobj.boxes.xyxy):
			# cropping
			x1,y1,x2,y2 = np.int64(bb)
			cropped = img[y1:y2, x1:x2, :]
			# saving
			im = PIL.Image.fromarray(cropped)
			tmpOutName = f'{outName}_crop_{i}.jpg'
			if verbose:
				print(colored(f'Now saving: {tmpOutName}', 'green'))

			im.save(os.path.join(output,'images',tmpOutName))
			someBoxes = True

		if not someBoxes:
			tmpOutName = f'{outName}_crop_0.jpg'
			im = PIL.Image.fromarray(img)
			im.save(os.path.join(output,'images',tmpOutName))

			if verbose:
				print(colored(f'  *** Face Not Found !!! *** saving  {tmpOutName} as it is.', 'red'))


	def cropFolder(self, folder, output, confidence):
		""" Run: cropping every image found on a given folder """

		if not self._initialCheck(folder):
			print('*** (WARNING) Some not valid image file format found!')
		else:
			print(colored('Initial check done!', 'green'))

		time.sleep(1)

		for item in os.listdir(folder):
			filename = os.path.join(folder, item)
			try:
				bbobj = self._cropSingle(filename, confidence=confidence)
				self._saveCroppedImage(bbobj, filename, output)
			except:
				print(f'*** File: {filename} is not a valid image format!')
		self.iscroppingdone = True
		print('\n')