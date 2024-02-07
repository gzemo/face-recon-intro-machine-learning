import os
import time
import numpy as np
import tensorflow as tf
from run.compilers import *
from run.augmenters import Augmenter
from basemodels import (
    Facenet,
    Facenet512,
    ArcFace,
    SFace,
)

# --- Basic plain model objects to be loaded and trained ---

class Model():
	
	def __init__(self):
		self.train_data = None
		self.train_labels = None
		self.areLabelSpecified = False # flag to check whether data are (array, label)
		self.dataAvalability = False # flag to check whether data have been stored
		self.isbuilt = False # flag to check whether the model is built or not
		self.model = None # initialized as None
		self.weightsDir = None # initialize weights directory (where model cp are saved)

	def dataIn(self, train_data, train_labels=None):
		"""
		Store data as attributes
		train_labels: (default: None) assuming we are using a dataset object
		Return None
		"""
		self.train_data = train_data
		self.train_labels = train_labels
		try:
			if self.train_labels == None: # if array, will give an error
				pass
		except:
			self.areLabelSpecified = True
		self.dataAvalability = True

		print(f'Current Train Data shape: {self.train_data.shape}')

	def checkDataAvailability(self):
		""" 
		Helper function to check if data have been stored 
		Return None
		"""
		if not self.dataAvalability:
			raise Exception('No data available yet!')

	def isModelBuilt(self):
		"""
		Helper function to check if the model has been defined
		Return None
		"""
		if not self.isbuilt:
			raise Exception('You need to compile the model first!')

	def in1Dvector(self):
		"""
		"Flattening" of the current input images into one dimensional vector 
		Modify self.train_data in order to be considered as input for further 
		shallows Neural Networks.
		"""
		input_shape = self.train_data.shape
		self.train_data = self.train_data.reshape(-1, input_shape[1] * input_shape[2]) / 255.0

	def getInputShape(self):
		"""
		Retutn the model input shape (if it's built first)
		"""
		if not self.isbuilt:
			raise Exception('Model needs to be built first!')
		return tuple(self.model.input.shape[1:3])

	def getOutputShape(self):
		"""
		Return the model output shape (if it's built first)
		"""
		if not self.isbuilt:
			raise Exception('Model needs to be built first!')
		return self.model.output.shape[-1]


	def compile(self, modelCompiler):
		""" 
		Compile the model according to the (optimizer, loss, metrics)
		tensorflow objects required by the method "tf.keras.models.Model.compile"
		
		modelCompiler: instane of Compiler class
		learning_rate_decay: (bool), whether to 

		Return None
		"""
		self.checkDataAvailability()

		if not self.isbuilt:
			self.build() # it must be specified per each model it's own building method
		assert self.isbuilt, 'Building process finished with errors!'
		print(self)

		if not isinstance(modelCompiler, Compiler):
			raise Exception('The modelCompiler is not a valid instance of Compiler Class')
		
		print(modelCompiler)
		self.model.compile(optimizer = modelCompiler.getOptimizer(),
			loss = modelCompiler.getLoss(), metrics = modelCompiler.getMetrics())


	def fit(self, epochs:int, batch_size:int=32, step_training:bool=False, 
		save_checkpoints:bool=False, validation_data=None):
		""" 
		Fit the model with the current self.train_data / self.train_labels
		by saving the newly updated weights into a ckpt file at each epoch
		
		epochs: (int), number of epochs
		batch_size: (int), (default = 32), number of input data to pass in 
			the batch size.
		step_training: (bool), if True allows the fitting process to proceed in 
			dynamically.
		save_checkpoints: (bool), if True, save the model's weights at each epochs.
		validation_data (tuple), (val_data, val_label).

		Return: history of tf.keras.models.Model.fit result
		"""
		self.isModelBuilt()
		callbacks = []

		if step_training:
			callbacks.append(tr.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
				factor=0.2, patience=5, min_lr=0.001))
		
		if save_checkpoints:
			output_filename = f'cp_{self.model.name}.ckpt'
			output_filepath = f'checkpoints_{self.model.name}/{output_filename}'
			if not os.path.exists(os.path.dirname(output_filepath)):
				os.system(f'mkdir {os.path.dirname(output_filepath)}')
			# set the weights directory in which to store checkpoints file in 
			self.weightsDir = output_filepath
			callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=output_filepath,
				save_weights_only=True, verbose=1))

		if callbacks == []:
			callbacks == None

		# if your dataset requires to manually specify a set of labels:
		if self.areLabelSpecified:
			history =self.model.fit(self.train_data, self.train_labels, 
				epochs = epochs, batch_size = batch_size,
				callbacks = callbacks, validation_data = validation_data) 
		else:
			history =self.model.fit(self.train_data, 
				epochs = epochs, batch_size = batch_size,
				callbacks = callbacks, validation_data = validation_data) 
		return history


	def load_weights(self, checkpointsDir:str=None):
		"""
		If model has been trained yet and checkpoints have been saved so far, 
		load the model's weights from its directory.

		checkpointsDir: (str), (default: None) provide the folder from where you 
			can load the model's weights (model's architecture must match)
		Return None
		"""
		# case if a checkpointsDir exists:
		if checkpointsDir:
			try:
				print(f'Loading weights from:  {checkpointsDir}')
				self.model.load_weights(checkpointsDir)
				self.weightsDir = checkpointsDir
			except:
				pass
			finally:
				print('Done.')
				return 
		# case if the model has been trained yet and still loaded in workspace 
		if self.weightsDir:
			print(f'Loading weights from:  {self.weightsDir}')
			self.model.load_weights(self.weightsDir)
			print('Done.')
		else:
			print('No checkpoints saved for the current model.')


	def evaluate(self, test_data, test_labels):
		"""
		Evaluate model performance
		"""
		self.model.evaluate(test_data, test_labels, verbose='auto')
		predictions = self.model.predict(test_data)
		return predictions

	def getName(self):
		"""
		Return the name of the model
		"""
		return self.model.name

	def predict(self, test_data):
		"""
		Run the default predict method
		"""
		return self.model.predict(test_data)

	def __str__(self):
		if bool(self.model):
			print(self.model.summary())
		else:
			return 'Model structure not implemented yet'
		return ''

	def build(self):
		""" 
		With this method each subclass is allowed to generates its own methods 
		and build the model architecture
		"""
		pass



class Plain(Model):
	def __init__(self, structure:tuple):
		""" structure must be a tuple of:
			(N of units: int,
			activation function: str, 
			dropout: float 0:1)
		"""
		super().__init__()
		self.structure = structure
		for layer in self.structure:
			if not len(layer) == 3:
				raise Exception('Check the model structure!')
			if not type(layer[0])==int or not type(layer[1])==str or \
				not type(layer[1])==float:
				raise Exception('Structure type is wrong!')

	def build(self):
		""" 
		Build the current Class' model according to the "structure obj"
		"""
		self.checkDataAvailability()
		self.in1Dvector()
		inputLayer = tf.keras.Input(shape = (self.train_data.shape[1],))
		x = tf.keras.layers.Dense(self.structure[0][0], 
			activation = self.structure[0][1])(inputLayer)
		if self.structure[0][2]:
			x = tf.keras.layers.Dropout(self.structure[0][2])(x)
		for i in range(1, len(self.structure)):
			x = tf.keras.layers.Dense(self.structure[i][0], 
			activation = self.structure[i][1])(x)
			if self.structure[i][2]:
				x = tf.keras.layers.Dropout(self.structure[i][2])(x)

		self.model = tf.keras.models.Model(inputs=inputLayer, outputs=x)
		self.isbuilt = True



class SuperPlain(Model):

	def __init__(self):
		super().__init__()
		
	def build(self):
		"""
		Assuming data are passed to the function as already flattened data:
		(#ofimages, dimension)
		"""
		self.checkDataAvailability()
		self.in1Dvector()
		inputLayer = tf.keras.Input(shape = (self.train_data.shape[1],))
		x = tf.keras.layers.Dense(512, activation='relu')(inputLayer)
		x = tf.keras.layers.Dropout(0.3)(x)
		x = tf.keras.layers.Dense(512, activation='relu')(x)
		x = tf.keras.layers.Dropout(0.3)(x)
		x = tf.keras.layers.Dense(512, activation='relu')(x)
		x = tf.keras.layers.Dropout(0.3)(x)
		x = tf.keras.layers.Dense(512, activation='relu')(x)
		x = tf.keras.layers.Dropout(0.3)(x)
		output = tf.keras.layers.Dense(10, activation='softmax')(x)
		self.model = tf.keras.models.Model(inputs=inputLayer, outputs=output, name='superPlain')
		self.isbuilt = True

	
class LeNet(Model):

	def __init__(self):
		super().__init__()

	def build(self):
		"""
		Description
		"""
		# case if images are provided as (W, H, #channels) dimension
		# check if the #channels dimension exists

		self.checkDataAvailability()
		input_shape = self.train_data.shape[1:]
		if len(input_shape) < 3:
			input_shape = input_shape + (1,) # case if does not exist, B/W
		print(input_shape)
		# build model
		inputLayer = tf.keras.Input(shape = (input_shape))
		x = tf.keras.layers.Rescaling(1./255)(inputLayer)
		x = tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu')(x)
		x = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
		x = tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='relu')(x)
		x = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
		x = tf.keras.layers.Flatten()(x)
		x = tf.keras.layers.Dense(84, activation='relu')(x)
		output = tf.keras.layers.Dense(10, activation='softmax')(x)
		self.model = tf.keras.models.Model(inputs=inputLayer, outputs=output, name='LeNet')
		self.isbuilt = True


class LeNetBadAss(Model):

	def __init__(self):
		super().__init__()

	def build(self):
		"""
		Description
		"""
		# case if images are provided as (W, H, #channels) dimension
		# check if the #channels dimension exists

		self.checkDataAvailability()
		input_shape = self.train_data.shape[1:]
		if len(input_shape) < 3:
			input_shape = input_shape + (1,) # case if does not exist, B/W
		print(input_shape)
		# build model
		inputLayer = tf.keras.Input(shape = (input_shape))
		x = tf.keras.layers.Rescaling(1./255)(inputLayer)
		x = tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='relu')(x)
		x = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
		x = tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='relu')(x)
		x = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
		x = tf.keras.layers.Flatten()(x)
		x = tf.keras.layers.Dense(512, activation='relu')(x)
		x = tf.keras.layers.Dropout(0.2)(x)
		x = tf.keras.layers.Dense(512, activation='relu')(x)
		x = tf.keras.layers.Dropout(0.2)(x)
		x = tf.keras.layers.Dense(512, activation='relu')(x)
		x = tf.keras.layers.Dropout(0.2)(x)
		output = tf.keras.layers.Dense(10, activation='softmax')(x)
		self.model = tf.keras.models.Model(inputs=inputLayer, outputs=output, name='LeNetBadAss')
		self.isbuilt = True


class LeNetBadAss2(Model):

	def __init__(self):
		super().__init__()

	def build(self):
		"""
		Description
		"""
		# case if images are provided as (W, H, #channels) dimension
		# check if the #channels dimension exists

		self.checkDataAvailability()
		input_shape = self.train_data.shape[1:]
		if len(input_shape) < 3:
			input_shape = input_shape + (1,) # case if does not exist, B/W
		print(input_shape)
		# build model
		inputLayer = tf.keras.Input(shape = (input_shape))
		x = tf.keras.layers.Rescaling(1./255)(inputLayer)
		x = tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu')(x)
		x = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
		x = tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation='relu')(x)
		x = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
		x = tf.keras.layers.Flatten()(x)
		x = tf.keras.layers.Dense(1024, activation='relu')(x)
		x = tf.keras.layers.Dropout(0.3)(x)
		x = tf.keras.layers.Dense(512, activation='relu')(x)
		x = tf.keras.layers.Dropout(0.3)(x)
		output = tf.keras.layers.Dense(10, activation='softmax')(x)
		self.model = tf.keras.models.Model(inputs=inputLayer, outputs=output, name='LeNetBadAss2')
		self.isbuilt = True


class LeNetBadAss2MP(Model):

	def __init__(self):
		super().__init__()

	def build(self):
		"""
		Description
		"""
		# case if images are provided as (W, H, #channels) dimension
		# check if the #channels dimension exists

		self.checkDataAvailability()
		input_shape = self.train_data.shape[1:]
		if len(input_shape) < 3:
			input_shape = input_shape + (1,) # case if does not exist, B/W
		print(input_shape)
		# build model
		inputLayer = tf.keras.Input(shape = (input_shape))
		x = tf.keras.layers.Rescaling(1./255)(inputLayer)
		x = tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu')(x)
		x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=None)(x)
		x = tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation='relu')(x)
		x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=None)(x)
		x = tf.keras.layers.Flatten()(x)
		x = tf.keras.layers.Dense(1024, activation='relu')(x)
		x = tf.keras.layers.Dropout(0.3)(x)
		x = tf.keras.layers.Dense(512, activation='relu')(x)
		x = tf.keras.layers.Dropout(0.3)(x)
		output = tf.keras.layers.Dense(10, activation='softmax')(x)
		self.model = tf.keras.models.Model(inputs=inputLayer, outputs=output, name='LeNetBadAss2MP')
		self.isbuilt = True



class FaceNet(Model):

	def __init__(self):
		super().__init__()

	def build(self):
		print('Loading pre-trained Facenet from ./facenet/ ...')
		self.model = tf.keras.models.load_model('./facenet/')
		if not self.model.compiled_loss:
			print('Check it out: you need to pass a Compiler!')
		self.model_name = 'FaceNetIncResNet50'
		self.isbuilt = True



class DeepFaceWrapper(Model):
	""" this class will specifically handle some models implemented by deepFace """

	def __init__(self, model_name:str):
		""" 
		model_name:  allowed:
			"Facenet", "Facenet512", "ArcFace",
        	"""
		super().__init__()
		self.allowed = set(("Facenet", "Facenet512", "ArcFace"))
		self.model_name = model_name
		if not self.model_name in self.allowed:
			raise Exception('Not valide model name to load!')

	def build(self):
		print(f'*** Loading pre-trained {self.model_name} from ./deepface/ ...')
		all_models = {
			"Facenet": Facenet,
			"Facenet512": Facenet512,
			"ArcFace": ArcFace,
			#"SFace": SFace,
		}

		self.model = all_models[self.model_name]
		if self.model_name == 'SFace':
			self.model = self.model.load_model()
		else:
			self.model = self.model.loadModel()
		self.isbuilt = True


###-----------------------------------------------------------------------------------------------
### Deprecated 
### No further mantainance 
###-----------------------------------------------------------------------------------------------


class SFaceLite(Model):

	def __init__(self):
		
		raise NotImplementedError
	
		super().__init__()
		self.interpreter = tf.lite.Interpreter(model_path = "./SFaceh5/face_recognition_sface_2021dec_float32.tflite")
		self.model_name = 'SFace'

	def build(self):
		self.interpreter.allocate_tensors()
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()
		self.isbuilt = True

	def getInputShape(self):
		"""
		Retutn the model input shape
		"""
		return tuple(self.input_details[0]['shape'])[1:3]

	def getOutputShape(self):
		"""
		Return the model output shape 
		"""
		return int(self.output_details[0]['shape'][-1])

	def _invokeSingle(self, img):
		"""
		Run the default predict method
		!!! Batch processing is not implemented !!!
		"""
		self.interpreter.set_tensor(self.input_details[0]['index'], img)
		self.interpreter.invoke()
		return self.interpreter.get_tensor(self.output_details[0]['index'])

	def predict(self, test_data):
		"""
		Args:
			est_data: (np.array/tensor) like (Nitems, ImgSize, ImgSize, Nchannels)
		Retrun:
			matrix of embeddings
		"""
		assert len(test_data.shape)==4, "Test dataset like (Nitems, ImgSize, ImgSize, Nchannels) is required!"
		
		print('Now running SFace ...')
		datasetSize = test_data.shape[0]
		re = []
		s = time.time()
		for i in range(datasetSize):
			re.append(self._invokeSingle(tf.expand_dims(test_data[i,:,:,:], axis=0)))
		e = time.time()
		print(f'Time elapsed: {round(e-s, 4)} sec')
		re = np.array(re)
		re.resize(re.shape[0], re.shape[2])
		return re

