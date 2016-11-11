# import the necessary packages
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib
import json
import cv2
import os
import sys

from IPython.display import clear_output # to clear command output when this notebook gets too cluttered

home_dir = os.getenv("HOME")
caffe_root = os.path.join(home_dir, 'caffe')  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, os.path.join(caffe_root, 'python'))

import caffe
# define the path to the face detector
FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
	base_path=os.path.abspath(os.path.dirname(__file__)))

@csrf_exempt
def detect(request):
	data = {"success": False}

	if request.method == "POST":
		if request.FILES.get("image", None) is not None:
			image = _grab_image(stream=request.FILES["image"])

		else:
			urls = request.POST.getlist("url")
			if urls is None:
				data["error"] = "No URL provided."
				return JsonResponse(data)

		faces = []
		for url in urls:
			image = _grab_image(url=url)

			face_cascade = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			rects = face_cascade.detectMultiScale(image, 1.3, 5)
			rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]

			data = dict()
			data["num_faces"]=len(rects)
			data["faces"]=rects
			data["success"]=True
			probability_response = _grab_probability(url)
			data['probabilities'] = probability_response

			faces.append(data)
			print 'x'
			print faces

		print urls

	return JsonResponse(faces, safe=False)

def _grab_image(path=None, stream=None, url=None):
	if path is not None:
		image = cv2.imread(path)

	else:
		if url is not None:
			resp = urllib.urlopen(url)
			data = resp.read()

		elif stream is not None:
			data = stream.read()

		arr = np.asarray(bytearray(data), dtype="uint8")
		image = cv2.imdecode(arr,-1)

	return image

def _grab_caffe_image(url=None):
	resp = urllib.urlopen(url)
	return resp


def _grab_probability(url=None):
	#caffe
	if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
		print 'CaffeNet found.'
	else:
		print 'Downloading pre-trained CaffeNet model...'
		#!~/caffe/scripts/download_model_binary.py ~/caffe/models/bvlc_reference_caffenet

	caffe.set_mode_cpu()

	model_def = os.path.join(caffe_root, 'models', 'bvlc_reference_caffenet','deploy.prototxt')
	model_weights = os.path.join(caffe_root, 'models','bvlc_reference_caffenet','bvlc_reference_caffenet.caffemodel')

	net = caffe.Net(model_def,      # defines the structure of the model
	            model_weights,  # contains the trained weights
	            caffe.TEST)     # use test mode (e.g., don't perform dropout)

	mu = np.load(os.path.join(caffe_root, 'python','caffe','imagenet','ilsvrc_2012_mean.npy'))
	mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

	# create transformer for the input called 'data'
	transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

	transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
	transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
	transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
	transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

	# set the size of the input (we can skip this if we're happy
	#  with the default; we can also change it later, e.g., for different batch sizes)
	net.blobs['data'].reshape(50,        # batch size
	                          3,         # 3-channel (BGR) images
	                          227, 227)  # image size is 227x227

	image_caffe = caffe.io.load_image(_grab_caffe_image(url=url))
	transformed_image = transformer.preprocess('data', image_caffe)

	# copy the image data into the memory allocated for the net
	net.blobs['data'].data[...] = transformed_image
	### perform classification
	output = net.forward()

	output_prob = output['prob'][0]  # the output probability vector for the first image in the batch
		# load ImageNet labels
	labels_file = os.path.join(caffe_root, 'data','ilsvrc12','synset_words.txt')
	#if not os.path.exists(labels_file):
	    #!~/caffe/data/ilsvrc12/get_ilsvrc_aux.sh

	labels = np.loadtxt(labels_file, str, delimiter='\t')

	probability_response = labels[output_prob.argmax()]

	return probability_response
