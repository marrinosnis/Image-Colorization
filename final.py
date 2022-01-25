from skimage.io import imread, imsave, imshow, show
from skimage.color import rgb2lab, gray2rgb, rgb2gray, lab2rgb
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float

from sklearn import svm
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from sklearn.cluster import KMeans

from time import time
from matplotlib.pyplot import ion

import matplotlib.pyplot as plt
import numpy as np
import cv2


def show_superpixels(img, img_slic, title):
	target2 = mark_boundaries(img_as_float(img), img_slic)
	return target2

def show_keypoints(img, img_kp):
	keyPoint = cv2.drawKeypoints(img, img_kp, None)
	return keyPoint

def show_kp_and_superpixels(img, img_slic, img_kp, ):
	reference2 = mark_boundaries(img_as_float(cv2.drawKeypoints(img, img_kp, None)), img_slic)
	return reference2

def generate_gkernels():
	g_kernels = []

	for theta in range(8):
		theta = theta * np.pi / 8.

		g_kernels.append(cv2.getGaborKernel(kernel_size, 8.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F))

	return np.array(g_kernels)

def get_gabor_bank(gabor_kernels, img):
	filtered_img = []

	for k in gabor_kernels:
		filtered_img.append(cv2.filter2D(img, cv2.CV_8UC3, k))

	return np.array(filtered_img)

def discretization(image, n_colors, n_samples):
	# no need to devide with 255, input image is in lab color space
	image = np.array(image, dtype=np.float64)

	w, h, d = original_shape = tuple(image.shape)
	assert d == 3
	image_array = np.reshape(image, (w * h, d))

	print('Fitting model on small sub-sample of the data')
	t0 = time()
	image_array_sample = shuffle(image_array, random_state=0)[:n_samples, 1:]
	kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
	print('Done in %0.3fs.' % (time() - t0))
	return kmeans


# Get the average SURF & Gabor features for each superpixel
# Returns an array with the following format:
# [(avg SURF, avg Gabor), ...]
def get_avg_feature_per_superpixel(img_slic, kp, descriptor, gabor_bank):
	# From keypoint get the location (x,y)
	pts = [kp[idx].pt for idx in range(0, len(kp))]

	avg = {}
	global_avg_SURF = []

	# For each superpixel create an empty list
	for (_, i) in enumerate(np.unique(img_slic)):
		avg[i] = []

	# Add each keypoint to the corresponding superpixel
	for i, (x, y) in enumerate(pts):
		x_int = round(x)
		y_int = round(y)

		superpixel_label = img_slic[y_int, x_int]

		avg[superpixel_label].append(descriptor[i])
		global_avg_SURF.append(descriptor[i])

	global_avg_SURF = np.mean(global_avg_SURF, 0)
	avg_list = []

	for i in sorted(avg):

		avg_gabor = []

		# construct a mask for the segment
		if gabor_on:
			mask = np.zeros(img_slic.shape[:2], dtype="uint8")
			mask[img_slic == i] = 255

			for j in range(len(gabor_bank)):
				merged = cv2.bitwise_and(gabor_bank[j], gabor_bank[j], mask=mask)
				avg_gabor.append(sum(sum(merged)) / np.count_nonzero(mask))

		if avg[i] == []:
			avg_list.append(np.concatenate((global_avg_SURF, np.array(avg_gabor))))
		else:
			avg_list.append(np.concatenate((np.mean(avg[i], 0), avg_gabor)))

	return avg_list


# Get the color bucket per superpixel
def get_color_per_superpixel(kmeans, segments, image):

	avg = {}

	# loop over the unique segment values
	for (i, segVal) in enumerate(np.unique(segments)):
		# construct a mask for the segment
		mask = np.zeros(image.shape[:2], dtype="uint8")
		mask[segments == segVal] = 255

		merged = cv2.bitwise_and(image, image, mask=mask)
		avg[segVal] = sum(sum(merged)) / np.count_nonzero(mask)

	avg_list = []
	for i in sorted(avg):
		avg_list.append(avg[i][1:])

	return kmeans.predict(avg_list)


def colorize_superpixels(img, img_slic, colors):

	ret_img = img.copy()
	for i, c in enumerate(colors):
		ret_img[img_slic == i, 1:] = c

	return ret_img

# Merge all gabor 2d arrays into single array
def merge_gabor_bank(bank):

	merged = bank[0]
	for i in range(len(bank) - 1):
		merged = merged + bank[i+1]

	return merged

# === PARAMETERS ===
# folder name with / suffix
folder='tree/'
segs = 10
comp = 700
sigma = 3
n_colors = 10
n_samples = 1000
gabor_on = False
# === END PARAMETERS ===


target = imread('images/'+folder+'gray.jpg')
reference = imread('images/'+folder+'rgb.jpg')
target = gray2rgb(target) # just to make the target 3d (x,y,3)

target_2d = rgb2gray(target)
reference_2d = rgb2gray(reference)

# Convert to CIELAB (i)
target_lab = rgb2lab(target)
reference_lab = rgb2lab(reference)

# Discretization of the CIELAB color space (ii)
kmeans = discretization(reference_lab, n_colors, n_samples)
plt.scatter(reference_lab[1], reference_lab[2])


# SLIC (iii)
target_slic = slic(target_lab, n_segments=segs, compactness=comp, sigma=sigma)
reference_slic = slic(reference_lab, n_segments=segs, compactness=comp, sigma=sigma)


# Exract SURF & Gabor textures (iv)
surf = cv2.xfeatures2d.SURF_create(extended=False)
target_kp, target_desc = surf.detectAndCompute(target, None)
reference_kp, reference_desc = surf.detectAndCompute(reference, None)

if gabor_on:
	gk = generate_gkernels()
	ref_gabor = get_gabor_bank(gk, reference_2d)
	target_gabor = get_gabor_bank(gk, target_2d)
else:
	ref_gabor = []
	target_gabor = []


# Training SVM (v)
X = get_avg_feature_per_superpixel(reference_slic, reference_kp, reference_desc, ref_gabor)
y = get_color_per_superpixel(kmeans, reference_slic, reference_lab)
clf = svm.SVC()
clf.fit(X, y)


# Colorize the target image
X = get_avg_feature_per_superpixel(target_slic, target_kp, target_desc, target_gabor)
predicted_colors = clf.predict(X)
img = colorize_superpixels(target_lab, target_slic, kmeans.cluster_centers_[predicted_colors])


fig = plt.figure()

if gabor_on:
	base = 320
else:
	base = 220

fig.add_subplot(base + 1)
plt.title("Target")
plt.imshow(target)
plt.axis("off")

fig.add_subplot(base + 2)
plt.title("Reference")
plt.imshow(reference)
plt.axis("off")

target2_show = show_kp_and_superpixels(target, target_slic, target_kp)
fig.add_subplot(base + 3)
plt.title("Target superpixels & SURF")
plt.imshow(target2_show)
plt.axis("off")

reference2_show = show_kp_and_superpixels(reference, reference_slic, reference_kp)
fig.add_subplot(base + 4)
plt.title("Reference superpixels & SURF")
plt.imshow(reference2_show)
plt.axis("off")

if gabor_on:
	target_gabor_all = merge_gabor_bank(target_gabor)
	fig.add_subplot(base + 5)
	plt.title("Target Gabor")
	plt.imshow(target_gabor_all)
	plt.axis("off")

	ref_gabor_all = merge_gabor_bank(ref_gabor)
	fig.add_subplot(base + 6)
	plt.title("Reference Gabor")
	plt.imshow(ref_gabor_all)
	plt.axis("off")

fig = plt.figure()
fig.add_subplot(121)
plt.title("Original")
plt.imshow(target)
plt.axis("off")

fig.add_subplot(122)
plt.title("Final")
plt.imshow(lab2rgb(img))
plt.axis("off")

if gabor_on:
	fig=plt.figure()
	for (i, kernel) in enumerate(gk):
		fig.add_subplot(2, 4, i+1)
		plt.imshow(kernel, cmap=plt.cm.gray)

imsave('images/'+folder+'final.jpg', lab2rgb(img))

# ion()
plt.show()

plt.pause(1)
input("<Hit Enter To Close>")
plt.close('all')
