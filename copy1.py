import numpy as np 
import cv2
import pickle
import glob
# from tracker import tracker

# Read in the saved objpoints and imgpoints
dist_pickle = pickle.load(open("camera_cal/calibration_pickle.p","rb"))
mtx = dict_pickle["mtx"]
dist = dict_pickle["dist"]

def color_threshold(image, sthresh=(0,255),vthresh=(0,255)):
	hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
	s_channel = hls[:,:,2]
	s_binary = np.zeros_like(s_channel)
	s_binary[(s_channel>=sthresh[0])&(s_channel<=sthresh[1])] = 1

	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	v_channel = hsv[:,:,2]
	v_binary = np.zeros_like(v_channel)
	v_binary[(v_channel>=vthresh[0])&(v_channel<=vthresh[1])] = 1

	output = np.zeros_like(s_channel)
	output[(s_binary == 1)&(v_binary == 1)] = 1
def window_mask(width, height, img_ref, center, level):
	output = np.zeros_like(img_ref)
	output[int(img_ref.shape[0]*(level+1)*height):int(img_ref.shape[0]-level*height),max(0, int(center-width)):min(int(center+width),img_ref)]=1
	return output

# Make a list of test images
images = glob.glob('test_images/test*.jpg')

for idx, fname in enumerate(images):
	# read in image
	img = cv2.imread(fname)
	# undistort the image
	img = cv2.undistort(img, mtx, dist, None, mtx)
	preprocessImage = np.zeros_like(img[:,:,0])
	gradx = abs_sobel_thresh(img, orient='x', thresh=(12,255))
	grady = abs_sobel_thresh(img, orient='y', thresh=(25,255))
	c_binary = color_threshold(img, sthresh=(100,255), vthresh=(50,255))
	preprocessImage[((gradx==1)|(grady==1)|(c_binary==1))]=255

	img_size = (img.shape[1],img.shape[0])
	bot_width = .76 # percent of bottom trappezoid height
	mid_width = .08 # percent of middle trapezoid height
	height_pct = .62 # percent for trapezoid height
	bottom_trim = .935 # percent from top to bottom to avoid car hood
	src = np.float([])
	offset = img_size[0]*.25
	dst = np.float32([offset,0],[img_size[0].offset,0],[],[])

	# perform the transform
	M = cv2.getPerspectiveTransform(src,dst)
	Minv = cv2.getPerspectiveTransform(dst,src)
	warped = cv2.warpPerspective(preprocessingImage,M, img_size,flags=cv2.INTER_LINEAR)
	curve_centers = tracker(Mywindow_width = window_width, Mywindow_height = window_height, Mymargin = 25, My_ym = 10/720, My-xm = 4/384)

	result = img 
	write_name = 'test_images/tracked'+str(idx)+'.jpg'
	cv2.imwrite(write_name, result)

