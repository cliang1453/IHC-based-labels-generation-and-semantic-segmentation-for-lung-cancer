from glob import glob
import os
import sys
import time
import pprint
import argparse
import glob
from datetime import datetime
from PIL import Image, ImageDraw
import cv2
import itertools

import matplotlib.pyplot as plt  
import matplotlib as mpl  
import numpy as np  
import openslide  
from openslide import open_slide  
from openslide.deepzoom import DeepZoomGenerator  

from scipy.ndimage.morphology import binary_fill_holes, binary_closing, binary_dilation  
from skimage.color import rgb2gray  
from skimage.morphology import closing, binary_closing, disk, remove_small_holes, dilation, remove_small_objects  
from skimage import color, morphology, filters, exposure, feature
import xml.etree.ElementTree as ET

# default values of parameters 
IM_DIR = '/home/chen/Downloads/Eric/complete_model/svs/im/'
STAINED_DIR = '/home/chen/Downloads/Eric/complete_model/svs/stained/'
IM_SAVE_DIR = '/home/chen/Downloads/Eric/complete_model/im/'
STAINED_SAVE_DIR = '/home/chen/Downloads/Eric/complete_model/stained/'
BLEND_SAVE_DIR = '/home/chen/Downloads/Eric/complete_model/svs/blend/'
LEVEL_1_CROP = 3000
LEVEL_1_DOWNSAMPLE = 6
LEVEL_2_CROP = 1200
LEVEL_2_DOWNSAMPLE = 12

# fixed global variables 
LEVEL_3_CROP = 600
MIN_MATCH_COUNT = 6
FLANN_INDEX_KDTREE = 0
ROI_THRES = 0.8
RESIZE_TO = 256


def parse_annotation(name = None):
	
	tree = ET.parse(name)
	root = tree.getroot()
	region_list = []
	
	for vertices in root.iter('Vertices'):
		region_vertices = []
		for vertex in vertices.iter('Vertex'):
			region_vertices.append((int(vertex.attrib['X']),int(vertex.attrib['Y'])))
		region_list.append(region_vertices)

	return region_list

def generate_mask(original_size = None, region_list = None):

	#region_mask = np.zeros((original_size[1], original_size[0]), np.uint8)
	img = Image.new('L', original_size, 0)
	for i in range(len(region_list)):
		ImageDraw.Draw(img).polygon(region_list[i], outline=1, fill=1)
	
	region_mask = np.array(img)

	return region_mask

def check_roi(w = 0, h = 0, region_mask = None, crop_size = None):
	roi = region_mask[h * crop_size : (h+1) * crop_size,  w * crop_size: (w+1) * crop_size]
	if((np.count_nonzero(roi)/float(crop_size**2)) < ROI_THRES):
		return False
	else:
		print((np.count_nonzero(roi)/float(crop_size**2)))
		return True


def _get_homography(src = None, dest = None, flann_ratio = 0.75):

	# Initiate SIFT detector
	sift = cv2.xfeatures2d.SIFT_create()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = sift.detectAndCompute(src, None)
	kp2, des2 = sift.detectAndCompute(dest, None)

	if des1 is None or des2 is None:
		print('Not enough descriptors found')
		return None
		
	if len(kp1)<2 or len(kp2)<2:
		print "Not enough key points found"
		return None
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks = 50)
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	matches = flann.knnMatch(des1,des2,k=2)
	
	# bf = cv2.BFMatcher()
	# matches = bf.knnMatch(des1, des2, k=2)
	# print(matches)

	# store all the good matches as per Lowe's ratio test.
	good = []
	if len(matches[0])<=1:
		print("Not enough matches are found")
		return None

	for m,n in matches:
	    if m.distance < flann_ratio*n.distance:
	        good.append(m)
	
	if len(good)>=MIN_MATCH_COUNT:

	    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
	    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

	    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
	    if M is None: 
	    	print('Calculated M is none')
	    	return None
	    matchesMask = mask.ravel().tolist()

	    h,w = src.shape
	    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	    dst = cv2.perspectiveTransform(pts,M)
	    src = cv2.polylines(dest,[np.int32(dst)],True,255,3, cv2.LINE_AA)
	else:
	    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
	    return None
	# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
	# 					singlePointColor = None,
	# 					matchesMask = matchesMask, # draw only inliers
	# 					flags = 2)
	# final = cv2.drawMatches(src,kp1,dest,kp2,good,None,**draw_params)

	# plt.imshow(final, 'gray'),plt.show()
	return M
	

def _third_level_manipulation(im_full_size = None, stained_full_size = None, original_size = (args.level_2_crop, args.level_2_crop), name = None, count = None):

	print('maximum crops in level III:' + str(original_size[0]/LEVEL_3_CROP) + 
		'*' + str(original_size[1]/LEVEL_3_CROP) + '=' + str(original_size[0]/LEVEL_3_CROP * original_size[1]/LEVEL_3_CROP))

	for w in range(original_size[0]/LEVEL_3_CROP):
		for h in range(original_size[1]/LEVEL_3_CROP):

			im = im_full_size[h * LEVEL_3_CROP : (h+1) * LEVEL_3_CROP,  w * LEVEL_3_CROP: (w+1) * LEVEL_3_CROP].copy()  
			stained = stained_full_size[h * LEVEL_3_CROP : (h+1) * LEVEL_3_CROP,  w * LEVEL_3_CROP: (w+1) * LEVEL_3_CROP].copy()  
			blend = cv2.addWeighted(im,0.5,stained,0.5,0)
			
			# discard crops with too much white regions (useless information) and black regions (due to homography transformation)
			im = np.array(Image.fromarray(im.astype(np.uint8)).resize((RESIZE_TO, RESIZE_TO)))
			if np.count_nonzero(im[im>230])/float(RESIZE_TO*RESIZE_TO*3) > 0.2:
				continue
			stained = np.array(Image.fromarray(stained.astype(np.uint8)).resize((RESIZE_TO, RESIZE_TO)))
			if np.count_nonzero(stained[stained>230])/float(RESIZE_TO*RESIZE_TO*3) > 0.2 or np.count_nonzero(stained==0)/float(RESIZE_TO*RESIZE_TO*3) > 0.05:
				continue

			count[1] = count[1] + 1
			im_result = Image.fromarray(im.astype(np.uint8))
			stained_result = Image.fromarray(stained.astype(np.uint8))
			blend_result = Image.fromarray(blend.astype(np.uint8))
			im_result.save(os.path.join(args.im_save_dir, name.strip('.svs') + str(count[0]) + '_' + str(count[1]) + '.png'))
			stained_result.save(os.path.join(args.stained_save_dir, name.strip('.svs') + str(count[0]) + '_' + str(count[1])  + '.png'))
			blend_result.save(os.path.join(args.blend_save_dir, name.strip('.svs') + str(count[0]) + '_' + str(count[1]) + '.png'))



def _second_level_manipulation(im_full_size = None, stained_full_size = None, original_size = (args.level_1_crop, args.level_1_crop), name = None, count = None):

	print('maximum crops in level II: ' + str(original_size[0]/args.level_2_crop) + 
		'*' + str(original_size[1]/args.level_2_crop) + '=' + str(original_size[0]/args.level_2_crop * original_size[1]/args.level_2_crop))

	for w in range(original_size[0]/args.level_2_crop):
		for h in range(original_size[1]/args.level_2_crop):
			
			im = im_full_size[h * args.level_2_crop : (h+1) * args.level_2_crop,  w * args.level_2_crop: (w+1) * args.level_2_crop].copy()  
			stained = stained_full_size[h * args.level_2_crop : (h+1) * args.level_2_crop,  w * args.level_2_crop: (w+1) * args.level_2_crop].copy()  

			im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			stained_gray = cv2.cvtColor(stained, cv2.COLOR_BGR2GRAY)

			im_ds = cv2.resize(im_gray, None, fx=1.0/args.level_2_ds, fy=1.0/args.level_2_ds, interpolation = cv2.INTER_LINEAR)
			stained_ds = cv2.resize(stained_gray, None, fx=1.0/args.level_2_ds, fy=1.0/args.level_2_ds, interpolation = cv2.INTER_LINEAR)

			# Get Transformation matrix from stained_ds to im_ds
			M = _get_homography(stained_ds, im_ds, flann_ratio = 0.75)
			if M is None:
				print('1000 * 1000 crop at position' + str(h) + ',' + str(w) + 'encounters problem in level II, discarded')
			
			if M is not None:
				M[0][2] = M[0][2] *args.level_2_ds
				M[1][2] = M[1][2] *args.level_2_ds
				M[2][0] = M[2][0] /args.level_2_ds
				M[2][1] = M[2][1] /args.level_2_ds

				# Apply Transformation matrix to original resolution stained im crop
				r = cv2.warpPerspective(stained[:, :, 0], M, (stained.shape[0], stained.shape[1]))
				g = cv2.warpPerspective(stained[:, :, 1], M, (stained.shape[0], stained.shape[1]))
				b = cv2.warpPerspective(stained[:, :, 2], M, (stained.shape[0], stained.shape[1]))
				stained_transformed = cv2.merge((r, g, b))

				count[1] = count[1] + 1
				_third_level_manipulation(im_full_size = im, stained_full_size = stained_transformed, 
					                      original_size = (args.level_2_crop, args.level_2_crop), name = name, count = count)


def _first_level_manipulation(im_full_size = None, stained_full_size = None, region_mask = None, 
				              original_size = None, name = None, count = None, roi_check = False):


	print('maximum crops in level I: ' + str(original_size[0]/args.level_1_crop) + 
		'*' + str(original_size[1]/args.level_1_crop) + '=' + str(original_size[0]/args.level_1_crop * original_size[1]/args.level_1_crop))
	
	for w in range(original_size[0]/args.level_1_crop):
		for h in range(original_size[1]/args.level_1_crop):
			
			if roi_check is True:
				if(check_roi(w = w, h = h, region_mask = region_mask, crop_size = args.level_1_crop) is False):
					continue

			im = np.array(im_full_size.read_region((args.level_1_crop * w, args.level_1_crop * h), 0, (args.level_1_crop , args.level_1_crop)).convert('RGB'))
			stained = np.array(stained_full_size.read_region((args.level_1_crop * w, args.level_1_crop * h), 0, (args.level_1_crop, args.level_1_crop)).convert('RGB'))

			im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
			stained_gray = cv2.cvtColor(stained, cv2.COLOR_RGB2GRAY)

			im_ds = cv2.resize(im_gray, None, fx=1.0/args.level_1_ds, fy=1.0/args.level_1_ds, interpolation = cv2.INTER_LINEAR)
			stained_ds = cv2.resize(stained_gray, None, fx=1.0/args.level_1_ds, fy=1.0/args.level_1_ds, interpolation = cv2.INTER_LINEAR)

			#Get Transformation matrix from stained_ds to im_ds
			M = _get_homography(stained_ds, im_ds, flann_ratio = 0.75)
			if M is None:
				print(str(args.level_1_crop) + '*' + str(args.level_1_crop) + 'crop at position' + str(h) + ',' + str(w) + 'encounters problem in level I, discarded')
				
			if M is not None:
				M[0][2] = M[0][2] *args.level_1_ds
				M[1][2] = M[1][2] *args.level_1_ds
				M[2][0] = M[2][0] /args.level_1_ds
				M[2][1] = M[2][1] /args.level_1_ds

				# Apply Transformation matrix to original resolution stained im crop
				r = cv2.warpPerspective(stained[:, :, 0], M, (stained.shape[0], stained.shape[1]))
				g = cv2.warpPerspective(stained[:, :, 1], M, (stained.shape[0], stained.shape[1]))
				b = cv2.warpPerspective(stained[:, :, 2], M, (stained.shape[0], stained.shape[1]))
				stained_transformed = cv2.merge((r, g, b))

				count[0] = count[0] + 1

				if args.level_2_crop is not None: 
					_second_level_manipulation(im_full_size = im, stained_full_size = stained_transformed, 
						                       original_size = (args.level_1_crop, args.level_1_crop), name = name, count = count)
				else:
					_third_level_manipulation(im_full_size = im, stained_full_size = stained_transformed, 
						                      original_size = (args.level_1_crop, args.level_1_crop), name = name, count = count)




def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--im_dir", type=str, default=IM_DIR,
                        help="H&E svs slide directory.")
    parser.add_argument("--stained_dir", type=str, default=STAINED_DIR,
                        help="IHC svs slide directory.")
    parser.add_argument("--im_save_dir", type=str, default=IM_SAVE_DIR,
                        help="H&E images save directory.")
    parser.add_argument("--stained_save_dir", type=str, default=STAINED_SAVE_DIR,
                        help="coregistered IHC images save directory.")
    parser.add_argument("--blend_save_dir", type=str, default=BLEND_SAVE_DIR,
                        help="coregistered blend images save directory.")
    parser.add_argument("--level_1_crop", type=str, default=LEVEL_1_CROP,
                        help="level 1 coregistration crop size.")
    parser.add_argument("--level_1_ds", type=str, default=LEVEL_1_DOWNSAMPLE,
                        help="level 1 coregistration downsample size.")
    parser.add_argument("--level_2_crop", type=str, default=LEVEL_2_CROP,
                        help="level 2 coregistration crop size.")
    parser.add_argument("--level_2_ds", type=str, default=LEVEL_2_DOWNSAMPLE,
                        help="level 2 coregistration downsample size.")

    return parser.parse_args()

def main():

	args = get_arguments()
	for name in glob.glob(args.args.stained_dir + '*.svs'):
		
		print('begin processing ' + os.path.basename(name))
		
		# Read images
		im = open_slide(args.args.im_dir + os.path.basename(name).strip('CK.svs') + 'HE.svs')
		stained = open_slide(name)

		# Error checking
		if(im is None or stained is None):
			print('No image readed')
			continue

		if im.level_count <= 0 or stained.level_count <= 0:
			print('Error image level')
			continue

		# Read 0 level images
		im_size = im.level_dimensions[0]
		stained_size = stained.level_dimensions[0]

		if(os.path.isfile(name.strip('.svs')+ '.xml')):
			# Read annotation file
			region_list = parse_annotation(name = name.strip('.svs')+ '.xml')
			region_mask = generate_mask(original_size = stained_size, region_list = region_list)

			count = np.array([0,0,0], np.uint8)
			_first_level_manipulation(im_full_size = im, stained_full_size = stained, region_mask = region_mask, 
				original_size = stained_size, name = os.path.basename(name), count = count, roi_check = True)
			print(os.path.basename(name) + ' complished')
		
		else:
			count = np.array([0,0,0], np.uint8)
			_first_level_manipulation(im_full_size = im, stained_full_size = stained, region_mask = None, 
				original_size = stained_size, name = os.path.basename(name), count = count, roi_check = False)
			print(os.path.basename(name) + ' complished')


if __name__ == '__main__':
	
	if not os.path.exists(args.im_save_dir):
		os.makedirs(args.im_save_dir)
	if not os.path.exists(args.stained_save_dir):
		os.makedirs(args.stained_save_dir)
	if not os.path.exists(args.blend_save_dir):
		os.makedirs(args.blend_save_dir)	
	main()