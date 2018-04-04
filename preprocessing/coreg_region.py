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
import pandas as pd

from scipy.ndimage.morphology import binary_fill_holes, binary_closing, binary_dilation  
from skimage.color import rgb2gray  
from skimage.morphology import closing, binary_closing, disk, remove_small_holes, dilation, remove_small_objects  
from skimage import color, morphology, filters, exposure, feature
import xml.etree.ElementTree as ET

IM_DIR = '/media/labshare/_Gertych_projects/_Lung_cancer/_SVS_/IHC-HE_3/HE/'
STAINED_DIR = '/media/labshare/_Gertych_projects/_Lung_cancer/_SVS_/IHC-HE_3/IHC/'
IM_SAVE_DIR = '/media/chen/data2/Lung_project/new_dataset/IHC-HE_3_2/images_rgb/'
STAINED_SAVE_DIR = '/media/chen/data2/Lung_project/new_dataset/IHC-HE_3_2/stained_rgb/'
BLEND_SAVE_DIR = '/media/chen/data2/Lung_project/new_dataset/IHC-HE_3_2/blend_rgb/'
NO_TRANS_SAVE_DIR = '/media/chen/data2/Lung_project/new_dataset/IHC-HE_3_2/no_trans_blend_rgb/'


LEVEL_1_CROP = 5000
LEVEL_1_DOWNSAMPLE = 100

LEVEL_2_CROP = 1000
LEVEL_2_DOWNSAMPLE = 5

LEVEL_3_CROP = 500

MIN_MATCH_COUNT = 6
FLANN_INDEX_KDTREE = 0
RESIZE_NEEDED = False
ROI_THRES = 0.8



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

	#if len(des1)<1 or des2 is None

	# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	# search_params = dict(checks = 50)
	# flann = cv2.FlannBasedMatcher(index_params, search_params)
	
	# if len(kp1)<2 or len(kp2)<2:
	# 	print "Not enough key points found"
	# 	return None

	# matches = flann.knnMatch(des1,des2,k=2)
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1, des2, k=2)
	print(matches)

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



	draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
	final = cv2.drawMatches(src,kp1,dest,kp2,good,None,**draw_params)

	plt.imshow(final, 'gray'),plt.show()
	return M
	

def _third_level_manipulation(im_full_size = None, stained_full_size = None, no_trans = None, original_size = (LEVEL_2_CROP, LEVEL_2_CROP), name = None, count = None):

	print('maximum crops in level III: 2*2=4')

	for w in range(original_size[0]/LEVEL_3_CROP):
		for h in range(original_size[1]/LEVEL_3_CROP):

			im = im_full_size[h * LEVEL_3_CROP : (h+1) * LEVEL_3_CROP,  w * LEVEL_3_CROP: (w+1) * LEVEL_3_CROP].copy()  
			stained = stained_full_size[h * LEVEL_3_CROP : (h+1) * LEVEL_3_CROP,  w * LEVEL_3_CROP: (w+1) * LEVEL_3_CROP].copy()  
			no_trans = no_trans[h * LEVEL_3_CROP : (h+1) * LEVEL_3_CROP,  w * LEVEL_3_CROP: (w+1) * LEVEL_3_CROP].copy()
			blend = cv2.addWeighted(im,0.5,stained,0.5,0)

			count[2] = count[2] + 1

			im_result = Image.fromarray(im.astype(np.uint8))
			stained_result = Image.fromarray(stained.astype(np.uint8))
			blend_result = Image.fromarray(blend.astype(np.uint8))

			im_result.save(os.path.join(IM_SAVE_DIR, name.strip('.svs') + str(count[0]) + '_' + str(count[1]) + '_' + str(count[2]) + '.png'))
			stained_result.save(os.path.join(STAINED_SAVE_DIR, name.strip('.svs') + str(count[0]) + '_' + str(count[1]) + '_' + str(count[2]) + '.png'))
			blend_result.save(os.path.join(BLEND_SAVE_DIR, name.strip('.svs') + str(count[0]) + '_' + str(count[1]) + '_' + str(count[2]) + '.png'))
			
			if w==0 and h==0:
				blend_no_trans = cv2.addWeighted(im,0.5,no_trans,0.5,0)
				blend_no_trans_result = Image.fromarray(blend_no_trans.astype(np.uint8))
				blend_no_trans_result.save(os.path.join(NO_TRANS_SAVE_DIR, name.strip('.svs') + str(count[0]) + '_' + str(count[1]) + '_' + str(count[2]) + '.png'))



def _second_level_manipulation(im_full_size = None, stained_full_size = None, original_size = (LEVEL_1_CROP, LEVEL_1_CROP), name = None, count = None):

	print('maximum crops in level II: ' + str(original_size[0]/LEVEL_2_CROP) + 
		'*' + str(original_size[1]/LEVEL_2_CROP) + '=' + str(original_size[0]/LEVEL_2_CROP * original_size[1]/LEVEL_2_CROP))

	for w in range(original_size[0]/LEVEL_2_CROP):
		for h in range(original_size[1]/LEVEL_2_CROP):
			
			im = im_full_size[h * LEVEL_2_CROP : (h+1) * LEVEL_2_CROP,  w * LEVEL_2_CROP: (w+1) * LEVEL_2_CROP].copy()  
			stained = stained_full_size[h * LEVEL_2_CROP : (h+1) * LEVEL_2_CROP,  w * LEVEL_2_CROP: (w+1) * LEVEL_2_CROP].copy()  

			im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			stained_gray = cv2.cvtColor(stained, cv2.COLOR_BGR2GRAY)

			im_ds = cv2.resize(im_gray, None, fx=1.0/LEVEL_2_DOWNSAMPLE, fy=1.0/LEVEL_2_DOWNSAMPLE, interpolation = cv2.INTER_LINEAR)
			stained_ds = cv2.resize(stained_gray, None, fx=1.0/LEVEL_2_DOWNSAMPLE, fy=1.0/LEVEL_2_DOWNSAMPLE, interpolation = cv2.INTER_LINEAR)

			# Get Transformation matrix from stained_ds to im_ds
			M = _get_homography(stained_ds, im_ds, flann_ratio = 0.75)
			if M is None:
				stained_transformed = stained
				print('1000 * 1000 crop at position' + str(h) + ',' + str(w) + 'encounters problem in level II, discarded')
			
			if M is not None:
				M[0][2] = M[0][2] *LEVEL_2_DOWNSAMPLE
				M[1][2] = M[1][2] *LEVEL_2_DOWNSAMPLE
				M[2][0] = M[2][0] /LEVEL_2_DOWNSAMPLE
				M[2][1] = M[2][1] /LEVEL_2_DOWNSAMPLE

				# Apply Transformation matrix to original resolution stained im crop
				r = cv2.warpPerspective(stained[:, :, 0], M, (stained.shape[0], stained.shape[1]))
				g = cv2.warpPerspective(stained[:, :, 1], M, (stained.shape[0], stained.shape[1]))
				b = cv2.warpPerspective(stained[:, :, 2], M, (stained.shape[0], stained.shape[1]))
				stained_transformed = cv2.merge((r, g, b))

			count[1] = count[1] + 1
			_third_level_manipulation(im_full_size = im, stained_full_size = stained_transformed, 
				                      no_trans = stained, original_size = (LEVEL_2_CROP, LEVEL_2_CROP), name = name, count = count)


def _first_level_manipulation(im_full_size = None, stained_full_size = None, region_mask = None, 
				original_size = None, crop_ratio = (1, 1), name = None, count = None):


	print('maximum crops in level I: ' + str(original_size[0]/LEVEL_1_CROP) + 
		'*' + str(original_size[1]/LEVEL_1_CROP) + '=' + str(original_size[0]/LEVEL_1_CROP * original_size[1]/LEVEL_1_CROP))
	crop_w_ratio = crop_ratio[0]
	crop_h_ratio = crop_ratio[1]

	
	for w in range(original_size[0]/LEVEL_1_CROP):
		for h in range(original_size[1]/LEVEL_1_CROP):
			
			if(check_roi(w = w, h = h, region_mask = region_mask, crop_size = LEVEL_1_CROP) is False):
				continue

			im = np.array(im_full_size.read_region((LEVEL_1_CROP * w, LEVEL_1_CROP * h), 0, (LEVEL_1_CROP , LEVEL_1_CROP)).convert('RGB'))
			stained = np.array(stained_full_size.read_region((int(LEVEL_1_CROP * crop_w_ratio * w), int(LEVEL_1_CROP * crop_h_ratio * h)),
				                          0, (int(LEVEL_1_CROP * crop_w_ratio), int(LEVEL_1_CROP * crop_h_ratio))).convert('RGB'))


			if crop_w_ratio is not 1 or crop_h_ratio is not 1:
				stained = cv2.resize(stained, (im.shape[0], im.shape[1]))

			im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
			stained_gray = cv2.cvtColor(stained, cv2.COLOR_RGB2GRAY)

			im_ds = cv2.resize(im_gray, None, fx=1.0/LEVEL_1_DOWNSAMPLE, fy=1.0/LEVEL_1_DOWNSAMPLE, interpolation = cv2.INTER_LINEAR)
			stained_ds = cv2.resize(stained_gray, None, fx=1.0/LEVEL_1_DOWNSAMPLE, fy=1.0/LEVEL_1_DOWNSAMPLE, interpolation = cv2.INTER_LINEAR)


			blend_ds = cv2.addWeighted(im_ds,0.5,stained_ds,0.5,0)
			blend_ds_result = Image.fromarray(blend_ds.astype(np.uint8))
			blend_ds_result.save(os.path.join("/media/chen/data2/Lung_project/new_dataset/IHC-HE_3_2/test/", name.strip('.svs') + str(count[0]) + '.png'))


			#Get Transformation matrix from stained_ds to im_ds
			M = _get_homography(stained_ds, im_ds, flann_ratio = 0.75)
			if M is None:
				print('3000 * 3000 crop at position' + str(h) + ',' + str(w) + 'encounters problem in level I, discarded')
				stained_transformed = stained
				
			if M is not None:
				M[0][2] = M[0][2] *LEVEL_1_DOWNSAMPLE
				M[1][2] = M[1][2] *LEVEL_1_DOWNSAMPLE
				M[2][0] = M[2][0] /LEVEL_1_DOWNSAMPLE
				M[2][1] = M[2][1] /LEVEL_1_DOWNSAMPLE

				# Apply Transformation matrix to original resolution stained im crop
				r = cv2.warpPerspective(stained[:, :, 0], M, (stained.shape[0], stained.shape[1]))
				g = cv2.warpPerspective(stained[:, :, 1], M, (stained.shape[0], stained.shape[1]))
				b = cv2.warpPerspective(stained[:, :, 2], M, (stained.shape[0], stained.shape[1]))
				stained_transformed = cv2.merge((r, g, b))

			count[0] = count[0] + 1
			_second_level_manipulation(im_full_size = im, stained_full_size = stained_transformed, 
				                       original_size = (LEVEL_1_CROP, LEVEL_1_CROP), name = name, count = count)




def main():

	for name in glob.glob(STAINED_DIR + '*.svs'):
		print('begin processing ' + os.path.basename(name))
		# CASE I: Dealing with marked up region
		if(os.path.isfile(name.strip('.svs')+ '.xml')):
			# Read images
			im = open_slide(IM_DIR + os.path.basename(name).strip('CK.svs') + 'HE.svs')
			stained = open_slide(name)


			# Error checking
			if(im is None or stained is None):
				print('No image readed')
				continue

			# Error checking 
			if im.level_count <= 0 or stained.level_count <= 0:
				print('Error image level')
				continue

			# Read 0 level images
			im_size = im.level_dimensions[0]
			stained_size = stained.level_dimensions[0]
			print(im_size)
			print(stained_size)

			
			# Read annotation file
			region_list = parse_annotation(name = name.strip('.svs')+ '.xml')
			region_mask = generate_mask(original_size = stained_size, region_list = region_list)
			
			if im_size != stained_size:
				crop_w_ratio = np.float32(stained_size[0])/np.float32(im_size[0])
				crop_h_ratio = np.float32(stained_size[1])/np.float32(im_size[1])

			count = np.array([0,0,0], np.uint8)
			_first_level_manipulation(im_full_size = im, stained_full_size = stained, region_mask = region_mask, 
				original_size = stained_size, crop_ratio = (crop_w_ratio, crop_h_ratio), name = os.path.basename(name), count = count)
			print(os.path.basename(name) + ' complished')
			# Case II: Dealing with WSI
		else:
			continue

		#print(os.path.basename(name) + ' complished')

if __name__ == '__main__':
	
	if not os.path.exists(IM_SAVE_DIR):
		os.makedirs(IM_SAVE_DIR)
	if not os.path.exists(STAINED_SAVE_DIR):
		os.makedirs(STAINED_SAVE_DIR)
	if not os.path.exists(BLEND_SAVE_DIR):
		os.makedirs(BLEND_SAVE_DIR)
	if not os.path.exists(NO_TRANS_SAVE_DIR):
		os.makedirs(NO_TRANS_SAVE_DIR)
	
	main()




