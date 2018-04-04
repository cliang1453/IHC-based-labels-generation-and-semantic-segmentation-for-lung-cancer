from glob import glob
import os
import sys
import time
import pprint
import argparse
import glob
from PIL import Image, ImageDraw 
import numpy as np  
import openslide  
from openslide import open_slide  
import xml.etree.ElementTree as ET


STAINED_DIR = '/home/chen/Downloads/Eric/additional_svs/svs/'
STAINED_SAVE_DIR = '/home/chen/Downloads/Eric/additional_svs/red_channel/'
ANNO_TYPE  = 'Attributes' #'Attributes' or 'Text'


def parse_annotation(name = None):
	tree = ET.parse(name)
	root = tree.getroot()
	att_list = []
	ver_list = []
	
	if ANNO_TYPE is 'Attributes':
		for attributes in root.iter('Attributes'):
			for attribute in attributes.iter('Attribute'):
				if attribute.attrib['Value']=='':
					break
				att_list.append(attribute.attrib['Value'])
	else:
		for attribute in root.iter('Region'):
			att_list.append(attribute.attrib['Text'])
	

	for vertices in root.iter('Vertices'):
		region_vertices = []
		for vertex in vertices.iter('Vertex'):
			region_vertices.append((int(float(vertex.attrib['X'])),
				                   int(float(vertex.attrib['Y']))))
		ver_list.append(region_vertices)

	return att_list, ver_list

def write_annotated_img(stained = None, att_list = [], ver_list = [], name = None):
	cnt = {'NT': 0, 
	        'MICROPAP': 0, 
	        'ACINAR': 0,
	        'SOLID': 0}

	for i in range(len(att_list)):

		attribute = att_list[i]
		vertices = ver_list[i] # lower left, lower right, upper right, upper left 
		current_crop = np.array(stained.read_region((vertices[0][0], vertices[0][1]), 0, (vertices[2][0]-vertices[0][0], vertices[2][1]-vertices[0][1])).convert('RGB'))	
		stained_save = Image.fromarray(current_crop.astype(np.uint8))
		stained_save.save(os.path.join(STAINED_SAVE_DIR, os.path.basename(name).strip('.svs') + attribute + str(cnt[attribute]) + '.tif'))
		cnt[attribute] = cnt[attribute] + 1


def main():

	for name in glob.glob(STAINED_DIR + '*.svs'):
		print('begin processing ' + os.path.basename(name))
		
		if(os.path.isfile(name.strip('.svs') + '.xml')):
			stained = open_slide(name)
			
			if(stained is None):
				print('No image readed')
				continue
			if stained.level_count <= 0:
				print('Error image level')
				continue

			stained_size = stained.level_dimensions[0]
			att_list, ver_list = parse_annotation(name = name.strip('.svs')+ '.xml')
			write_annotated_img(stained, att_list, ver_list, name)
			

if __name__ == '__main__':
	
	if not os.path.exists(STAINED_SAVE_DIR):
		os.makedirs(STAINED_SAVE_DIR)

	main()