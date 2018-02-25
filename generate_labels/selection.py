import getch
import numpy as np 
import os
import sys
from PIL import Image
from os.path import join
import time
import psutil
import glob


# Press "Enter" key -- select image for training 
# Press "Delete" key -- move unsatisfactory image from DATA_DIR to NOT_SELECTED_DIR

DATA_DIR = # Absolute path to the stained images foler
NOT_SELECTED_DIR = # Absolute path to a new folder that temporarily store the unsatisfactory stained images


def enter(filename = None):
	print(os.path.basename(filename) + " is selected for training")

def delete(filename = None, curr_list = None):
	print(os.path.basename(filename) + " is not selected")
	curr_list.append(filename)

def switch_folder(curr_list = None):
	for file in curr_list:
		im = Image.open(file)
		im.save(NOT_SELECTED_DIR + os.path.basename(file))
		os.remove(file)

def main():

	not_selected = []
	for file in glob.glob(os.path.join(DATA_DIR, '*.png')):
		im = Image.open(file)
		im.show()

		while True:
			key = ord(getch.getch())
			if key == 10: #Enter
				enter(filename = file)
				break
			elif key == 127: #Delete
				delete(filename = file, curr_list = not_selected)
				break
			else:
				print("wrong key pressed!")
				continue
			
		for proc in psutil.process_iter():
		    if proc.name() == "display":
		        proc.kill()

	if not os.path.exists(NOT_SELECTED_DIR):
		os.makedirs(NOT_SELECTED_DIR)
	switch_folder(curr_list = not_selected)


if __name__ == '__main__':
    main()
