import numpy as np 
import os
import pprint
import sys

DATA_DIR = '/media/labshare/_Gertych_projects/_Lung_cancer/_SVS_/Registered_Mask/dataset/images/'



def printTrain():
    
    file_paths = []  # List which will store all of the full filepaths.
    file_paths2 = []
   
    count = 0
    for root, directories, files in os.walk(DATA_DIR):
        for filename in files:
            out = 'images/'+ os.path.basename(filename) + ' labelID/' + os.path.basename(filename)
            if (count>=10000):
                file_paths2.append(out)
            else:
                file_paths.append(out)
            count = count + 1
            print(count) 

    with open('train.txt', 'w') as f:
        pp = pprint.PrettyPrinter(stream=f)
        pp.pprint(file_paths)

    with open('val.txt', 'w') as f:
        pp = pprint.PrettyPrinter(stream=f)
        pp.pprint(file_paths2)



def main():
    printTrain()




if __name__ == "__main__":
    main()