import numpy as np
import time
import os
import cv2

# PYCORAL IMPLEMENTATION
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter


'''
    This script will run the current model on some fingertracking 

'''

pathToOutputsNpy = 'FT_unitTest/outputs'

def runImgOnTPU(_allImgPaths, _interpreter):
    Outputs = []
    for i_img, imgPath in enumerate(_allImgPaths):

        # read the images
        img = cv2.imread(imgPath)
        img = cv2.resize(img, (224, 224))  # resize to correct image dimensions
        img = img.astype(dtype=np.uint8)  # set to correct input type
        img = np.reshape(img, (1, 224, 224, 3))  # resize to correct input size

        # run our interpreter
        common.set_input(_interpreter, img)
        _interpreter.invoke()

        # get the output of the interpreter
        coordinates = common.output_tensor(_interpreter, 2).copy()
        visibility = common.output_tensor(_interpreter, 0).copy()
        handedness = common.output_tensor(_interpreter, 1).copy()

        Outputs.append([coordinates, visibility, handedness])
    return Outputs


pathToOutputsNpy = 'FT_unitTest/outputs'
allCoorGT = np.load(pathToOutputsNpy+"coor.npy")
allvisGT = np.load(pathToOutputsNpy+"vis.npy")
allhandGT = np.load(pathToOutputsNpy+"hand.npy")

# initialize our interpreter
pathToQuantizedEdgeTPUModel = 'test_FT_quantized_int8_uint8_edgetpu.tflite'
print("Initializing the FingerTracking model with name"+ str(pathToQuantizedEdgeTPUModel))
interpreter = make_interpreter(pathToQuantizedEdgeTPUModel)
interpreter.allocate_tensors()
print("Initialized the FingerTracking model with name"+ str(pathToQuantizedEdgeTPUModel))

# get all images we wish to run through the image test
print("Commencing the unit test")
pathToImg = 'FT_unitTest'
allImgPaths = [pathToImg + "/" +filePath for filePath in os.listdir(pathToImg) if '.png' in filePath or '.jpg' in filePath]
Outputs = runImgOnTPU(allImgPaths, interpreter)
allCoor = [output[0] for output in Outputs]
allvis = [output[1] for output in Outputs]
allhand = [output[2] for output in Outputs]

mse_allCoor = allCoorGT - allCoor
mse_allvis = allvisGT - allvis
mse_allhand = allhandGT - allhand


diffvis = sum(mse_allvis)[0][0]
diffcoor = sum(sum(mse_allCoor))
diffcoor = sum([sum(coor) for coor in diffcoor])
diffhand = sum(mse_allhand)[0][0]

print("\nResults unit test, these values should all be 0 : ")
print("    Diff vis : "+str(diffvis))
print("    Diff hand : "+str(diffhand))
print("    Diff coor : "+str(diffcoor))

# Try a speed test
random_input = np.random.rand(224, 224, 3)
random_input = random_input[:, :] * 255
random_input = random_input.astype(dtype=np.uint8)
random_input = np.reshape(random_input, (1, 224, 224, 3))  # resize to correct input size

# run our interpreter
print("\nInitializing speed test:")
tstart = time.time()
t1 = tstart
numSpeedtest = 1000
for i in range(numSpeedtest):
    common.set_input(interpreter, random_input)
    interpreter.invoke()

    if (i+1) % 100 == 0:
        totTime = time.time()-t1
        print("We are at "+str(i+1) + " our of "+str(numSpeedtest) + " : current avg speed per frame is "+str(round(100/totTime))+" Hz")
        t1 = time.time()

tend = time.time()-tstart
print("\nFinished with speed test.\n    Avg speed per frame is "+str(round(numSpeedtest/tend)) + " Hz")