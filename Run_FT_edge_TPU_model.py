import numpy as np
import time
import os
import cv2
import matplotlib.pyplot as plt
import math

# PYCORAL IMPLEMENTATION
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter

'''
    This script will test out a single edge TPU model

'''



pathToQuantizedEdgeTPUModel = 'tflites/test_FT_quantized_int8_uint8_edgetpu.tflite'
assert os.path.exists(pathToQuantizedEdgeTPUModel), "Path to the quantized edge tpu model does not exist"
SpeedTest = True          # if set to true, we run a speedtest to see how fast the model can run
SpeedTest_num = 1000        # the amount of img to run the speedtest with

ImgTest = False             # if set to true, will try to run the model on some images which are also
ImgTest_pathToTestImg = 'testingImg'            # full things that are wonderful
ImgTest_pathToResultFolder = 'resulting_img'
ImgTest_showResults = False                     # if set to true, we show the results from the edge tpu
ImgTest_saveResults = True                      # if set to true, we save our results in ImgTest_pathToResultsFolder

UnitTest = False
UnitTest_path = 'C:/Users/MFolmer/Downloads/PostProcessed'
UnitTest_num = 100

# which keypoints are connected (relevant for the Img Test)
kp_connected = [[[0, 1], [1, 2], [2, 3], [3, 4]],
                [[0, 5], [5, 6], [6, 7], [7, 8]],
                [[0, 9], [9, 10], [10, 11], [11, 12]],
                [[0, 13], [13, 14], [14, 15], [15, 16]],
                [[0, 17], [17, 18], [18, 19], [19, 20]]
                ]
kp_color = ['r', 'g', 'b', 'c', 'y']

# clear the files from our results folder
if ImgTest or UnitTest:
    if not os.path.exists(ImgTest_pathToResultFolder):
        os.mkdir(ImgTest_pathToResultFolder)
    for f in [ImgTest_pathToResultFolder + "/" + f for f in os.listdir(ImgTest_pathToResultFolder)]:
        os.remove(f)

interpreter = make_interpreter(pathToQuantizedEdgeTPUModel)
interpreter.allocate_tensors()

# Get the information of the model
print("")
print("The input details : " + str(interpreter.get_input_details()))
for i_in, tflite_input in enumerate(interpreter.get_input_details()):
    print(f"Input {i_in} -> index :{tflite_input['index']}, Shape :{tflite_input['shape']}, "
          f"dtype : {tflite_input['dtype']}")
print("")
print("The output details : " + str(interpreter.get_output_details()))
for i_out, tflite_output in enumerate(interpreter.get_output_details()):
    print(f"Output {i_out} -> index :{tflite_output['index']}, Shape :{tflite_output['shape']}, "
          f"dtype : {tflite_output['dtype']}")
print("")

# run a Speed Test (check how fast the TPU model can run)
if SpeedTest:
    tot_time = 0
    for ir in range(SpeedTest_num):
        # get a random input image
        random_input = np.random.rand(1, 224, 224, 3)
        random_input *= 255
        random_input = random_input.astype(dtype=np.uint8)

        t1 = time.time()   # start the time
        # setting input
        common.set_input(interpreter, random_input)
        interpreter.invoke()

        # Getting output
        coordinates = common.output_tensor(interpreter, 1).copy()
        visibility = common.output_tensor(interpreter, 2).copy()
        handedness = common.output_tensor(interpreter, 0).copy()

        tot_time += time.time()-t1
        if ir % 100 == 0:
            print(f"Speed test, we are at : {ir} out of {SpeedTest_num}")
    print(f"Results from speed test -> total time : {tot_time} s,  average time : {tot_time/SpeedTest_num}, "
          f"avg {1/(tot_time/SpeedTest_num)} Hz")

# test our Test images
def turn_int_to_float(x,y,z):
    # altering coordinates to uint8
    newXYRange = (1.33 - -0.33)
    oldXYRange = (127 - -128)
    newXYMin = -0.33
    oldXYMin = -128
    multXY = newXYRange / oldXYRange

    newZRange = (0.8 - -0.8)
    oldZRange = (127 - -128)
    newZMin = -0.8
    oldZMin = -128
    multZ = newZRange / oldZRange

    x = [(xc - oldXYMin) * multXY + newXYMin for xc in x]
    y = [(yc - oldXYMin) * multXY + newXYMin for yc in y]
    z = [(zc - oldZMin) * multZ + newZMin for zc in z]

    return x, y, z

def dist(point1, point2):
    return math.sqrt(math.pow(point2[0]-point1[0],2) + math.pow(point2[1]-point1[1],2)+math.pow(point2[2]-point1[2],2))

num_img_result = 0
if ImgTest:

    allImgPath = [ImgTest_pathToTestImg + "/" + f for f in os.listdir(ImgTest_pathToTestImg) if '.png' in f or '.jpg' in f]
    for i_img, imgPath in enumerate(allImgPath):
        img = cv2.imread(imgPath)               # read the image
        img = cv2.resize(img, (224,224))        # resize to correct image dimensions
        img_original = img.copy()
        img = img.astype(dtype=np.uint8)        # set to correct input type
        img = np.reshape(img, (1, 224, 224, 3)) # resize to correct input size

        # run our interpreter
        common.set_input(interpreter, img)
        interpreter.invoke()

        # get the output of the interpreter
        coordinates = common.output_tensor(interpreter, 1).copy()
        visibility = common.output_tensor(interpreter, 2).copy()
        handedness = common.output_tensor(interpreter, 0).copy()

        # postprocessing the coordinates so we can plot them
        xcoor = [coor[0] for coor in coordinates[0]]
        ycoor = [coor[1] for coor in coordinates[0]]
        zcoor = [coor[2] for coor in coordinates[0]]
        xcoor, ycoor, zcoor = turn_int_to_float(xcoor, ycoor, zcoor)
        xcoor = [int(coor * 224) for coor in xcoor]
        ycoor = [int(coor * 224) for coor in ycoor]
        zcoor = [int(coor * 224) for coor in zcoor]

        # plot on top of our image
        for (xc, yc) in zip(xcoor, ycoor):
            cv2.circle(img_original, (xc, yc), 4, (0, 0, 0), -1)
            cv2.circle(img_original, (xc, yc), 3, (255, 255, 255), -1)


        # create a 3D plot of the coordinates (kp_connected)
        fig = plt.figure()
        ax_local = fig.add_subplot(1, 1, 1, projection='3d')

        # draw the links between each connected keypoint
        for i_finger, finger in enumerate(kp_connected):
            for link in finger:
                ax_local.plot((xcoor[link[0]], xcoor[link[1]]), (ycoor[link[0]], ycoor[link[1]]),
                              (zcoor[link[0]], zcoor[link[1]]), kp_color[i_finger])

        # Set visuals for the 3D graph
        ax_local.scatter(xcoor, ycoor, zcoor)
        ax_local.set_xlim(0, 224)
        ax_local.set_ylim(0, 224)
        ax_local.set_zlim(-112, 112)
        ax_local.set_xlabel("x_label")
        ax_local.set_ylabel("y_label")
        ax_local.set_zlabel("z_label")
        ax_local.view_init(-90, -90)


        # show the images
        if ImgTest_showResults:
            cv2.imshow("Plotted xy", img_original)
            plt.show()
            cv2.waitKey(-1)

        if ImgTest_saveResults:   # save the images in our results folder
            cv2.imwrite(ImgTest_pathToResultFolder + f"/img_{num_img_result}_1.png", img_original)
            plt.savefig(ImgTest_pathToResultFolder + f"/img_{num_img_result}_2.png", bbox_inches='tight')

        # Show progress
        if i_img % 25 == 0:
            print(f"We are at testImg {i_img} out of {len(allImgPath)}")

        # increment the amount of images we have in our resulting
        num_img_result += 1
        plt.close()


# unit testing:
if UnitTest:

    X_filenames = np.load(UnitTest_path + "/X_filenames.npy", allow_pickle=True)
    y_coor = np.load(UnitTest_path+"/y_coor.npy", allow_pickle=True)
    y_hand = np.load(UnitTest_path + "/y_handedness.npy", allow_pickle=True)
    y_vis = np.load(UnitTest_path + "/y_vis.npy", allow_pickle=True)

    X_filenames = X_filenames[0:UnitTest_num]
    y_coor = y_coor[0:UnitTest_num]
    y_hand = y_hand[0:UnitTest_num]
    y_vis = y_vis[0:UnitTest_num]

    print("\nUnit test ground truth shapes")
    print(f"X_filenames : {X_filenames.shape}")
    print(f"y_coor : {y_coor.shape}")
    print(f"y_vis : {y_vis.shape}")
    print(f"y_hand : {y_hand.shape}")



    # run the images through the things, compare the output
    MSE_tot = []
    num_correct_hand = 0
    num_correct_vis = 0
    for i_img, (imgFilename, coor_gt, vis_gt, hand_gt) in enumerate(zip(X_filenames, y_coor, y_vis, y_hand)):

        imgGT = cv2.imread(imgFilename)
        imgGT = cv2.resize(imgGT, (224, 224))           # resize to correct image dimensions
        img_original = imgGT.copy()
        imgGT = imgGT.astype(dtype=np.uint8)            # set to correct input type
        imgGT = np.reshape(imgGT, (1, 224, 224, 3))     # resize to correct input size

        # run our interpreter
        common.set_input(interpreter, imgGT)
        interpreter.invoke()

        # get the output of the interpreter
        coordinates = common.output_tensor(interpreter, 1).copy()
        visibility = common.output_tensor(interpreter, 2).copy()
        handedness = common.output_tensor(interpreter, 0).copy()

        # postprocessing the coordinates so we can plot them
        xcoor = [coor[0] for coor in coordinates[0]]
        ycoor = [coor[1] for coor in coordinates[0]]
        zcoor = [coor[2] for coor in coordinates[0]]
        xcoor, ycoor, zcoor = turn_int_to_float(xcoor, ycoor, zcoor)
        xcoor = [int(coor * 224) for coor in xcoor]
        ycoor = [int(coor * 224) for coor in ycoor]
        zcoor = [int(coor * 224) for coor in zcoor]
        coormodel = [(x, y, z) for (x, y, z) in zip(xcoor, ycoor, zcoor)]

        # get the MAE for each image, and the number of correct visibilities and handedness
        xgt = [int(coor[0] * 224) for coor in coor_gt]
        ygt = [int(coor[1] * 224) for coor in coor_gt]
        zgt = [int(coor[2] * 224) for coor in coor_gt]
        coorgt = [(x, y, z) for (x, y, z) in zip(xgt, ygt, zgt)]

        # get the MSE loss between ground truth and model
        MSE = np.mean([dist(p1, p2) for (p1, p2) in zip(coormodel, coorgt)])
        MSE_tot.append(MSE)

        # get the visible/handedness (and whether it has been predicted correctly)
        if visibility[0][0] < 0.5:
            vis_mod = 0
        else:
            vis_mod = 1
        if handedness[0][0] < 0.5:
            hand_mod = 0
        else:
            hand_mod = 1
        if vis_mod == vis_gt[0]:
            num_correct_vis += 1
        if hand_mod == hand_gt[0]:
            num_correct_hand += 1

        # plot the circles on the images
        img_gt = img_original.copy()
        # plot on top of our image
        for (xc, yc) in zip(xgt, ygt):
            cv2.circle(img_gt, (xc, yc), 4, (0, 0, 0), -1)
            cv2.circle(img_gt, (xc, yc), 3, (255, 255, 255), -1)

        for (xc, yc) in zip(xcoor, ycoor):
            cv2.circle(img_original, (xc, yc), 4, (0, 0, 0), -1)
            cv2.circle(img_original, (xc, yc), 3, (255, 255, 255), -1)
        img_final = np.concatenate([img_gt, img_original], axis=1)
        cv2.putText(img_final, "Ground truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
        cv2.putText(img_final, "From model", (240, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        # cv2.imshow('img final', img_final)
        # cv2.waitKey(-1)

        # save the image
        cv2.imwrite(ImgTest_pathToResultFolder + f"/img_{num_img_result}_1.png", img_final)


        # 3D model
        # create a 3D plot of the coordinates (kp_connected)
        fig = plt.figure()
        ax_local = fig.add_subplot(1, 1, 1, projection='3d')

        # draw the links between each connected keypoint
        for i_finger, finger in enumerate(kp_connected):
            for link in finger:
                ax_local.plot((xcoor[link[0]], xcoor[link[1]]), (ycoor[link[0]], ycoor[link[1]]),
                              (zcoor[link[0]], zcoor[link[1]]), kp_color[i_finger])

        # Set visuals for the 3D graph
        ax_local.scatter(xcoor, ycoor, zcoor)
        ax_local.set_xlim(0, 224)
        ax_local.set_ylim(0, 224)
        ax_local.set_zlim(-112, 112)
        ax_local.set_xlabel("x_label")
        ax_local.set_ylabel("y_label")
        ax_local.set_zlabel("z_label")
        ax_local.view_init(-90, -90)

        # Save plots
        plt.savefig(ImgTest_pathToResultFolder + f"/img_{num_img_result}_2_1.png", bbox_inches='tight')
        ax_local.view_init(0, -90) # set to top view
        plt.savefig(ImgTest_pathToResultFolder + f"/img_{num_img_result}_2_2.png", bbox_inches='tight')

        # combine both plots
        img1 = cv2.imread(ImgTest_pathToResultFolder + f"/img_{num_img_result}_2_1.png")
        img2 = cv2.imread(ImgTest_pathToResultFolder + f"/img_{num_img_result}_2_2.png")
        img1 = cv2.resize(img1, (300, 300))
        img2 = cv2.resize(img2, (300, 300))
        imgTot = np.concatenate([img1,img2], axis=1)
        cv2.putText(imgTot, "From Model", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv2.imwrite(ImgTest_pathToResultFolder + f"/img_{num_img_result}_2.png", imgTot)
        os.remove(ImgTest_pathToResultFolder + f"/img_{num_img_result}_2_1.png")
        os.remove(ImgTest_pathToResultFolder + f"/img_{num_img_result}_2_2.png")

        # close the plot
        plt.close()

        # Do the same, but for ground truth
        fig = plt.figure()
        ax_local = fig.add_subplot(1, 1, 1, projection='3d')

        # draw the links between each connected keypoint
        for i_finger, finger in enumerate(kp_connected):
            for link in finger:
                ax_local.plot((xgt[link[0]], xgt[link[1]]), (ygt[link[0]], ygt[link[1]]),
                              (zgt[link[0]], zgt[link[1]]), kp_color[i_finger])

        # Set visuals for the 3D graph
        ax_local.scatter(xgt,ygt, zgt)
        ax_local.set_xlim(0, 224)
        ax_local.set_ylim(0, 224)
        ax_local.set_zlim(-112, 112)
        ax_local.set_xlabel("x_label")
        ax_local.set_ylabel("y_label")
        ax_local.set_zlabel("z_label")
        ax_local.view_init(-90, -90)

        # save the plots
        plt.savefig(ImgTest_pathToResultFolder + f"/img_{num_img_result}_3_1.png", bbox_inches='tight')
        ax_local.view_init(0, -90)      # top view
        plt.savefig(ImgTest_pathToResultFolder + f"/img_{num_img_result}_3_2.png", bbox_inches='tight')

        # combine both plots
        img1 = cv2.imread(ImgTest_pathToResultFolder + f"/img_{num_img_result}_3_1.png")
        img2 = cv2.imread(ImgTest_pathToResultFolder + f"/img_{num_img_result}_3_2.png")
        img1 = cv2.resize(img1, (300, 300))
        img2 = cv2.resize(img2, (300, 300))
        imgTot = np.concatenate([img1, img2], axis=1)
        cv2.putText(imgTot, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv2.imwrite(ImgTest_pathToResultFolder + f"/img_{num_img_result}_3.png", imgTot)
        os.remove(ImgTest_pathToResultFolder + f"/img_{num_img_result}_3_1.png")
        os.remove(ImgTest_pathToResultFolder + f"/img_{num_img_result}_3_2.png")

        # close the plot
        plt.close()

        # increment our saving
        num_img_result+=1

        # progressbar
        if i_img % 25==0:
            print(f"We are at unit test : {i_img} / {X_filenames.shape[0]}")

    # Give us a report
    print(f"\nReport on : {pathToQuantizedEdgeTPUModel.split('/')[-1]}")
    print(f"   Avg MSE coordinates : {np.mean(MSE_tot)}")
    print(f"   Percentage visibility : {num_correct_vis/X_filenames.shape[0]}, ({num_correct_vis}/{X_filenames.shape[0]})")
    print(f"   Percentage handedness : {num_correct_hand/X_filenames.shape[0]}, ({num_correct_hand}/{X_filenames.shape[0]})")

    if SpeedTest:
        print(f"   Speed test -> total time : {tot_time} s,  average time : {tot_time / SpeedTest_num}, "
              f"avg {1 / (tot_time / SpeedTest_num)} Hz")

