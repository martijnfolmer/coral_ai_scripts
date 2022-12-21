import numpy as np
import time
import os
import cv2
import gc

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# PYCORAL IMPLEMENTATION
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter

SpeedTest = True                # if set to true, create a speed test to see how fast the TPU works
SpeedTest_num = 1000

imgTest = False                # Testing on the images
imgTest_num = 100
imgTest_path = 'testing_img'
imgTest_path_to_Save = 'resulting_testing_img_full_int'

unitTest = False                 # testing on training image
unitTest_path_to_test ='D:/DataSets/HandDetection/RLAndBlender_200k_40boxes/'   # location of X_filenames and y_coor
unitTest_num_img = 100

offset = 0.0
scale = 1.0

# Set up our layers and number of boxes
layerWidths = [28, 14, 7, 4, 2]  # for 224x224 (1 class, 1 channel) in mobilenet. # same as Classification_i (a,a,w)

# 190 boxes
# aspect_ratios = [[4.0, 5.0, 6.0, 7.0],
#                 [3.0, 4.0, 5.0, 6.0],
#                 [1.0, 1.5, 2.0, 3.0],
#                 [2.0/3.0, 1.0, 1.5, 2.0],
#                 [1.0 / 3.0, 2.0/3.0, 1.0]]

# 40 boxes
aspect_ratios = [[5.0],  # for each layer, this is the ratio of the box we check for. So 4.0 = box is 4 times
                 [4.0],  # larger than if we make the boxes in the grid with the same width and height
                 [0.75, 1.0, 1.5, 2.0, 3.0],
                 [2.0 / 3.0, 1.0, 1.5, 2.0],
                 [1.0 / 3.0, 2.0 / 3.0, 1.0, 2.0]]


# create the tpu model
pathToQuantizedEdgeTPUModel = 'tflites/hand_detection_1_0_50_boxes_uint8_input_int8_output_fullIntCompiled_edgetpu.tflite'
interpreter = make_interpreter(pathToQuantizedEdgeTPUModel)
interpreter.allocate_tensors()

# Get information about the input and output of the edge TPU model
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
        random_input = np.random.rand(1, 224, 224, 3)           # can automate this as input size
        random_input *= 255
        random_input = random_input.astype(dtype=np.uint8)

        t1 = time.time()   # start the time
        # setting input
        common.set_input(interpreter, random_input)
        interpreter.invoke()

        # Getting output
        output = common.output_tensor(interpreter, 0).copy()

        tot_time += time.time()-t1
        if ir % 100 == 0:
            print(f"Speed test, we are at : {ir} out of {SpeedTest_num}")
    print(f"Results from speed test -> total time : {tot_time} s,  average time : {tot_time/SpeedTest_num}, "f"avg {1/(tot_time/SpeedTest_num)} Hz")



# all the functions for postprocessing the bounding boxes
def get_boxes(image_size, num_boxes, scale, aspect_ratio):

    '''

    :param image_size: The width and height of the image we want to create boxes for
    :param num_boxes: How many boxes we want in the width and height
    :param scale: The size of the box beyond (if >0, boxes will overlap)
    :param aspect_ratio : list of width/height of our box, so if=2,  width = 2 * height, while width * height remains the same

    :return: boxes [x1,y1,x2,y2], centres [xcentre, ycentre], hw [height,width]
    '''

    # get all centres
    width = (image_size / num_boxes)  # width and height of the boxes if we use
    height = (image_size / num_boxes)
    all_x = np.linspace(width / 2, image_size - width / 2, num_boxes)
    all_y = np.linspace(height / 2, image_size - height / 2, num_boxes)

    boxes = []  # [x1,y1,x2,y2]
    centres = []  # [x centre, y centre]
    hw = []  # [height, width]
    for x_coor in all_x:
        for y_coor in all_y:
            for asp in aspect_ratio:
                asp1 = asp ** 0.5  # width aspect ratio
                asp2 = 1 / asp1  # height aspect ratio            # asp1 * asp2 = 1.0

                asp1 = asp
                asp2 = asp

                width = (image_size / num_boxes) * asp1
                height = (image_size / num_boxes) * asp2

                centres.append([x_coor, y_coor])
                hw.append([height * scale, width * scale])
                boxes.append(
                    [x_coor - (width * scale) / 2, y_coor - (height * scale) / 2, x_coor + (width * scale) / 2,
                     y_coor + (height * scale) / 2])

    return boxes, centres, hw


def softmax_func(x):

    '''

    :param x: Returns the softmax of a vector x
    :return: returns softmax of a vector x
    '''

    # e_x = np.exp(x - np.max(x))
    # return e_x / e_x.sum(axis=0)  # only difference

    x = [list(np.exp(coor)/np.sum(np.exp(coor))) for coor in x]      # axis = 1
    x = np.asarray(x)
    return x


def get_max_indices(arr, n):

    '''

    :param arr : array that we want to get the max from
    :param n: How many n indices we want
    :return: a list of all the max values
    '''

    arr2 = list(arr.copy())           # copy our array
    arr2.sort(reverse=True)                  # sort it, from largest to smalles
    n_largest = arr2[n]
    l_indices = []
    for i in range(len(arr)):
        if arr[i] > n_largest:
            l_indices.append(i)

    return l_indices


def get_ouput_boxes(layerWidths, numBoxes, aspect_ratios):

    BOXES = sum([a * a * b for a, b in zip(layerWidths, numBoxes)])
    centres = np.zeros((BOXES, 2))
    hw = np.zeros((BOXES, 2))
    boxes = np.zeros((BOXES, 4))
    gc.collect()

    IMG_SIZE = 224
    scale = 1.2

    kn = 0
    for width, ar in zip(layerWidths, aspect_ratios):
        boxes_cur, centres_cur, hw_cur = get_boxes(IMG_SIZE, width, scale, ar)
        boxes[kn:kn + len(boxes_cur)] = boxes_cur
        centres[kn:kn + len(centres_cur)] = centres_cur
        hw[kn:kn + len(hw_cur)] = hw_cur
        kn += len(boxes_cur)

    return boxes, centres, hw, BOXES


def infer(Y):

    NUM_CLASSES = 1                         # how many classes we trained for (in our case, 1, for something)
    outputChannels = NUM_CLASSES + 1 + 4        # total number of channels, which is amount of classes + background and delta
    OBJperCLASS = 100
    Y = np.squeeze(Y)     # There is one too many dimensions, so we need to get rid of it

    # class predictions = confidence score for each of the boxes, for each of the classes
    class_predictions = softmax_func(Y[:, :outputChannels - 4])     # softmax across horizontal axis (so which is most likely)
    # class predictions has same length as BOXES, and same width as number of classes + 1 for background
    # if there is only one class we are interested in, (such as the nova), we only look at that class

    # so for index 0 and 1, which are nova and background. But we are only interested in the nova for now:
    # classes = the indexes of where our max values are.

    nova_class_index = 0
    max_indices = get_max_indices(class_predictions[:, nova_class_index], OBJperCLASS)          # we only want the nova predictions
    conf = [class_predictions[coor, nova_class_index] for coor in max_indices]
    delta = [Y[coor, outputChannels-4:] for coor in max_indices]        # get last 4 elements from boxes, which are delta

    return conf, max_indices, delta


def Bbox(confidence, box_idx, delta):

    '''

    :param confidence:      The confidence score for each of the boxes
    :param box_idx:         The indexes of the maximum
    :param delta:           cx, cy, h, w
    :return:
    '''

    OBJperCLASS = min(len(box_idx),100)
    # delta contains delta(cx,cy,h,w)
    bbox_centre = np.zeros((OBJperCLASS, 2))
    bbox_hw = np.zeros((OBJperCLASS, 2))

    for i in range(OBJperCLASS):
        bbox_centre[i, 0] = centres[box_idx[i]][0] + delta[i][0]
        bbox_centre[i, 1] = centres[box_idx[i]][1] + delta[i][1]
        bbox_hw[i, 0] = hw[box_idx[i]][0] + delta[i][2]
        bbox_hw[i, 1] = hw[box_idx[i]][1] + delta[i][3]

    return bbox_centre, bbox_hw


def box_in_box_iou(boxes, confidence, threshold):

    # boxes = [[x1,y1,x2,y2]]
    # This calculates how much of an area of a box is inside another box. If too much (>threshold), we remove it

    all_idx_blocked = []
    for i in range(len(boxes)):
        box1 = boxes[i]
        for j in range(len(boxes)):
            if i is not j:
                box2 = boxes[j]

                # compute the area of intersection rectangle
                xA = max(box1[0], box2[0])
                yA = max(box1[1], box2[1])
                xB = min(box1[2], box2[2])
                yB = min(box1[3], box2[3])
                SI = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                box1_area = (abs(box1[2]-box1[0])*abs(box1[3]-box1[1]))
                box2_area = (abs(box2[2]-box2[0])*abs(box2[3]-box2[1]))

                if box1_area == 0:
                    box1_area = 0.0001
                if box2_area == 0:
                    box2_area = 0.001

                iou_single_box1 = SI/(box1_area) # how much of the first box overlaps with the second
                iou_single_box2 = SI/(box2_area)

                if iou_single_box1 > threshold and box2_area>box1_area:
                    if i not in all_idx_blocked:
                        all_idx_blocked.append(i)
                elif iou_single_box2 > threshold and box1_area>box2_area:
                    if j not in all_idx_blocked:
                        all_idx_blocked.append(j)

    # delete all elements which have too much stuff
    all_idx_blocked.sort()
    all_idx_blocked.reverse()
    for idx in all_idx_blocked:
        del boxes[idx]
        del confidence[idx]

    return boxes, confidence


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def turn_to_boxes(centres, hw):
    box_coor = [[int(coor[0] - coor2[0] / 2), int(coor[1] - coor2[1] / 2), int(coor[0] + coor2[0] / 2),
                 int(coor[1] + coor2[1] / 2)] for (coor, coor2) in zip(centres, hw)]

    return box_coor


def nms_leaky(confidence, box_coor, iou_threshold, leaky_frac):
    # for each of the boxes, we check out stuff



    if len(box_coor) == 0:
        return [], []
        # initialize the list of picked indexes

    # print(confidence)
    # print("Maximum confidence : " +str(max(confidence)))

    toremove = []
    # for each box, we go down
    for ib in range(len(box_coor) - 1):
        for jb in range(ib + 1, len(box_coor)):
            if ib in toremove or jb in toremove:
                continue
            b1 = box_coor[ib]
            b2 = box_coor[jb]
            iou = bb_intersection_over_union(b1, b2)

            if iou > iou_threshold:   # means it overlaps
                if confidence[ib] > confidence[jb]:
                    toremove.append(jb)
                    box_coor[ib][0] = (b1[0] * confidence[ib] + b2[0] * confidence[jb] * leaky_frac) / (confidence[ib] + confidence[jb] * leaky_frac)
                    box_coor[ib][1] = (b1[1] * confidence[ib] + b2[1] * confidence[jb] * leaky_frac) / (confidence[ib] + confidence[jb] * leaky_frac)
                    box_coor[ib][2] = (b1[2] * confidence[ib] + b2[2] * confidence[jb] * leaky_frac) / (confidence[ib] + confidence[jb] * leaky_frac)
                    box_coor[ib][3] = (b1[3] * confidence[ib] + b2[3] * confidence[jb] * leaky_frac) / (confidence[ib] + confidence[jb] * leaky_frac)
                else:
                    toremove.append(ib)
                    box_coor[jb][0] = (b1[0] * leaky_frac * confidence[ib] + b2[0] * confidence[jb]) / (confidence[ib] * leaky_frac + confidence[jb])
                    box_coor[jb][1] = (b1[1] * leaky_frac * confidence[ib] + b2[1] * confidence[jb]) / (confidence[ib] * leaky_frac + confidence[jb])
                    box_coor[jb][2] = (b1[2] * leaky_frac * confidence[ib] + b2[2] * confidence[jb]) / (confidence[ib] * leaky_frac + confidence[jb])
                    box_coor[jb][3] = (b1[3] * leaky_frac * confidence[ib] + b2[3] * confidence[jb]) / (confidence[ib] * leaky_frac + confidence[jb])

    toreturn_box = []
    toreturn_conf = []
    for i in range(len(box_coor)):
        if i not in toremove:
            toreturn_box.append(box_coor[i])
            toreturn_conf.append(confidence[i])

    return toreturn_box, toreturn_conf


# saving the image
def draw_bb_results(img, confidence, boxes, path_to_save, show_now):

    fig, ax = plt.subplots(1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)

    for k in range(len(confidence)):
        # print("{}: Confidence-{}\t\tx1,y1-{} x2,y2-{}".format(k, confidence[k], [boxes[k][0], boxes[k][1]],
        #                                                               [boxes[k][2],boxes[k][3]]))

        # draw bounding box only if confidence scores are high

        if confidence[k] < 0.2:
            continue

        # print("{}: Confidence-{}\t\tx1,y1-{} x2,y2-{}".format(k, confidence[k], [boxes[k][0], boxes[k][1]],
        #                                                       [boxes[k][2], boxes[k][3]]))

        x = boxes[k][0]
        y = boxes[k][1]
        w = boxes[k][2] - boxes[k][0]
        h = boxes[k][3] - boxes[k][1]

        rect = patches.Rectangle((y, x), h, w, linewidth=4, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        rect = patches.Rectangle((y, x), h, w, linewidth=2, edgecolor='w', facecolor='none')
        ax.add_patch(rect)

    plt.savefig(path_to_save)

    if show_now:
        plt.show()
    plt.close()


numBoxes = [len(asp) for asp in aspect_ratios]  # the number of anchor boxes (based on number of ratios)

# could load this beforehand (because we don't always need it)
boxes, centres, hw, BOXES = get_ouput_boxes(layerWidths=layerWidths, numBoxes=numBoxes, aspect_ratios=aspect_ratios)
print(f"Number of boxes : {len(boxes)}")

# remove all images in image
if imgTest or unitTest:
    [os.remove(imgTest_path_to_Save + "/" + f) for f in os.listdir(imgTest_path_to_Save)]
    created_num_img = 0         # how many images have been created

if imgTest:

    all_img_path = [imgTest_path + "/" + f for f in os.listdir(imgTest_path)][0:imgTest_num]

    for i_img, img_path in enumerate(all_img_path):

        # read and prepare the image
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = (img[:, :] + offset) * scale
        img = img.astype(dtype=np.uint8)

        # running model
        common.set_input(interpreter, img)
        interpreter.invoke()

        # Getting output
        output = common.output_tensor(interpreter, 0).copy()

        # turn from our full int8 output back to float
        turnToFloat = True
        if turnToFloat:
            newXYRange = (150 - -150)
            oldXYRange = (127 - -128)
            newXYMin = -150
            oldXYMin = -128
            multXY = newXYRange / oldXYRange
            output = np.asarray(output, dtype=np.float32)
            # alter out y_output not at runtime
            for i_out, outputC in enumerate(output):
                output[i_out] = [(coor - oldXYMin) * multXY + newXYMin for coor in outputC]

        # Postprocessing the outputs
        confidence, box_idx, delta = infer(output)

        # use [x1, y1, x2, y2]
        bbox_centre, bbox_hw = Bbox(confidence, box_idx, delta)  # get the bounding box information we want
        box_coor = turn_to_boxes(bbox_centre, bbox_hw)  # turn to x1, y1, x2, y2
        box_coor, confidence = nms_leaky(confidence=confidence, box_coor=box_coor, iou_threshold=0.3, leaky_frac=0.0)
        # print(box_coor)
        # remove all those that don't meet confidence score
        box_coor_cur = []
        confidence_cur = []
        for i_conf, conf in enumerate(confidence):
            if conf >= 0.6:  # confidence which we
                confidence_cur.append(conf)
                box_coor_cur.append(box_coor[i_conf])
        box_coor = box_coor_cur.copy()
        confidence = confidence_cur.copy()

        # get iou of single boxes
        box_coor, confidence = box_in_box_iou(box_coor, confidence, 0.3)  # threshold is 50%

        # draw the bounding boxes on the image
        draw_bb_results(img, confidence, box_coor, imgTest_path_to_Save + f"/img_{created_num_img}.png", False)

        created_num_img += 1


if unitTest:

    all_mse = []

    path_X_filenames = unitTest_path_to_test + "X_filenames.npy"
    path_y_output = unitTest_path_to_test + "y_output.npy"

    all_ImgPaths = np.load(path_X_filenames, allow_pickle=True)
    all_coor = np.load(path_y_output, allow_pickle=True)

    all_ImgPaths = all_ImgPaths[0:unitTest_num_img]
    all_coor = all_coor[0:unitTest_num_img]

    # for each images
    for i_img, imgPath in enumerate(all_ImgPaths):

        img = cv2.imread(imgPath)
        img = cv2.resize(img, (224, 224))
        img = (img[:, :] + offset) * scale
        img = img.astype(dtype=np.uint8)

        # running model
        common.set_input(interpreter, img)
        interpreter.invoke()

        # Getting output
        output = common.output_tensor(interpreter, 0).copy()

        # turn from our full int8 output back to float
        turnToFloat = True
        if turnToFloat:
            newXYRange = (150 - -150)
            oldXYRange = (127 - -128)
            newXYMin = -150
            oldXYMin = -128
            multXY = newXYRange / oldXYRange
            output = np.asarray(output, dtype=np.float32)
            # alter out y_output not at runtime
            for i_out, outputC in enumerate(output):
                output[i_out] = [(coor - oldXYMin) * multXY + newXYMin for coor in outputC]

        # get the mse between the groundtruth and detected
        output_coordinates = output[0]
        groundtruth_coordinates = all_coor[i_img]

        output_box_coor = output_coordinates[:, 2:]
        groundtruth_box_coor = groundtruth_coordinates[:, 1:]

        mse_cur = ((output_box_coor - groundtruth_box_coor)**2).mean(axis=None)
        all_mse.append(mse_cur)

        # Postprocess the outputs
        confidence, box_idx, delta = infer(output)

        # use [x1, y1, x2, y2]
        bbox_centre, bbox_hw = Bbox(confidence, box_idx, delta)  # get the bounding box information we want
        box_coor = turn_to_boxes(bbox_centre, bbox_hw)  # turn to x1, y1, x2, y2
        box_coor, confidence = nms_leaky(confidence=confidence, box_coor=box_coor, iou_threshold=0.3, leaky_frac=0.0)
        # print(box_coor)
        # remove all those that don't meet confidence score
        box_coor_cur = []
        confidence_cur = []
        for i_conf, conf in enumerate(confidence):
            if conf >= 0.6:  # confidence which we
                confidence_cur.append(conf)
                box_coor_cur.append(box_coor[i_conf])
        box_coor = box_coor_cur.copy()
        confidence = confidence_cur.copy()

        # get iou of single boxes
        box_coor, confidence = box_in_box_iou(box_coor, confidence, 0.3)  # threshold is 50%

        # draw text
        img = cv2.putText(img, str(mse_cur), (10+1, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        img = cv2.putText(img, str(mse_cur), (10-1, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        img = cv2.putText(img, str(mse_cur), (10, 20+1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        img = cv2.putText(img, str(mse_cur), (10, 20-1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        img = cv2.putText(img, str(mse_cur), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # draw the bounding boxes on the image
        draw_bb_results(img, confidence, box_coor, imgTest_path_to_Save + f"/img_unit_{created_num_img}.png", False)

        created_num_img += 1

        if i_img % 25 == 0:
            print(f"We are at image number : {i_img} / {all_ImgPaths.shape[0]}")

    # get the mean squared error
    print("")
    print(f"Mean squared error of all unit test images : {np.mean(all_mse)}")
