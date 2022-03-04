# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import pandas as pd

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../../x64/Release;' +  dir_path + '/../../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="12_test.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../../models/"
    params["face"] = True
    params["hand"] = False

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Define Path and List
    trainImagePath=os.listdir('G:/openpose-master/openpose-master/build_CPU/examples/tutorial_api_python/trainimage')
    trainImagePath.sort(key= lambda x:int(x[:-4]))
    bodykeypoints, bodykeypoints, facekeypoints, lefthandkeypoints, righthandkeypoints = [],[],[],[],[]
    index=1

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Loop of process image
    for item in trainImagePath:
        print(item)
        args[0].image_path = '../trainimage/' + item
        print('Precessing image' + str(args[0].image_path) + '...')

        # Process Image
        datum = op.Datum()
        imageToProcess = cv2.imread(args[0].image_path)
        inputimage = cv2.resize(imageToProcess, (560, 420), interpolation=cv2.INTER_CUBIC)
        datum.cvInputData = inputimage
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # Display Image
        bodykeypoints.append(str(datum.poseKeypoints))
        facekeypoints.append(str(datum.faceKeypoints))
        #lefthandkeypoints.append(str(datum.handKeypoints[0]))
        #righthandkeypoints.append(str(datum.handKeypoints[1]))
        #cv2.imshow('OutputImageTrain/imageWithKeyPoint'+str(args[0].image_path), datum.cvOutputData)
        cv2.waitKey(0)
        cv2.imwrite('../OutputImageTrain/Train'+str(index)+'.jpg', datum.cvOutputData)
        index+=1


    dfbodykeypoints=pd.DataFrame(bodykeypoints)
    dffacekeypoints=pd.DataFrame(facekeypoints)
    #dflefthandkeypoints=pd.DataFrame(lefthandkeypoints)
    #dfrighthandkeypoints=pd.DataFrame(righthandkeypoints)
    dfbodykeypoints.to_csv('train_bodykeypoints.csv', index=False)
    dffacekeypoints.to_csv('train_facekeypoints.csv', index=False)
    #dflefthandkeypoints.to_csv('train_lefthandkeypoints.csv', index=False)
    #dfrighthandkeypoints.to_csv('train_righthandkeypoints.csv', index=False)


except Exception as e:
    print(e)
    sys.exit(-1)
