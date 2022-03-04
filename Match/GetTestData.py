import pandas as pd
import numpy as np


def GetOneImageDataPoint(image_num):
    # Get data from csv to dict with imagenumber * pointnumber * [x,y,p]
    dict_all_point_in_one_image = {}
    trainBodyPointFromCSV = pd.read_csv("../GenerateDataFromImage/test_bodykeypoints.csv")
    trainBodyPointFromCSV = trainBodyPointFromCSV["0"]
    trainBodyPointRow = trainBodyPointFromCSV[image_num][2:-2]
    trainBodyPointRow = trainBodyPointRow.split("\n")
    for i in range(len(trainBodyPointRow)):
        listrow = trainBodyPointRow[i].split("[")[1].split(']')[0].split(" ")
        listrow_without_empty = list(filter(None, listrow))
        for numindex in range(len(listrow_without_empty)):
            listrow_without_empty[numindex] = eval(listrow_without_empty[numindex])
        dict_all_point_in_one_image[i] = listrow_without_empty
    return (dict_all_point_in_one_image)

def OnePointInOneImageFront(num, train_body_image):
    # Delete point 4/7 (hand) and 19-24 (foot) and make it to be relative axis location
    pointlist = []
    image_to_vector = train_body_image
    for i in range(len(image_to_vector)):
        if i == 4 or i == 7 or i == 19 or i == 20 or i == 21 or i == 22 or i == 23 or i == 24:
            continue
        int_image_to_vector = list(map(float, image_to_vector[i]))
        pointlist.append(int_image_to_vector)
    pointlist_np = np.array(pointlist)
    pointlist_np -= pointlist_np[0]
    onepointaxis = pointlist_np[num][0:2]
    return onepointaxis

def OnePointInOneImageSide(num, train_body_image):
    # Delete point 4/7 (hand) and 19-24 (foot) and make it to be relative axis location
    pointlist = []
    image_to_vector = train_body_image
    for i in range(len(image_to_vector)):
        if i == 0 or i == 1 or i == 5 or i == 6 or i == 7 or i == 16 or i == 18:
            int_image_to_vector = list(map(float, image_to_vector[i]))
            pointlist.append(int_image_to_vector)
        else:
            continue
    pointlist_np = np.array(pointlist)
    pointlist_np -= pointlist_np[0]
    onepointaxis = pointlist_np[num][0:2]
    return onepointaxis

def OnePointInOneImageSelectedArrayFront(indexofpoints, indexofimage):
    test_body_all_image = GetOneImageDataPoint(indexofimage-1)
    point_a = OnePointInOneImageFront(indexofpoints, test_body_all_image)
    list_a = list(point_a)
    roundlist = [round(i, 3) for i in list_a]
    return np.array(roundlist)

def OnePointInOneImageSelectedArraySide(indexofpoints, indexofimage):
    test_body_all_image = GetOneImageDataPoint(indexofimage-1)
    point_a = OnePointInOneImageSide(indexofpoints, test_body_all_image)
    list_a = list(point_a)
    roundlist = [round(i, 3) for i in list_a]
    return np.array(list_a)

#array2 = OnePointInOneImageSelectedArrayFront(12, 22)
#print(array2)
