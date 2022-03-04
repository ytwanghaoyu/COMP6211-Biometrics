from sklearn.neighbors import NearestNeighbors
import numpy as np
import ExtractPointData
import GetTestData
import heapq
from collections import Counter

def euclidean(x, y):
    return np.sqrt(np.sum((x - y)**2))


def FindCorrespondingImageByPoint(PointIndex,ImageIndex):
    OnePointDataTrain = ExtractPointData.OnePointInAllImage(PointIndex)
    OnePointDataTest= GetTestData.OnePointInOneImageSelectedArrayFront(PointIndex, ImageIndex - 1)
    distance=[]
    final_index_list=[]
    for num in OnePointDataTrain:
        distance.append(euclidean(num,OnePointDataTest))
    print(distance)
    min_num_index_list = heapq.nsmallest(3, distance)
    print(min_num_index_list)
    for i in range(3):
        final_index_list.append(distance.index(min_num_index_list[i]))
    print(final_index_list)
    return final_index_list

def FinalResultForOneImage(ImageIndex):
    finalresult=[]
    for PointIndex in range(1,17):
        result=FindCorrespondingImageByPoint(PointIndex,ImageIndex)
        for num in result:
            finalresult.append(num)
    dic_final_result_sorted=list(Counter(finalresult))
    return dic_final_result_sorted[0]

a=FindCorrespondingImageByPoint(1,10)

#for i in range(len(final_index_list)):
#    final_index_list[i] = final_index_list[i] * 2 + 1