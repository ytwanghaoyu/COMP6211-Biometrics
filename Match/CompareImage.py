import numpy as np
import ExtractPointData
import GetTestData


def euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def CompareImageFront(TestImageIndex):
    TestImageIndex=Check2122(TestImageIndex)
    CostList = []
    for TrainImageIndex in range(0, 88, 2):
        SingleImageCost = 0
        for PointIndex in range(1, 17):
            OnePointDataTrain = ExtractPointData.OnePointInOneImageFront(PointIndex, TrainImageIndex + 1)
            # print("TrainImageIndex",TrainImageIndex+1,"PointIndex",PointIndex,"PointPosition",OnePointDataTrain)
            OnePointDataTest = GetTestData.OnePointInOneImageSelectedArrayFront(PointIndex, TestImageIndex)
            # print("TrainImageIndex",TrainImageIndex+1,"PointIndex",PointIndex,"PointPosition",OnePointDataTest)
            CostOnePoint = euclidean(OnePointDataTest, OnePointDataTrain)
            SingleImageCost += CostOnePoint
        CostList.append(SingleImageCost)
    index = CostList.index(min(CostList))
    return index,CostList


def CompareImageSide(TestImageIndex):
    TestImageIndex=Check2122(TestImageIndex)
    CostList = []
    for TrainImageIndex in range(0, 88, 2):
        SingleImageCost = 0
        for PointIndex in range(1, 7):
            OnePointDataTrain = ExtractPointData.OnePointInOneImageSide(PointIndex, TrainImageIndex + 2)
            # print("TrainImageIndex",TrainImageIndex+1,"PointIndex",PointIndex,"PointPosition",OnePointDataTrain)
            OnePointDataTest = GetTestData.OnePointInOneImageSelectedArraySide(PointIndex, TestImageIndex)
            # print("TrainImageIndex",TrainImageIndex+1,"PointIndex",PointIndex,"PointPosition",OnePointDataTest)
            CostOnePoint = euclidean(OnePointDataTest, OnePointDataTrain)
            SingleImageCost += CostOnePoint
        CostList.append(SingleImageCost)
    index = CostList.index(min(CostList))
    return index,CostList


def ImageNameFix(num):
    if num <= 20:
        if num % 2 == 0:
            return num + 45
        else:
            return num + 47
    else:
        if num <= 21:
            return 88
        else:
            return 87


def Check2122(num):
    if num == 21:
        return 22
    else:
        if num ==22:
            return 21
        else :
            return num


def MatchResult(guess,real):
    if guess==real:
        return True
    else:
        return False



# 23
TFList=[]
CostList=[]
for i in range(1, 23):
    if i % 2 == 0:
        index,Cost = CompareImageFront(i)
        ImageIndex = index * 2 + 1
        TrueOrFalse = MatchResult(ImageIndex, ImageNameFix(i))
        print("test image index is", i, "  guess result is", ImageIndex, "  real result is", ImageNameFix(i), "  ",TrueOrFalse)
        TFList.append(TrueOrFalse)
        CostList.append(Cost)
    else:
        index,Cost = CompareImageSide(i)
        ImageIndex = index * 2 + 2
        TrueOrFalse = MatchResult(ImageIndex, ImageNameFix(i))
        print("test image index is", i, "  guess result is", ImageIndex, "  real result is", ImageNameFix(i), "  ",TrueOrFalse)
        TFList.append(TrueOrFalse)
        CostList.append(Cost)
CostArray=np.array(CostList)
np.savetxt("CostArray.csv",CostArray,delimiter=",")
print(TFList.count(True)/len(TFList)*100,"%")