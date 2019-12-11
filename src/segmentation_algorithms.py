import numpy as np
from matplotlib import pyplot as plt
import math
import multiprocessing as mp
from skimage import data
from skimage import io
from scipy import ndimage

class Pixel:
    def __init__(self, color):
        self.color = color
        self.colorCost = {}

#Function used to calculate how
def data_cost(pixel, mainColor, mainColors):
    sum = 0
    for color in mainColors:
        sum += pow(pow(pixel.color[0] - color[0],2) + pow(pixel.color[1] - color[1],2) + pow(pixel.color[2] - color[2],2),0.5)
    colorSum = pow(pow(pixel.color[0] - mainColor[0],2) + pow(pixel.color[1] - mainColor[1],2) + pow(pixel.color[2] - mainColor[2],2),0.5)
    try:
        return -math.log(1 - (colorSum/sum))
    except Exception:
        return 0


#The difference in smooth cost if a pixel is changed to another color. (Number of similar neighbour equals the difference)
def smooth_cost_expansion(pixels, pos, newCol):
    epsilon = 0.5
    sumSmoothCost = 0
    try:
        if set(newCol) == set(pixels[pos[0] + 1][pos[1]].label):
            sumSmoothCost += epsilon
    except:
        pass
    try:
        if set(newCol) == set(pixels[pos[0]][pos[1] + 1].label):
            sumSmoothCost += epsilon
    except Exception:
        pass
    try:
        if set(newCol) == set(pixels[pos[0] - 1][pos[1]].label):
            sumSmoothCost += epsilon
    except:
        pass
    try:
        if set(newCol) == set(pixels[pos[0]][pos[1] - 1].label):
            sumSmoothCost += epsilon
    except Exception:
        pass
    return sumSmoothCost

#Initial labeling of all pixels according to the main colors
def initial_labeling(pixels, allColors):
    for row in pixels:
        for pixel in row:
            dataCost = 100000
            for mainColor in allColors:
                tmpCost = data_cost(pixel, mainColor, allColors)
                pixel.colorCost[str(mainColor)] = tmpCost
                if tmpCost <= dataCost:
                    pixel.label = mainColor
                    pixel.cost = tmpCost
                    dataCost = tmpCost
    return pixels

#Makes every pixel surrounded by the same color into that same color. Smooths out the image to lower the smooth cost of a cluster.
def smooth_out(pixels):
    for i in range(0, len(pixels)):
        for j in range(0, len(pixels[0])):
            surroundingColor = None
            try:
                if surroundingColor is None:
                    surroundingColor = (pixels[i + 1][j].label)
            except:
                pass
            try:
                if surroundingColor is None:
                    surroundingColor = (pixels[i][j + 1].label)
                if surroundingColor != (pixels[i][j + 1].label):
                    continue
            except Exception:
                pass
            try:
                if surroundingColor is None:
                    surroundingColor = (pixels[i - 1][j].label)
                if surroundingColor != (pixels[i - 1][j].label):
                    continue
            except:
                pass
            try:
                if surroundingColor != (pixels[i][j - 1].label):
                    continue
            except Exception:
                pass
            pixels[i][j].label = surroundingColor

#Finds all the clusters, where all points of each color is a cluster
def find_clusters(pixels, allColors):
    clusters = []
    clusterMean = []
    for color in allColors:
        clusteredPixel = []
        clustMean = [0,0]
        for i in range(0, len(pixels)):
            for j in range(0, len(pixels[0])):
                if set(pixels[i][j].label) == set(color):
                    clusteredPixel.append([i,j])
                    clustMean[0] += i
                    clustMean[1] += j
        if len(clusteredPixel) > 0:
            clusters.append(clusteredPixel)
            clusterMean.append([clustMean[0]/len(clusteredPixel),clustMean[1]/len(clusteredPixel)])
    return [clusters, clusterMean]

#Does the fast-alpha expansion. Looks at all the clusters for a specific color (alpha),
# looks at if the energy function is better if that cluster is changed to the color of alpha.
def fast_alpha_expansion(pixels, clusters, mainColor, allColors, clusterChanges):
    changeDuringIteration = False
    for i in range(len(clusters)):
        if set(pixels[clusters[i][0][0]][clusters[i][0][1]].label) != set(mainColor):
            mainColorCost = 0
            for pos in clusters[i]:
                mainColorCost += pixels[pos[0]][pos[1]].cost - pixels[pos[0]][pos[1]].colorCost[str(mainColor)]
                mainColorCost += smooth_cost_expansion(pixels, [pos[0],pos[1]], mainColor)

            if mainColorCost > clusterChanges[i][0]:
                changeDuringIteration = True
                clusterChanges[i][0] = mainColorCost
                clusterChanges[i][1] = mainColor
    return [changeDuringIteration, clusterChanges]

#Gets the main colors. Number of bins per channel is changeable by changing the 'binNum' value
def get_colors(img):
    binsSum = []
    binsAmount = []
    binNum = 3
    for i in range(0,binNum * binNum * binNum):
        binsSum.append([0, 0, 0])
        binsAmount.append(0)
    binTotal = 0
    for row in img:
        for pixel in row:
            pos = 0
            pos += int((pixel[0]/math.ceil(256/binNum))) *  (binNum*binNum)
            pos += int((pixel[1]/math.ceil(256/binNum))) *  binNum
            pos += int((pixel[2]/math.ceil(256/binNum)))
            binsAmount[pos] += 1
            binsSum[pos][0] += pixel[0]
            binsSum[pos][1] += pixel[1]
            binsSum[pos][2] += pixel[2]
            binTotal += 1
    bins = []
    for bin in range(0,binNum * binNum * binNum):
        if binsAmount[bin]/binTotal > 0.05:
            bins.append([int(binsSum[bin][0]/binsAmount[bin]), int(binsSum[bin][1]/binsAmount[bin]), int(binsSum[bin][2]/binsAmount[bin])])
    return bins



def makeThing(img):
    i = io.imread(img)
    iar = np.asarray(i)
    plt.imshow(np.array(iar), interpolation='none')
    plt.show()
    mainColors = get_colors(iar)
    pixels = [ [ None for y in range(iar.shape[1] - 1) ] for x in range(iar.shape[0] - 1) ]
    for z in range(0, iar.shape[0] - 1):
        for j in range(0, iar.shape[1] - 1):
            pixels[z][j] = Pixel(iar[z][j])
    initial_labeling(pixels, mainColors)
    nPixel = [ [ None for y in range(iar.shape[1] - 1) ] for x in range(iar.shape[0] - 1) ]
    for z in range(0, iar.shape[0] - 1):
        for j in range(0, iar.shape[1] - 1):
            nPixel[z][j] = pixels[z][j].label
    plt.imshow(np.array(nPixel), interpolation='none')
    plt.show()
    for mainColor in mainColors:
        bPixel = [[None for y in range(iar.shape[1] - 1)] for x in range(iar.shape[0] - 1)]
        for z in range(0, iar.shape[0] - 1):
            for j in range(0, iar.shape[1] - 1):
                if set(nPixel[z][j]) == set(mainColor):
                    bPixel[z][j] = 1
                else:
                    bPixel[z][j] = 0
        bPixel = ndimage.binary_closing(bPixel)
        for z in range(0, iar.shape[0] - 1):
            for j in range(0, iar.shape[1] - 1):
                if bPixel[z][j] == 1:
                    nPixel[z][j] = mainColor
                    pixels[z][j].label = mainColor
    plt.imshow(np.array(nPixel), interpolation='none')
    plt.show()
    redo = True
    times = 0
    while redo and times < 6:
        times += 1
        smooth_out(pixels)
        clusterChanges = []
        clusters, means = find_clusters(pixels, mainColors)
        if len(clusters) <= 2:
            break
        for i in range(len(clusters)):
            clusterChanges.append([0.0, None, i])
        redo = False
        for mainColor in mainColors:
            change = fast_alpha_expansion(pixels, clusters, mainColor, mainColors, clusterChanges)
            print(change)
            if change[0]:
                redo = True
                clusterChanges = change[1]
        clusterChanges.sort()
        usedColors = []
        for i in range(len(clusters)):
            col = clusterChanges[i][1]
            index = clusterChanges[i][2]
            oldCol = pixels[clusters[index][0][0]][clusters[index][0][1]].label
            if col != None and oldCol not in usedColors:
                usedColors.append(col)
                for pos in clusters[index]:
                    pixels[pos[0]][pos[1]].label = col
                    pixels[pos[0]][pos[1]].cost = pixels[pos[0]][pos[1]].colorCost[str(col)]
        nPixel = [[None for y in range(iar.shape[1] - 1)] for x in range(iar.shape[0] - 1)]
        for z in range(0, iar.shape[0] - 1):
            for j in range(0, iar.shape[1] - 1):
                nPixel[z][j] = pixels[z][j].label
        print("CHANGE")
        for mainColor in mainColors:
            bPixel = [[None for y in range(iar.shape[1] - 1)] for x in range(iar.shape[0] - 1)]
            for z in range(0, iar.shape[0] - 1):
                for j in range(0, iar.shape[1] - 1):
                    if set(nPixel[z][j]) == set(mainColor):
                        bPixel[z][j] = 1
                    else:
                        bPixel[z][j] = 0
            bPixel = ndimage.binary_closing(bPixel)
            for z in range(0, iar.shape[0] - 1):
                for j in range(0, iar.shape[1] - 1):
                    if bPixel[z][j] == 1:
                        nPixel[z][j] = mainColor
                        pixels[z][j].label = mainColor
        plt.imshow(np.array(nPixel), interpolation='none')
        plt.show()
        if len(mainColors) == 2:
            smooth_out(pixels)
            break
    for i in clusters:
        nPixel = [[None for y in range(iar.shape[1] - 1)] for x in range(iar.shape[0] - 1)]
        for z in range(0, iar.shape[0] - 1):
            for j in range(0, iar.shape[1] - 1):
                if set(pixels[i[0][0]][i[0][1]].label) == set(pixels[z][j].label):
                    nPixel[z][j] = pixels[z][j].label
                else:
                    nPixel[z][j] = [0,0,0]
        plt.imshow(np.array(nPixel), interpolation='none')
        plt.show()


    finalArray = [[None for y in range(iar.shape[1] - 1)] for x in range(iar.shape[0] - 1)]
    for z in range(0, iar.shape[0] - 1):
        for j in range(0, iar.shape[1] - 1):
            finalArray[z][j] = mainColors.index(pixels[z][j].label)
    sums = []
    for i in range(len(mainColors)):
        sums.append([[0,0,0],0])
    for z in range(0, iar.shape[0] - 1):
        for j in range(0, iar.shape[1] - 1):
            index = finalArray[z][j]
            sums[index][0][0] += pixels[z][j].color[0]
            sums[index][0][1] += pixels[z][j].color[1]
            sums[index][0][2] += pixels[z][j].color[2]
            sums[index][1] += 1
    means = []
    for i in range(len(sums)):
        if sums[i][1] != 0:
            means.append([sums[i][0][0]/sums[i][1],sums[i][0][1]/sums[i][1],sums[i][0][2]/sums[i][1]])
        else:
            means.append([-1,-1,-1])

    sumVariance = []
    sumSkew = []
    for i in range(len(mainColors)):
        sumVariance.append([0, 0, 0])
        sumSkew.append([0, 0, 0])
    for z in range(0, iar.shape[0] - 1):
        for j in range(0, iar.shape[1] - 1):
            index = finalArray[z][j]
            sumVariance[index][0] += pow(pixels[z][j].color[0] - means[index][0],2)
            sumVariance[index][1] += pow(pixels[z][j].color[1] - means[index][1],2)
            sumVariance[index][2] += pow(pixels[z][j].color[2] - means[index][2],2)
            sumSkew[index][0] += pow(pixels[z][j].color[0] - means[index][0], 3)
            sumSkew[index][1] += pow(pixels[z][j].color[1] - means[index][1], 3)
            sumSkew[index][2] += pow(pixels[z][j].color[2] - means[index][2], 3)

    variance = []
    skew = []
    for i in range(len(sumVariance)):
        if sums[i][1] != 0:
            variance.append([pow(sumVariance[i][0] / sums[i][1],0.5), pow(sumVariance[i][1] / sums[i][1],0.5), pow(sumVariance[i][2] / sums[i][1],0.5)])
            if sumSkew[i][0] < 0 or sumSkew[i][1] < 0 or sumSkew[i][2] < 0:
                skew.append([pow(abs(sumSkew[i][0]) / sums[i][1], 1 / 3), pow(abs(sumSkew[i][0]) / sums[i][1], 1 / 3), pow(abs(sumSkew[i][0]) / sums[i][1], 1 / 3)])
            else:
                skew.append([pow(sumSkew[i][0] / sums[i][1],1/3), pow(sumSkew[i][1] / sums[i][1],1/3), pow(sumSkew[i][2] / sums[i][1],1/3)])
        else:
            variance.append([-1, -1, -1])
            skew.append([-1, -1, -1])

    segmentInfo = []
    for i in range(len(mainColors)):
        if sums[i][1] != 0:
            segmentInfo.append([means[i], variance[i], skew[i]])
    return segmentInfo