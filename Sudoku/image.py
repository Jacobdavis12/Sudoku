import numpy as np
from PIL import Image
import io
import random

def generateGaussianKernel(sigma, size):
    xDistance = np.array((2*size+1)*[[i for i in range(-size, size+1)]])
    distance = -0.5 * (xDistance**2 + xDistance.T**2)
    kernel = np.exp(distance/sigma**2)
    return kernel/kernel.sum()


def display(pixles):
    imageFromPixles = Image.fromarray(np.uint8([pixles[i:i+28]*255 for i in range(0,784,28)]))
    imageFromPixles.show()

class sudokuImage():
    def __init__(self, imageData, Directory = 'images/'):
        #Inherit Module by delegated wrapper
        self._img = Image.open(io.BytesIO(imageData)).convert('L')
        self.directory = Directory

        #Initialise properties
        self.pixles = np.array([[self.getpixel((w, h)) for w in range(self.width)] for h in range(self.height)])
        self.values = np.empty((9,9))
        self.confidences = np.empty((9,9))

    #Inherit Module by delegated wrapper
    def __getattr__(self,key):
        return getattr(self._img,key)

    def transform(self, size, method, data, pixles):
        img = Image.fromarray(np.uint8(pixles)).transform(size, method, data)
        return np.array([[img.getpixel((w, h)) for w in range(img.width)] for h in range(img.height)])

    def getValues(self):
        outputValues = {}

        for row in range(9):
            for col in range(9):
                outputValues[str(row)+str(col)] = self.values[row][col]

        return outputValues

    def updatePixles(self):
        self.pixles = np.array([[self.getpixel((w, h)) for w in range(self.width)] for h in range(self.height)])

    def savePixles(self):
        self._img = Image.fromarray(np.uint8(self.pixles))

    def recognise(self):
        if verbose:
            self.doSave('greyscale.jpg')

        #Detect edges
        threshold = self.adaptiveThreshold(self.pixles)
        if verbose:
            self.doSave('Threshold.jpg', threshold)

        #Isolate sudoku
        corners = self.findCorners(threshold)

        #Format into 252x252 square
        warped = self.transform((252,252), Image.QUAD, [i for j in corners for i in j[::-1]], threshold)

        if verbose:
            self.doSave('warped.jpg', warped)

        #separate into grid
        self.grid = self.getCells(corners, warped)

        #clean grid and recognise
        cleanedRow = []
        for row in range(9):
            for col in range(9):
                if verbose:
                    self.doSave('dirtGrid/' + str(row) + str(col) + '.jpg', self.grid[row][col])
                self.grid[row][col] = self.removeAllButlargest(self.grid[row][col])
                if verbose:
                    self.doSave('grid/' + str(row) + str(col) + '.jpg', self.grid[row][col])
                self.values[row][col] = nt.recognise(self.grid[row][col]/255)
            cleanedRow.append(np.concatenate(self.grid[row], axis = 1))

        cleaned = np.concatenate(cleanedRow)
        if verbose:
            self.doSave('removed.jpg', cleaned)

        print(self.values)

    def removeAllButlargest(self, pixles):
        components = self.connectedComponents(pixles)

        #If on edges remove
        componentI = 0
        while componentI < len(components):
            remove = 0
            for pix in components[componentI]:
                if pix[0] == 0 or pix[0] == pixles.shape[0]-1 or pix[1] == 0 or pix[1] == pixles.shape[1]-1:
                    remove = 1
                    break
                if pix[0] == 1 or pix[0] == pixles.shape[0]-2 or pix[1] == 1 or pix[1] == pixles.shape[1]-2:
                    remove+=0.0625

                if pix[0] == 2 or pix[0] == pixles.shape[0]-3 or pix[1] == 2 or pix[1] == pixles.shape[1]-3:
                    remove+=0.03125

                if remove>=1:
                    break

            if remove>=1:
                del components[componentI]
            else:
                componentI += 1

        components = sorted(components, key = len)

        componentPixles = np.empty(pixles.shape)

        if len(components) != 0:
            for pix in components[-1]:
                componentPixles[pix[0]][pix[1]] = 255

        return componentPixles


    def convolve(self, kernel, pixles):
        kernel = np.flipud(np.fliplr(kernel))
        newPixles = np.zeros(pixles.shape)
        kernelHeight = kernel.shape[0] // 2
        kernelWidth = kernel.shape[1] // 2
        pixlesHeight = pixles.shape[0]
        pixlesWidth = pixles.shape[1]

        for row in range(0, pixlesHeight):
            for col in range(0, pixlesWidth):
                lowRow = row-kernelWidth
                upRow = row+kernelWidth+1
                lowCol = col-kernelHeight
                upCol = col+kernelHeight+1
                div = kernelWidth*kernelHeight

                if lowRow < 0:
                    div += lowRow
                    lowRow = 0
                if lowCol < 0:
                    div += lowCol
                    lowCol = 0
                if upRow >= pixlesHeight-1:
                    div -= upRow-pixlesHeight-1
                    upRow = pixlesHeight-1
                if upCol >= pixlesWidth-1:
                    div -= upCol-pixlesWidth-1
                    upCol = pixlesWidth-1

                #Map kernel to sub-matrix then sum to give central pixle
                newPixles[row][col] = (kernel[lowRow-row+kernelWidth: upRow-row+kernelWidth, lowCol-col+kernelHeight:upCol-col+kernelHeight]*pixles[lowRow:upRow, lowCol:upCol]).sum()/div*kernelWidth*kernelHeight

        return newPixles

    def adaptiveThreshold(self, pixles):
        size = 15
        kernel = -generateGaussianKernel(5, size)
        kernel[size][size] = kernel[size][size]+1

        localMean = self.convolve(kernel, pixles)
        if verbose:
            self.doSave('lm.jpg', 255*(localMean-localMean.min())/(localMean-localMean.min()).max())

        #threshvalue = (90/255*(localMean-localMean.min()).max()+localMean.min())/localMean.mean()
        localMean[localMean>-3] = 0
        localMean[localMean!=0] = 255

        return localMean

    def countOnEdge(self, component, corners):
        complete = np.full(self.pixles.shape, 255)
        count = 0

        for a, b in [[0,1],[1,2],[2,3],[3,0]]:#left, bottom, right top
            edgeImage = np.zeros(self.pixles.shape)
            dx = (corners[b][0] - corners[a][0])
            dy = (corners[b][1] - corners[a][1])
            lengthXY = (dx**2+dy**2)**(1/2)
            if lengthXY == 0:
                count += 1
            else:
                for pixle in component:
                    if abs((pixle[0] - corners[a][0])*dx - (pixle[1] - corners[a][1])*dy)/lengthXY <= 3:#On same line
                        count+=1
                        edgeImage[pixle[0]][pixle[1]] = 255

                    if complete[pixle[0]][pixle[1]] > abs((pixle[0] - corners[a][0])*dx - (pixle[1] - corners[a][1])*dy)/lengthXY:
                        complete[pixle[0]][pixle[1]] = abs((pixle[0] - corners[a][0])*dx - (pixle[1] - corners[a][1])*dy)/lengthXY

        if verbose:
            self.doSave('edge' + str(a) + str(b) + '.jpg', edgeImage)
            self.doSave('complete.jpg', complete)

        return count

    def findCorners(self, pixles):
        components = self.connectedComponents(pixles)
        components = sorted(components, key = len)
        possibleSudokus = components[-4:]
        possibleCorners = [[min(sudoku, key = lambda x:x[0]+x[1]), max(sudoku, key = lambda x:x[0]-x[1]), max(sudoku, key = lambda x:x[0]+x[1]), min(sudoku, key = lambda x:x[0]-x[1])] for sudoku in possibleSudokus]#lt lb rb rt
        
        #Find most Sudoku-like
        biggest = [0]
        for i in range(0, 4):
            self.saveArray('currentComponent.jpg', possibleSudokus[i], pixles)
            size = self.countOnEdge(possibleSudokus[i], possibleCorners[i])*(i+2)/5
            
            if size >= biggest[0]:
                biggest = [size, i]

        if verbose:
            for i in range(1, 5):
                self.saveArray('component' + str(i) + '.jpg', components[-i], pixles)
        
        corners = possibleCorners[biggest[1]]
        if verbose:
            self.saveArray('corners.jpg', corners, pixles)

        if verbose:
            self.doSave('pix.jpg')

        return corners

    def removeComponent(self, component, pixles):
        for i in component:
            pixles[i[0]][i[1]] = 0

        return pixles

    def connectedComponents(self, pixles):
        self.pixlesTemp = np.copy(pixles)
        components = []
        for row in range(pixles.shape[0]):
            for col in range(pixles.shape[1]):
                if self.pixlesTemp[row][col] == 255:
                    component, overflows = self.floodFill(row, col)

                    while len(overflows) > 0:
                        moreOverflows = []
                        print(len(overflows))
                        for row, col in overflows:
                            for r in range(-1,2):
                                for c in range(-1,2):
                                    if r != 0 or c != 0 :
                                        componentIn, overflowIn = self.floodFill(row+r, col+c)
                                        component += componentIn
                                        moreOverflows += overflowIn

                        overflows = [i for i in moreOverflows]

                        print('uo', len(moreOverflows))

                    components.append(component)

        return components

    def floodFill(self, row, col, size = 0):
        if row >= 0 and row < self.pixlesTemp.shape[0] and col >= 0 and col < self.pixlesTemp.shape[1] and self.pixlesTemp[row][col] == 255:
            if size > 900:#overflow
                return [], [[row,col]]
            else:
                self.pixlesTemp[row][col] = 0
                component = [[row,col]] 
                overflow = []
                for r in range(-1,2):
                    for c in range(-1,2):
                        if r != 0 or c != 0:
                            componentIn, overflowIn = self.floodFill(row+r, col+c, size+1)
                            component += componentIn
                            overflow += overflowIn

                return component, overflow
        else:
            return [], []

    def getCells(self, corners, pixles):
        grid = []
        cellWidth = 28
        cellHeight = 28

        yPosition = 0
        for y in range(9):
            xPosition = 0
            grid.append([])
            for x in range(9):
                grid[-1].append(pixles[yPosition:yPosition+28, xPosition:xPosition+28])#warning
                xPosition += 28
            yPosition += 28

        return grid

    #Test Saves
    def doSave(self, fileName, pixles=''):
        if verbose:
            print(fileName)
        if pixles == '':
            imageFromPixles = Image.fromarray(np.uint8(self.pixles))
        else:
            imageFromPixles = Image.fromarray(np.uint8(pixles))#.convert('RGB')
        imageFromPixles.save(self.directory + fileName)

    def saveArray(self, fileName, array, pixles):
        if verbose:
            print(fileName)
        validPixles = np.empty(pixles.shape)
        for i in array:
            validPixles[i[0]][i[1]] = 255

        self.doSave(fileName, validPixles)

class network:
    def __init__(self, fileName = False):
        self.loadNet()
        self.dimensions = [784] + [len(i) for i in self.weights]

    def loadNet(self, filename = 'network.npz'):
        data = np.load(filename, allow_pickle=True)
        self.weights = data['weight']
        self.biases = data['bias']

    def run(self, activation):
        activations = [activation]
        for weight, bias in zip(self.weights, self.biases):
            activation = self.sigmoid(np.dot(weight, activation)+bias)
            activations.append(activation)
        return activations

    def sigmoid(self, matrix):
        return 1/(1+np.exp(-matrix))

    def recognise(self, image):
        activation = self.run(image.flatten())[-1]
        #display(image.flatten())
        return np.argmax(activation)

    def getCost(self, image, actual):
        wanted = np.zeros(10)
        wanted[int(actual)] = 1
        inputImage = image.flatten()
        inputImage[np.isnan(image.flatten())] = 0
        activation = self.run(inputImage)[-1]
        c = self.cost(activation, wanted)
        if np.isnan(c):
            c = 0
        return c

    def cost(self, activation, wanted):
        return sum((activation - wanted)**2)/self.dimensions[-1]/2

class testImage(sudokuImage):
    def recognise(self, tn):
        self.testName = tn
        if verbose:
            self.doSave('greyscale.jpg')

        #Detect edges
        for size in range(11, 22):
            for sigma in range(7, 11):
                self.adaptiveThreshold(self.pixles, size, sigma)

    def adaptiveThreshold(self, pixles, size, sigma):
        kernel = -generateGaussianKernel(sigma, size)
        kernel[size][size] = kernel[size][size]+1

        localMean = self.convolve(kernel, pixles)
        self.doSave(self.testName + '_' + str(size) + '_' + str(sigma) + '.jpg', 255*(localMean-localMean.min())/(localMean-localMean.min()).max())

class variImage(sudokuImage):
    def __init__(self, imageData, variables, sudokuValues, Directory = 'images/'):
        super(variImage, self).__init__(imageData, Directory)
        self.edgeRemove1=variables[0]
        self.edgeRemove2=variables[1]
        self.edgeRemove3=variables[2]
        self.thresholdValue=variables[3]
        self.thresholdSize=variables[4]
        self.thresholdSigma=variables[5]

    def removeAllButlargest(self, pixles):
        components = self.connectedComponents(pixles)

        #If on edges remove
        componentI = 0
        while componentI < len(components):
            remove = 0
            for pix in components[componentI]:
                if pix[0] == 0 or pix[0] == pixles.shape[0]-1 or pix[1] == 0 or pix[1] == pixles.shape[1]-1:
                    remove = self.edgeRemove1#1
                    break
                if pix[0] == 1 or pix[0] == pixles.shape[0]-2 or pix[1] == 1 or pix[1] == pixles.shape[1]-2:
                    remove+=self.edgeRemove2#0.0625

                if pix[0] == 2 or pix[0] == pixles.shape[0]-3 or pix[1] == 2 or pix[1] == pixles.shape[1]-3:
                    remove+=self.edgeRemove3#0.03125

                if remove>=1:
                    break

            if remove>=1:
                del components[componentI]
            else:
                componentI += 1

        components = sorted(components, key = len)

        componentPixles = np.empty(pixles.shape)

        if len(components) != 0:
            for pix in components[-1]:
                componentPixles[pix[0]][pix[1]] = 255

        return componentPixles

    def adaptiveThreshold(self, pixles):
        size = self.thresholdSize#11
        kernel = -generateGaussianKernel(self.thresholdSigma, size)#4
        kernel[size][size] = kernel[size][size]+1

        localMean = self.convolve(kernel, pixles)
        if verbose:
            self.doSave('lm.jpg', 255*(localMean-localMean.min())/(localMean-localMean.min()).max())

        localMean[localMean>self.thresholdValue] = 0#-1.5
        localMean[localMean!=0] = 255

        return localMean

    def recognise(self):
        if verbose:
            self.doSave('greyscale.jpg')

        #Detect edges
        threshold = self.adaptiveThreshold(self.pixles)
        if verbose:
            self.doSave('Threshold.jpg', threshold)

        #Isolate sudoku
        corners = self.findCorners(threshold)

        #Format into 252x252 square
        warped = self.transform((252,252), Image.QUAD, [i for j in corners for i in j[::-1]], threshold)

        if verbose:
            self.doSave('warped.jpg', warped)

        #separate into grid
        self.grid = self.getCells(corners, warped)

        #clean grid and recognise# and getCost
        self.netCost = 0
        cleanedRow = []
        for row in range(9):
            for col in range(9):
                if verbose:
                    self.doSave('dirtGrid/' + str(row) + str(col) + '.jpg', self.grid[row][col])
                self.grid[row][col] = self.removeAllButlargest(self.grid[row][col])
                if verbose:
                    self.doSave('grid/' + str(row) + str(col) + '.jpg', self.grid[row][col])
                self.netCost += nt.getCost(self.grid[row][col]/255, sudokuValues[row][col])
            cleanedRow.append(np.concatenate(self.grid[row], axis = 1))

        self.pixles = np.concatenate(cleanedRow)
        if verbose:
            self.doSave('removed.jpg')

def vary(variance=100):
    variables = [0,0,0,0,0,0]
    if random.randint(0, 100)>variance*0.8:
        variables[0] = edgeRemove1 + random.randrange(-1,1)/5
    else:
        variables[0] = edgeRemove1
    if random.randint(0, 100)>variance*0.8:
        variables[1] = edgeRemove2 + random.randrange(-1,1)/10
    else:
        variables[1] = edgeRemove2
    if random.randint(0, 100)>variance*0.8:
        variables[2] = edgeRemove3 + random.randrange(-1,1)/50
    else:
        variables[2] = edgeRemove3
    if random.randint(0, 100)>variance*0.2:
        variables[3] = thresholdValue + random.randrange(-1,1)/4
    else:
        variables[3] = thresholdValue
    if random.randint(0, 100)>variance*0.3:
        variables[4] = thresholdSize + random.randint(-1,2)
    else:
        variables[4] = thresholdSize
    if random.randint(0, 100)>variance*0.25:
        variables[5] = thresholdSigma + random.randint(-2,3)
    else:
        variables[5] = thresholdSigma

    return variables

nt = network()



##singular

#verbose = True
#fileName = 'test15'
#with open(fileName + '.jpg', 'rb') as f:
#    testData = f.read()

#print(fileName)

#sudoku = sudokuImage(testData)
#print(sudoku.pixles.shape)
#sudoku.recognise()
#sudoku.doSave('fr.jpg')



#bulk
verbose = True
for i in range(17):
     with open('test' + str(i) +'.jpg', 'rb') as f:
        testdata = f.read()
     sudoku = sudokuImage(testdata, 'images/bulk/' + str(i))
     print('test' + str(i))
     sudoku.recognise()
     del sudoku



##test
#verbose = False
#for i in range(17):
#    with open('test' + str(i) +'.jpg', 'rb') as f:
#        testData = f.read()
#    print('test' + str(i))
#
#    sudoku = testImage(testData)
#    sudoku.recognise('test' + str(i))



##variance
#edgeRemove1=0.8
#edgeRemove2=-0.1375
#edgeRemove3=0.01125
#thresholdValue=-2.25
#thresholdSize=11
#thresholdSigma=4
#nt = network('mnist0s.npz')
#verbose = False
#netcosts = {}

##for name in [0,1,11,5,7,3,14]:
##        with open('test' + str(name) + '.jpg', 'rb') as f:
##            testData = f.read()

##        with open('test' + str(name) + '.dat', 'r') as f:
##            datData = f.read()

##        sudokuValues = [i.split(' ') for i in datData.split('\n')[2:11]]
##        sudoku = variImage(testData, [edgeRemove1, edgeRemove2, edgeRemove3, thresholdValue, thresholdSize, thresholdSigma], sudokuValues)
##        sudoku.recognise()
##        netcosts[name] = sudoku.netCost
##        del sudoku

#oldNetCosts = {0: 3.2067628608927055, 1: 3.54680734590637, 3: 3.1840925215609164, 5: 3.8721358455020525, 7: 3.7042401285306643, 11: 3.833493281511013, 14: 4.325441664804166}#netcosts

#while True:
#    variables = vary()
#    for name in [0,1,11,5,7,3,14]:
#        with open('test' + str(name) + '.jpg', 'rb') as f:
#            testData = f.read()

#        with open('test' + str(name) + '.dat', 'r') as f:
#            datData = f.read()

#        sudokuValues = [i.split(' ') for i in datData.split('\n')[2:11]]
#        sudoku = variImage(testData, variables, sudokuValues)
#        sudoku.recognise()
#        netcosts[name] = sudoku.netCost
#        del sudoku

#    totalCost = 0
#    for i in netcosts:
#        totalCost += (oldNetCosts[i] - netcosts[i])*abs(oldNetCosts[i] - netcosts[i])

#    if totalCost > 0:
#        edgeRemove1 = variables[0]
#        edgeRemove2 = variables[1]
#        edgeRemove3 = variables[2]
#        thresholdValue = variables[3]
#        thresholdSize = variables[4]
#        thresholdSigma = variables[5]
#        oldNetCosts = netcosts
#    print(oldNetCosts)
#    print(variables)

