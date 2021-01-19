import numpy as np
from PIL import Image
import io

class sudokuImage():
    def __init__(self, imageData):
        #Inherit Module by delegated wrapper
        self._img = Image.open(io.BytesIO(imageData)).convert('L')

        #Initialise properties
        self.pixlesHeight = self.height
        self.pixlesWidth = self.width

        self.pixles = np.array([[self.getpixel((w, h)) for w in range(self.pixlesWidth)] for h in range(self.pixlesHeight)])
        self.values = np.empty((9,9))

        #Initialise kernels
        self.gaussianKernel = np.array([[2,4,5,4,2],[4,9,12,9,4],[5,12,15,12,5],[2,4,5,4,2],[4,9,12,9,4]])/159#generateGaussianKernel(1,2)#
        self.xKernel = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.yKernel = np.array([[1, 2, 1],
                                 [0, 0, 0],
                                 [-1, -2, -1]])

    #Inherit Module by delegated wrapper
    def __getattr__(self,key):
        return getattr(self._img,key)

    def transform(self, size, method, data):
        self._img = self._img.transform(size, method, data)

    def getValues(self):
        outputValues = {}

        for row in range(9):
            for col in range(9):
                outputValues[str(row)+str(col)] = self.values[row][col]

        return outputValues

    def updatePixles(self):
        self.pixlesHeight = self.height
        self.pixlesWidth = self.width

        self.pixles = np.array([[self.getpixel((w, h)) for w in range(self.pixlesWidth)] for h in range(self.pixlesHeight)])

    def savePixles(self):
        self._img = Image.fromarray(np.uint8(self.pixles))

    def recognise(self):
        if verbose:
            self.doSave('greyscale.jpg')

        #Detect edges
        #self.pixles = self.canny(self.pixles)
        self.pixles = self.adaptiveThreshold(self.convolve(self.gaussianKernel, self.pixles))
        if verbose:
            self.doSave('Threshold.jpg', self.pixles)

        #Isolate sudoku
        corners = self.findCorners(self.pixles)

        #Format into 9*9 28 pixle cells
        self.savePixles()
        self.transform((252,252), Image.QUAD, [i for j in corners for i in j[::-1]])
        self.updatePixles()
        if verbose:
            self.save('warped.jpg')
        self.save('warped/n' + imageName)

        #Seperate

        self.grid = self.getCells(corners)

    def canny(self, pixles):
        #Apply gaussian blur
        gaussian = self.convolve(self.gaussianKernel, pixles)

        if verbose:
            self.doSave('gaussian.jpg', gaussian)

        #Obtain sobel operators
        Gx = self.convolve(self.xKernel, gaussian)
        Gy = self.convolve(self.yKernel, gaussian)

        gradient = np.hypot(Gx, Gy)
        theta = np.arctan2(Gy, Gx)
        theta[theta<0] += np.pi

        if verbose:
            self.doSave('gradient.jpg', gradient)
            self.doSave('theta.jpg', (255/np.pi)*theta)
            self.doSave('gx.jpg', 127.5+Gx/8)
            self.doSave('gy.jpg', 127.5+Gy/8)
            self.sTheta(theta)

        #Thin edge
        supPix = self.suppress(gradient, theta)
        if verbose:
            self.doSave('suppress.jpg', supPix)

        supPix[supPix < 50] =0
        supPix[supPix >= 50] = 255
        if verbose:
            self.doSave('Threshold.jpg', supPix)

        return supPix

    def convolve(self, kernel, pixles):
        kernel = np.flipud(np.fliplr(kernel))
        newPixles = np.zeros(pixles.shape)
        kernelHeight = kernel.shape[0] // 2
        kernelWidth = kernel.shape[1] // 2
        pixlesHeight = pixles.shape[0]
        pixlesWidth = pixles.shape[1]

        for row in range(kernelHeight, pixlesHeight-kernelHeight):
            for col in range(kernelWidth, pixlesWidth-kernelWidth):
                #Map kernel to sub-matrix then sum to give central pixle
                newPixles[row][col] = (kernel*pixles[row-kernelWidth:row+kernelWidth+1, col-kernelHeight:col+kernelHeight+1]).sum()

        return newPixles[kernelHeight:pixlesHeight-kernelHeight, kernelWidth:pixlesWidth-kernelWidth]

    def suppress(self, gradient, theta, weight = 1):
        supressedPixles = np.zeros(gradient.shape)
        for row in range(1, theta.shape[0]-1):
            for col in range(1, theta.shape[1]-1):
                if theta[row][col] < np.pi/8 or theta[row][col] >= 7*np.pi/8:#|
                    if gradient[row][col] >= weight*max(gradient[row][col+1], gradient[row][col-1]):
                        supressedPixles[row][col] = gradient[row][col]
                elif theta[row][col] < 3*np.pi/8:#/
                    if gradient[row][col] >= weight*max(gradient[row-1][col-1], gradient[row+1][col+1]):
                        supressedPixles[row][col] = gradient[row][col]
                elif theta[row][col] < 5*np.pi/8:#-
                    if gradient[row][col] >= weight*max(gradient[row+1][col], gradient[row-1][col]):
                        supressedPixles[row][col] = gradient[row][col]
                elif theta[row][col] < 7*np.pi/8:#\
                    if gradient[row][col] >= weight*max(gradient[row-1][col+1], gradient[row+1][col-1]):
                        supressedPixles[row][col] = gradient[row][col]
                else:
                    print(theta[row][col])

        return supressedPixles

    def adaptiveThreshold(self, pixles, size=5):
        thresholdPixles = np.zeros(pixles.shape)
        d = 2*size+1
        for row in range(pixles.shape[0]):
            rowLow = row-size
            rowUp = row+size+1
            if rowLow < 0:
                di = (2*size+1+rowLow)
                rowLow = 0
            elif rowUp >= pixles.shape[0]:
                di = (2*size+1+pixles.shape[0]-rowUp+1)
                rowLow = pixles.shape[0]-1
            else:
                di = d

            for col in range(pixles.shape[1]):
                colLow = col-size
                colUp = col+size+1
                if colLow < 0:
                    div = di * (2*size+1+colLow)
                    colLow = 0
                elif colUp >= pixles.shape[1]:
                    div = di * (2*size+1+pixles.shape[1]-colUp+1)
                    colLow = pixles.shape[1]-1
                else:
                    div = di * d

                if pixles[row][col] <= (pixles[rowLow:rowUp, colLow:colUp]).sum()/div-0.9:
                    thresholdPixles[row][col] = 255

        return thresholdPixles
                        
    def sTheta(self, theta):
        thetaPixles = np.empty(theta.shape)
        for row in range(1, theta.shape[0]-1):
            for col in range(1, theta.shape[1]-1):
                if theta[row][col] < np.pi/8 or theta[row][col] >= 7*np.pi/8:#|
                    thetaPixles[row][col] = 0
                elif theta[row][col] < 3*np.pi/8:#/
                    thetaPixles[row][col] = 85
                elif theta[row][col] < 5*np.pi/8:#-
                    thetaPixles[row][col] = 170
                elif theta[row][col] < 7*np.pi/8:#\
                    thetaPixles[row][col] = 255
                else:
                    print(theta[row][col])
        self.doSave('sTheta.jpg', thetaPixles)
        
    def findCorners(self, pixles):
        components = self.connectedComponents(pixles)
        components = sorted(components, key = len)
        possibleSudokus = components[-4:]
        possibleCorners = [[min(sudoku, key = lambda x:x[0]+x[1]), max(sudoku, key = lambda x:x[0]-x[1]), max(sudoku, key = lambda x:x[0]+x[1]), min(sudoku, key = lambda x:x[0]-x[1])] for sudoku in possibleSudokus]#lt lb rb rt
        
        #Find most Sudoku-like
        biggest = [0]
        for i in range(0, 4):
            size = ((i+1)/4)*((possibleCorners[i][2][0] - possibleCorners[i][0][0])**2 + (possibleCorners[i][2][1] - possibleCorners[i][0][1])**2 + (possibleCorners[i][1][0] - possibleCorners[i][3][0])**2 + (possibleCorners[i][1][1] - possibleCorners[i][3][1])**2)#rb-lt,lb-rt
            
            if size >= biggest[0]:
                biggest = [size, i]

        if verbose:
            for i in range(1, 5):
                self.saveArray('component' + str(i) + '.jpg', components[-i])
        
        corners = possibleCorners[biggest[1]]
        if verbose:
            self.saveArray('corners.jpg', corners)

        #self.updatePixles()

        self.removeComponent(possibleSudokus[biggest[1]])
        
        while len(components[0]) < 150:
            self.removeComponent(components[0])
            del components[0]

        if verbose:
            self.doSave('pix.jpg')

        return corners

    def removeComponent(self, component):
        for i in component:
            self.pixles[i[0]][i[1]] = 0

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

                        if len(moreOverflows) > 0:
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

    def getCells(self, corners):
        grid = []
        cellWidth = 28
        cellHeight = 28

        yPosition = 0
        for y in range(9):
            xPosition = 0
            grid.append([])
            for x in range(9):
                grid[-1].append(self.pixles[yPosition:yPosition+28, xPosition:xPosition+28])#warning
                xPosition += 28
            yPosition += 28

        return grid

    #Test Saves
    def doSave(self, fileName, pixles=''):
        if pixles == '':
            imageFromPixles = Image.fromarray(np.uint8(self.pixles))
        else:
            imageFromPixles = Image.fromarray(np.uint8(pixles))#.convert('RGB')
        imageFromPixles.save(fileName)

    def saveArray(self, fileName, array):
        validPixles = np.empty(self.pixles.shape)
        for i in array:
            validPixles[i[0]][i[1]] = 255

        self.doSave(fileName, validPixles)

verbose = False
dataSetName = 'dataset/normal/'

try:
    with open('genLog.txt', 'r') as f:
        fileStart = int(f.read())
except Exception as e:
    print(e)
    fileStart = 1

try:
    sudokuSets = np.load('dataSet.npy', allow_pickle=True)
except Exception as e:
    print(e)
    sudokuSets = np.array([[1,2]])
    
fileNo = fileStart

#try:
for fileNo in range(fileStart, 1088):
    sudokuSet = []
    imageName = 'image' + str(fileNo) + '.jpg'
    try:
        with open(dataSetName + imageName, 'rb') as f:
            imageData = f.read()
        print('opened', imageName)
    except Exception as e:
        print(e)
        continue

    datName = 'image' + str(fileNo) + '.dat'
    try:
        with open(dataSetName + datName, 'r') as f:
            datData = f.read()
        print('opened', datName)
    except Exception as e:
        print(e)
        continue

    sudokuValues = [i.split(' ') for i in datData.split('\n')[2:11]]
    sudoku = sudokuImage(imageData)
    sudoku.recognise()
    for row in range(9):
        for col in range(9):
            sudokuSet.append([sudoku.grid[row][col].flatten()/255, int(sudokuValues[row][col])])

    sudokuSets = np.append(sudokuSets, sudokuSet, axis = 0)
    np.save('dataSet.npy', sudokuSets)

#except Exception as e:
#    print(e)
#    with open('genLog.txt', 'w') as f:
#        f.write(str(fileNo))

#with open('genLog.txt', 'w') as f:
#        f.write(str(fileNo+1))