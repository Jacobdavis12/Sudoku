import numpy as np
from PIL import Image
import io

def generateGaussianKernel(sigma, size):
    xDistance = np.array((2*size+1)*[[i for i in range(-size, size+1)]])
    distance = -0.5 * (xDistance**2 + xDistance.T**2)
    kernel = np.exp(distance/sigma**2)
    return kernel/kernel.sum()

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
            cleanedRow.append(np.concatenate(self.grid[row], axis = 1))

        cleaned = np.concatenate(cleanedRow)
        if verbose:
            self.doSave('removed.jpg', cleaned)

        self.doSave('warped/n' + imageName, cleaned)

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

        componentPixles = np.zeros(pixles.shape)

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
        validPixles = np.zeros(pixles.shape)
        for i in array:
            validPixles[i[0]][i[1]] = 255

        self.doSave(fileName, validPixles)
verbose = True
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
for fileNo in range(1089):#fileStart
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

    del sudoku
    sudokuSets = np.append(sudokuSets, sudokuSet, axis = 0)
    np.save('dataSet.npy', sudokuSets)

#except Exception as e:
#    print(e)
#    with open('genLog.txt', 'w') as f:
#        f.write(str(fileNo))

#with open('genLog.txt', 'w') as f:
#        f.write(str(fileNo+1))