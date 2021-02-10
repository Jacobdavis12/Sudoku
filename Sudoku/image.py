import numpy as np
from PIL import Image
import io

#def generateGaussianKernel(sigma, size):
#    #kernel1D = []
#    #for distance in range(-size,size+1):
#    #    kernel1D.append(1 / ((2 * np.pi)**(1/2) * sigma) * np.exp(-(distance / sigma)**2 / 2))
#    #kernel = np.outer(kernel1D, kernel1D)

#    kernel = np.zeros((2*size+1,2*size+1))
#    coefficient = 2*sigma**2

#    for i in range(1, 2*size+2):
#        for j in range(1, 2*size+2):
#            kernel[i-1][j-1] = np.exp(-((i-(size+1))**2+(j-(size+1))**2)/2/coefficient)/(coefficient*np.pi)**(1/2)
#    return kernel

#def generateGaussianKernel(sigma, size):
#        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
#        xx, yy = np.meshgrid(ax, ax)

#        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

#        return kernel / np.sum(kernel)

def generateGaussianKernel(sigma, size):
    xDistance = np.array((2*size+1)*[[i for i in range(-size, size+1)]])
    distance = -0.5 * (xDistance**2 + xDistance.T**2)
    kernel = np.exp(distance/sigma**2)
    return kernel/kernel.sum()

class sudokuImage():

    def __init__(self, imageData):
        #Inherit Module by delegated wrapper
        self._img = Image.open(io.BytesIO(imageData)).convert('L')

        #Initialise properties
        self.pixlesHeight = self.height
        self.pixlesWidth = self.width

        self.pixles = np.array([[self.getpixel((w, h)) for w in range(self.pixlesWidth)] for h in range(self.pixlesHeight)])
        self.values = np.empty((9,9))
        self.confidence = np.empty((9,9))

        #Initialise kernels
        self.gaussianKernel = generateGaussianKernel(1, 5)#np.array([[2,4,5,4,2],[4,9,12,9,4],[5,12,15,12,5],[4,9,12,9,4],[2,4,5,4,2]])/159#
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

        self.pixles = self.convolve(self.gaussianKernel, self.pixles)

        if verbose:
            self.doSave('gaussian.jpg')

        #Detect edges
        #self.pixles = self.canny(self.pixles)
        self.pixles = self.adaptiveThreshold(self.pixles)
        if verbose:
            self.doSave('Threshold.jpg')

        #Isolate sudoku
        corners = self.findCorners(self.pixles)

        #Format into 9*9 28 pixle cells
        self.savePixles()
        self.transform((252,252), Image.QUAD, [i for j in corners for i in j[::-1]])
        self.updatePixles()

        #while self.pixles[1][0] == 255 and self.pixles[1][-1] == 255:
        #    self.pixles = self.pixles[1:-1, :]
        #while self.pixles[-2][0] == 255 and self.pixles[-2][-1] == 255:
        #    self.pixles = self.pixles[1:-1, :]
        #while self.pixles[0][1] == 255 and self.pixles[1][-1] == 255:
        #    self.pixles = self.pixles[:, 1:-1]
        #while self.pixles[0][-2] == 255 and self.pixles[-1][-2] == 255:
        #    self.pixles = self.pixles[:, 1:-1]

        #self.savePixles()
        #self.resize((252,252))
        #self.updatePixles()
        #self.show()

        if verbose:
            self.doSave('warped.jpg')

        #self.pixles = self.removeNonNumbers(self.pixles)
        #if verbose:
        #    self.doSave('removed.jpg')

        #Seperate
        self.grid = self.getCells(corners)

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

        self.pixles = np.concatenate(cleanedRow)
        if verbose:
            self.doSave('removed.jpg')

        print(self.values)

    def removeNonNumbers(self, pixles):
        components = self.connectedComponents(pixles)
        components = sorted(components, key = len)

        for i in range(len(components)):
            self.saveArray('compon/' + str(len(components[i])) + 'l' + str(i) + '.jpg', components[i], pixles)
        s = sum([len(i) for i in components])/len(components)

        while len(components[0]) < s*5:
            pixles = self.removeComponent(components[0], pixles)
            del components[0]

        while len(components[-1]) > s*8:
            pixles = self.removeComponent(components[-1], pixles)
            del components[-1]      

        for i in range(len(components)):
            self.saveArray('compon/' + str(len(components[i])) + 'l' + str(i) + '.jpg', components[i], pixles)

        return pixles

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

        #for row in range(kernelHeight, pixlesHeight-kernelHeight):
        #    for col in range(kernelWidth, pixlesWidth-kernelWidth):
        #        #Map kernel to sub-matrix then sum to give central pixle
        #        newPixles[row][col] = (kernel*pixles[row-kernelWidth:row+kernelWidth+1, col-kernelHeight:col+kernelHeight+1]).sum()

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

        return newPixles#[kernelHeight:pixlesHeight-kernelHeight, kernelWidth:pixlesWidth-kernelWidth]

    def adaptiveThreshold(self, pixles):
        size = 11
        kernel = -generateGaussianKernel(4, size)
        kernel[size][size] = kernel[size][size]+1

        localMean = self.convolve(kernel, pixles)
        if verbose:
            self.doSave('lm.jpg', 255*(localMean-localMean.min())/(localMean-localMean.min()).max())

        localMean[localMean>-1.5] = 0
        localMean[localMean!=0] = 255

        return localMean
        
    def findCorners(self, pixles):
        components = self.connectedComponents(pixles)
        components = sorted(components, key = len)
        possibleSudokus = components[-4:]
        possibleCorners = [[min(sudoku, key = lambda x:x[0]+x[1]), max(sudoku, key = lambda x:x[0]-x[1]), max(sudoku, key = lambda x:x[0]+x[1]), min(sudoku, key = lambda x:x[0]-x[1])] for sudoku in possibleSudokus]#lt lb rb rt
        
        #Find most Sudoku-like
        biggest = [0]
        for i in range(0, 4):
            size = ((i+1)**2/16)*((possibleCorners[i][2][0] - possibleCorners[i][0][0])**2 + (possibleCorners[i][2][1] - possibleCorners[i][0][1])**2 + (possibleCorners[i][1][0] - possibleCorners[i][3][0])**2 + (possibleCorners[i][1][1] - possibleCorners[i][3][1])**2)#rb-lt,lb-rt
            
            if size >= biggest[0]:
                biggest = [size, i]

        if verbose:
            for i in range(1, 5):
                self.saveArray('component' + str(i) + '.jpg', components[-i], pixles)
        
        corners = possibleCorners[biggest[1]]
        if verbose:
            self.saveArray('corners.jpg', corners, pixles)

        #self.updatePixles()

        #self.pixles = self.removeComponent(possibleSudokus[biggest[1]], pixles)

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
        imageFromPixles.save('images/' + fileName)

    def saveArray(self, fileName, array, pixles):
        validPixles = np.empty(pixles.shape)
        for i in array:
            validPixles[i[0]][i[1]] = 255

        self.doSave(fileName, validPixles)

    #CANNY
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

def display(pixles):
    imageFromPixles = Image.fromarray(np.uint8([pixles[i:i+28]*255 for i in range(0,784,28)]))
    imageFromPixles.show()

class testImage(sudokuImage):
    def recognise(self, tn):
        self.testName = tn
        if verbose:
            self.doSave('greyscale.jpg')

        self.pixles = self.convolve(self.gaussianKernel, self.pixles)

        if verbose:
            self.doSave('gaussian.jpg')

        #Detect edges
        for size in range(11, 22):
            for sigma in range(7, 11):
                self.adaptiveThreshold(self.pixles, size, sigma)

    def adaptiveThreshold(self, pixles, size, sigma):
        kernel = -generateGaussianKernel(sigma, size)
        kernel[size][size] = kernel[size][size]+1

        localMean = self.convolve(kernel, pixles)
        self.doSave(self.testName + '_' + str(size) + '_' + str(sigma) + '.jpg', 255*(localMean-localMean.min())/(localMean-localMean.min()).max())

nt = network()

verbose = True
with open('test9.jpg', 'rb') as f:
    testData = f.read()

sudoku = sudokuImage(testData)
sudoku.recognise()
sudoku.doSave('fr.jpg')

#verbose = True
#for i in range(15):
#     with open('test' + str(i) +'.jpg', 'rb') as f:
#        testdata = f.read()
#     sudoku = sudokuImage(testdata)
#     sudoku.recognise()
#     sudoku.doSave('resultd' + str(i) +'.jpg')
#     del sudoku
#     #sudoku.dosave('median' + str(i) +'.jpg', sudoku.median) = f.read()


#verbose = False
#for i in range(11):
#    with open('test' + str(i) +'.jpg', 'rb') as f:
#        testData = f.read()

#    sudoku = testImage(testData)
#    sudoku.recognise('test' + str(i))

