import numpy as np
from PIL import Image
import io

def generateGaussianKernel(sigma, size):
    #kernel1D = []
    #for distance in range(-size,size+1):
    #    kernel1D.append(1 / ((2 * np.pi)**(1/2) * sigma) * np.exp(-(distance / sigma)**2 / 2))
    #kernel = np.outer(kernel1D, kernel1D)

    kernel = np.zeros((2*size+1,2*size+1))
    coefficient = 2*sigma**2

    for i in range(1, 2*size+2):
        for j in range(1, 2*size+2):
            kernel[i-1][j-1] = np.exp(-((i-(size+1))**2+(j-(size+1))**2)/2/coefficient)/(coefficient*np.pi)**(1/2)
    return kernel

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
        self.canny()
        
        #Isolate sudoku
        corners = self.findCorners()

        #Format into 9*9 28 pixle cells
        #self.savePixles()
        self.transform((252,252), Image.QUAD, [i for j in corners for i in j[::-1]])
        self.updatePixles()
        if verbose:
            self.save('warped.jpg')

        #Seperate

        grid = self.getCells(corners)
        
        for row in range(9):
            for col in range(9):
                self.doSave('grid/' + str(row) + str(col) + '.jpg', grid[row][col])
                self.values[row][col] = nt.recognise(grid[row][col])

    def canny(self):
        #Apply gaussian blur
        self.pixles = self.convolve(self.gaussianKernel)

        self.pixlesHeight -= np.shape(self.gaussianKernel)[0]*2
        self.pixlesWidth -= np.shape(self.gaussianKernel)[1]*2

        if verbose:
            self.doSave('gaussian.jpg')

        #Obtain sobel operators
        Gx = self.convolve(self.xKernel)
        Gy = self.convolve(self.yKernel)
        self.pixles = Gx

        gradient = np.hypot(Gx, Gy)
        theta = np.arctan2(Gy, Gx)
        theta[theta<0] += np.pi

        self.pixlesHeight -= 2
        self.pixlesWidth -= 2

        if verbose:
            self.doSave('gradient.jpg', gradient)
            self.doSave('theta.jpg', (255/np.pi)*theta)
            self.doSave('gx.jpg', 127.5+Gx/8)
            self.doSave('gy.jpg', 127.5+Gy/8)
            self.sTheta(theta)

        #Thin edge
        self.suppress(gradient, theta)
        if verbose:
            self.doSave('suppress.jpg')

        self.pixles[self.pixles < 15] =0
        self.pixles[self.pixles >= 15] = 255
        if verbose:
            self.doSave('Threshold.jpg')

    def convolve(self, kernel):
        kernel = np.flipud(np.fliplr(kernel))
        newPixles = np.zeros(self.pixles.shape)
        kernelHeight = kernel.shape[0] // 2
        kernelWidth = kernel.shape[1] // 2

        for row in range(kernelHeight, self.pixlesHeight-kernelHeight):
            for col in range(kernelWidth, self.pixlesWidth-kernelWidth):
                #Map kernel to sub-matrix then sum to give central pixle
                newPixles[row][col] = (kernel*self.pixles[row-kernelWidth:row+kernelWidth+1, col-kernelHeight:col+kernelHeight+1]).sum()

        return newPixles[kernelHeight:self.pixlesHeight-kernelHeight, kernelWidth:self.pixlesWidth-kernelWidth]

    def suppress(self, gradient, theta, weight = 0.9):
        self.pixles = np.zeros(self.pixles.shape)
        for row in range(1, self.pixlesHeight-1):
            for col in range(1, self.pixlesWidth-1):
                if theta[row][col] < np.pi/8 or theta[row][col] >= 7*np.pi/8:#|
                    if gradient[row][col] >= weight*max(gradient[row][col+1], gradient[row][col-1]):
                        self.pixles[row][col] = gradient[row][col]
                elif theta[row][col] < 3*np.pi/8:#/
                    if gradient[row][col] >= weight*max(gradient[row-1][col-1], gradient[row+1][col+1]):
                        self.pixles[row][col] = gradient[row][col]
                elif theta[row][col] < 5*np.pi/8:#-
                    if gradient[row][col] >= weight*max(gradient[row+1][col], gradient[row-1][col]):
                        self.pixles[row][col] = gradient[row][col]
                elif theta[row][col] < 7*np.pi/8:#\
                    if gradient[row][col] >= weight*max(gradient[row-1][col+1], gradient[row+1][col-1]):
                        self.pixles[row][col] = gradient[row][col]
                else:
                    print(theta[row][col])
                        
    def sTheta(self, theta):
        thetaPixles = np.empty(self.pixles.shape)
        for row in range(1, self.pixlesHeight-1):
            for col in range(1, self.pixlesWidth-1):
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
        
    def findCorners(self):
        components = self.connectedComponents()
        components = sorted(components, key = len)
        possibleSudokus = components[-4:]
        possibleCorners = [[min(sudoku, key = lambda x:x[0]+x[1]), max(sudoku, key = lambda x:x[0]-x[1]), max(sudoku, key = lambda x:x[0]+x[1]), min(sudoku, key = lambda x:x[0]-x[1])] for sudoku in possibleSudokus]#lt lb rb rt
        
        biggest = [0]
        for i in range(0, 4):
            size = (possibleCorners[i][2][0] - possibleCorners[i][0][0])**2 + (possibleCorners[i][2][1] - possibleCorners[i][0][1])**2
            if size >= biggest[0]:
                biggest = [size, possibleCorners[i]]

        if verbose:
            for i in range(1, 5):
                self.saveArray('component' + str(i) + '.jpg', components[-i])
        
        corners = biggest[1]
        if verbose:
            self.saveArray('corners.jpg', corners)

        return corners

    def connectedComponents(self):
        components = []
        for row in range(self.pixlesHeight):
            for col in range(self.pixlesWidth):
                if self.pixles[row][col] == 255:
                    component, overflows = self.floodFill(row, col)

                    while len(overflows) > 0:
                        moreOverflows = []
                        print(len(overflows))
                        for row, col in overflows:
                            for r in range(-1,2):
                                for c in range(-1,2):
                                    if r != 0 or c != 0:
                                        componentIn, overflowIn = self.floodFill(row+r, col+c)
                                        component += componentIn
                                        moreOverflows += overflowIn

                        overflows = [i for i in moreOverflows]

                        if len(moreOverflows) > 0:
                            print('uo', len(moreOverflows))

                    components.append(component)

        return components

    def floodFill(self, row, col, size = 0):
        if self.pixles[row][col] == 255 and row>=0 and row<self.pixlesHeight and col>=0 and col<self.pixlesWidth:
            if size > 900:#overflow
                return [], [[row,col]]
            else:
                self.pixles[row][col] = 0
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
        activation = nt.run(image.flatten()/255)[-1]
        return np.argmax(activation)

with open('testImage.jpg', 'rb') as f:
    testData = f.read()

verbose = True
nt = network()


sudoku = sudokuImage(testData)
sudoku.recognise()