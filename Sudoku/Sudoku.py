from http.server import BaseHTTPRequestHandler
import socketserver
import numpy as np
from PIL import Image
import io

class handler(BaseHTTPRequestHandler):
    header = '<html> <head> <title>Sudoku solver</title> <style>#header { font-size: 10vh; height: 10vh; margin: 0px; } #content { display: flex; flex-wrap: wrap; } #sudoku { height: 80vh; width: 80vh; display: grid; grid-template-columns: 26vh 26vh 26vh; grid-gap: 1vh; background-color: black; border:solid #cccccc 1vh; float:left; } .square { position: relative; height: 100%; width: 100%; display: grid; grid-template-columns: 8vh 8vh 8vh; grid-gap: 1vh; background-color: #c0c0c0; } .cell { font-size: 8vh; height: 8vh; text-align: center; background-color: white; border: 0px; } #image, .type { display: none; } .control { background-color: #004080; border-radius: 5px; border: 1px solid white; color: white; } #controls { flex-grow: 100; display: grid; grid-template-rows: 26vh 26vh 26vh; grid-gap: 1vh; white-space: nowrap; font-size: 6vh; padding: 1vh 0px 0px 1vh; } #controls > .control{ display: flex; height: 26vh; width: 100%; } @media screen and (orientation: portrait) { #header { font-size: 10vw; height: 10vw; } #sudoku { height: 80vw; width: 80vw; grid-template-columns: 26vw 26vw 26vw; grid-gap: 1vw; border-width: 1vw; } .square { grid-template-columns: 8vw 8vw 8vw; grid-gap: 1vw; } .cell { font-size: 8vw; height: 8vw; } #controls { padding-left: 0px; font-size: 6vw; height: 6vw; grid-template-rows: 26vw 26vw 26vw; grid-gap: 1vw; padding: 1vw 0px 0px 1vw; } #controls > .control{ height: 26vw; } } </style> </head> <body> <h1 id="header">Sudoku solver</h1> <div id="content"> <form id="sudoku" method="post"> <input name="type" class="type" value="sudoku"></input>'
    footer = '</form> <div id="controls"> <label for="image" class="control"> Upload Sudoku image <form id = "upload" method="post" enctype="multipart/form-data"> <input name="type" class="type" value="image"></input> <input id="image" type="file" name="filename" onchange="form.submit()"> </form> </label> <label class="control" for="photo"> Preferences <form id = "preference" method="post"> <input name="type" class="type" value="preference"></input> </form> </label> <label class="control" onclick="sudoku.submit()"> Solve Sudoku </label> </div> </div> </body> </html>'
    
    def sendHtml(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(data.encode('utf8'))
        
    def do_GET(self, params = {}):
        index = self.header
        for h in range(9):
            index += '<div id="s' + str(h) + '" class="square">\n'
            for w in range(9):
                if str(h) + str(w) in params:
                    index += '<input type="text" size="1" maxlength="1" name="' + str(h) + str(w) + '" value="' + str(params[str(h) + str(w)]) + '" class="cell">\n'
                else:
                    index += '<input type="text" size="1" maxlength="1" name="' + str(h) + str(w) + '" value="" class="cell">\n'
            index += '</div>\n'

        index += self.footer
            
        self.sendHtml(index)
            
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        postData = self.rfile.read(content_length)
        print(postData)
        if b'sudoku' in postData:
            sudokuInput = sudoku(self.parseSudoku(postData))
            sudokuInput.solve()
            self.do_GET(sudokuInput.getValues())
        elif b'image' in postData:
            imageInput = sudokuImage(self.parseImage(postData))
            imageInput.recognise()
            self.do_GET(imageInput.getValues())
        elif b'preference' in postData:
            self.parsePref(postData)
            self.do_GET()
        else:
            print('error')

    def parseSudoku(self, rawData):
        rawData = str(rawData)
        data = {}
        for value in rawData[:-1].split('&')[1:]:
            values = value.split('=')
            data[values[0]] = values[1]
        print(data)
        return data

    def parseImage(self, rawData):
        filename = rawData.split(b'filename="')[1].split(b'"')[0]
        data = rawData.split(b'\r\n\r\n')
        with open('test.jpg', 'wb') as f:
            f.write(data[-1])
        return data[-1]

    def parsePref(self, rawData):
        pass

class sudoku():
    def __init__(self, rawData):
        self.rawValues = rawData
        self.grid = self.rowToSquare([[rawData[str(row)+str(col)] for row in range(9)] for col in range(9)])

    def squareToRow(self, grid):
        return [[grid[row+row2][col+col2] for row2 in range(0,3,1) for col2 in range(0,3,1)] for row in range(0,9,3) for col in range(0,9,3)]

    def rowToSquare(self, grid):
        return [[grid[row+row2][col+col2] for row2 in range(0,3,1) for col2 in range(0,3,1)] for row in range(0,9,3) for col in range(0,9,3)]

    def solve(self):
        print('solved')

    def getValues(self):
        values = {}
        squareGrid = self.squareToRow(self.grid)
        for square in range(9):
            for cell in range(9):
                values[str(cell)+str(square)] = squareGrid[square][cell]
        return values

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
                if self.values[row][col] != 0:
                    outputValues[str(row)+str(col)] = str(self.values[row][col])[0]
                else:
                    outputValues[str(row)+str(col)] = ''

        return outputValues

    def updatePixles(self):
        self.pixlesHeight = self.height
        self.pixlesWidth = self.width

        self.pixles = np.array([[self.getpixel((w, h)) for w in range(self.pixlesWidth)] for h in range(self.pixlesHeight)])

    def savePixles(self):
        self._img = Image.fromarray(np.uint8(self.pixles))

    def recognise(self):

        #Detect edges
        #self.pixles = self.canny(self.pixles)
        self.pixles = self.adaptiveThreshold(self.convolve(self.gaussianKernel, self.pixles))

        #Isolate sudoku
        corners = self.findCorners(self.pixles)

        #Format into 9*9 28 pixle cells
        self.savePixles()
        self.transform((252,252), Image.QUAD, [i for j in corners for i in j[::-1]])
        self.updatePixles()

        #Seperate

        grid = self.getCells(corners)
        
        for row in range(9):
            for col in range(9):
                self.values[row][col] = nt.recognise(grid[row][col]/255)

    def canny(self, pixles):
        #Apply gaussian blur
        gaussian = self.convolve(self.gaussianKernel, pixles)

        #Obtain sobel operators
        Gx = self.convolve(self.xKernel, gaussian)
        Gy = self.convolve(self.yKernel, gaussian)

        gradient = np.hypot(Gx, Gy)
        theta = np.arctan2(Gy, Gx)
        theta[theta<0] += np.pi

        #Thin edge
        supPix = self.suppress(gradient, theta)

        supPix[supPix < 50] =0
        supPix[supPix >= 50] = 255

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

    def adaptiveThreshold(self, pixles, size=3):
        thresholdPixles = np.zeros(pixles.shape)
        d = 2*size+1
        for row in range(pixles.shape[0]):
            for col in range(pixles.shape[1]):
                if pixles[row][col] <= np.mean(pixles[row-size:row+size+1, col-size:col+size+1])-1:
                    thresholdPixles[row][col] = 255

        return thresholdPixles
        
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
        
        corners = possibleCorners[biggest[1]]

        #self.updatePixles()

        self.removeComponent(possibleSudokus[biggest[1]])
        
        while len(components[0]) < 150:
            self.removeComponent(components[0])
            del components[0]

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

class network:
    def __init__(self, fileName = False):
        self.loadNet()
        self.dimensions = [784] + [len(i) for i in self.weights]

    def loadNet(self, filename = 'network.npz'):
        data = np.load(filename, allow_pickle=True)
        self.biases = data['bias']
        self.weights = data['weight']

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
        #print(activation)
        #input(np.argmax(activation))
        return np.argmax(activation)

def display(pixles):
    imageFromPixles = Image.fromarray(np.uint8([pixles[i:i+28]*255 for i in range(0,784,28)]))
    imageFromPixles.show()

nt = network()
addr = 'localhost'
port = 8080
serverAddress = (addr, port)

with socketserver.TCPServer(serverAddress, handler) as httpd:
    print(f'Starting httpd server on {addr}:{port}')
    httpd.serve_forever()