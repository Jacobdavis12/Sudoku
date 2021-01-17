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
        self.values = []

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

    def recognise(self):
        #Detect edges
        self.canny()
        
        #Isolate sudoku
        corners = self.findCorners()

        #Format into 9*9 28 pixle cells
        self.transform((252,252), Image.QUAD, [i for j in corners for i in j[::-1]])
        self.updatePixles()

        #Seperate
        grid = self.getCells(corners)

        for row in range(9):
            self.values.append([])
            for col in range(9):
                self.values[row].append(nt.recognise(grid[row][col]))

    def canny(self):
        #Apply gaussian blur
        self.pixles = self.convolve(self.gaussianKernel)

        self.pixlesHeight -= np.shape(self.gaussianKernel)[0]*2
        self.pixlesWidth -= np.shape(self.gaussianKernel)[1]*2

        #Obtain sobel operators
        Gx = self.convolve(self.xKernel)
        Gy = self.convolve(self.yKernel)
        self.pixles = Gx

        gradient = np.hypot(Gx, Gy)
        theta = np.arctan2(Gy, Gx)
        theta[theta<0] += np.pi

        self.pixlesHeight -= 2
        self.pixlesWidth -= 2

        #Thin edge
        self.suppress(gradient, theta)

        self.pixles[self.pixles < 15] =0
        self.pixles[self.pixles >= 15] = 255

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
        
        corners = biggest[1]

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
        activation = nt.run(image.flatten()/255)[-1]
        #print(activation)
        imageFromPixles = Image.fromarray(np.uint8(image))
        imageFromPixles.save('trala.jpg')
        #input(np.argmax(activation))
        return np.argmax(activation)

nt = network()
addr = 'localhost'
port = 8080
serverAddress = (addr, port)

with socketserver.TCPServer(serverAddress, handler) as httpd:
    print(f'Starting httpd server on {addr}:{port}')
    httpd.serve_forever()