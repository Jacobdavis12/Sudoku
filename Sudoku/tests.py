import numpy as np
from PIL import Image
import random

#def generate44grid(sudokuList = []):
#    if len(sudokuList) >= 16:
#        yield sudokuList
#    else:
#        yield from generate44grid(sudokuList + [''])
#        yield from generate44grid(sudokuList + ['1'])
#        yield from generate44grid(sudokuList + ['2'])
#        yield from generate44grid(sudokuList + ['3'])
#        yield from generate44grid(sudokuList + ['4'])

#def generate44grid(sudokuList = []):
#    if len(sudokuList) >= 16:
#        return sudokuList
#    else:
#        case = random.randint(0, 4)
#        if case == 0:
#            return generate44grid(sudokuList + [''])
#        else:
#            return generate44grid(sudokuList + [str(case)])
pixlesTemp = []

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
        return np.argmax(activation), np.max(activation)

class sudoku():
    firstChange = []
    noOfSolutions = 1

    def __init__(self, rawData, SquareRowCount, SquareColCount, CellRowCount, CellColCount):
        self.squareRowCount = SquareRowCount
        self.squareColCount = SquareColCount
        self.cellRowCount = CellRowCount
        self.cellColCount = CellColCount

        self.grid = np.asarray(self.squareToRow([[rawData[str(col)+str(row)] for row in range(self.squareRowCount*self.squareColCount)] for col in range(self.squareRowCount*self.squareColCount)]))

        self.possible = ''
        for row in range(self.squareRowCount*self.squareColCount):
            for col in range(self.squareRowCount*self.squareColCount):
                if len(self.grid[row][col]) == 1 and self.grid[row][col] not in self.possible:
                    self.possible += self.grid[row][col]

        i = 0
        chars = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']
        while len(self.possible) < self.cellRowCount*self.cellColCount:
            if chars[i] not in self.possible:
                self.possible += chars[i]
            i+=1

        self.possibilities = np.full(self.grid.shape, self.possible)
        for row in range(self.squareRowCount*self.squareColCount):
            for col in range(self.squareRowCount*self.squareColCount):
                if self.grid[row][col] != '':
                    self.possibilities[row][col] = str(self.grid[row][col])

    def getValues(self):
        values = {}
        squareGrid = self.squareToRow(self.grid)
        for square in range(self.squareRowCount*self.squareColCount):
            for cell in range(self.cellRowCount*self.cellColCount):
                values[str(square)+str(cell)] = squareGrid[square][cell]
        return values

    def getNoOfSolutions(self):
        return self.noOfSolutions 

    def solve(self):
        #print('Solving')
        
        if self.check(self.possibilities):
            #self.display()
            self.applyRestrictions(self.possibilities)
            #self.display()
            
            if self.isSolved():
                #print('Solved')
                self.grid = self.possibilities
            else:
                #print('Backtracking')
                solutions = self.recursiveSolve(self.possibilities)
                self.noOfSolutions = len(solutions)

                if len(solutions) == 0:#If no solution dont update grid
                    self.noOfSolutions = 0
                else:
                    self.possibilities = solutions[0]
                    if self.firstChange == []:
                        for row in range(self.squareRowCount*self.cellRowCount):
                            for col in range(self.squareColCount*self.cellColCount):
                                if self.possibilities[row][col] != self.grid[row][col]:
                                    self.firstChange = [[row, col], self.possibilities[row][col]]
            
                    self.grid = self.possibilities
        else:#If no solution dont update grid
            self.noOfSolutions = 0

    def getHint(self):
        self.solve()
        if self.noOfSolutions == 0:
            return False
        else:
            return self.firstChange
  
    def checkSudoku(self):
        return self.check(self.possibilities)

    def check(self, grid):
        for restriction in self.getRestrictionPointers():
            for pointers in restriction:
                inRestriction = ''
                for pointer in pointers:
                    if len(grid[pointer[0]][pointer[1]]) == 1:
                        if grid[pointer[0]][pointer[1]] in inRestriction:
                            return [pointer, pointers]
                        inRestriction += grid[pointer[0]][pointer[1]]
                    if len(grid[pointer[0]][pointer[1]]) == 0:
                        return False

        return True

    def applyRestrictions(self, possibilities):
        previousPossibilities = []
        
        while previousPossibilities != str(possibilities):
            previousPossibilities = str(possibilities)
            for restriction in self.getRestrictionPointers():
                for pointers in restriction:
                    possibilities = self.remove(pointers, possibilities)
                    possibilities = self.insert(pointers, possibilities)

        return possibilities
    
    def getRestrictionPointers(self):
        yield self.rowRestriction()
        yield self.columnRestriction()
        yield self.squareRestriction()

    def recursiveSolve(self, possibilities):
        if self.check(possibilities) == True:
            row = 0
            col = 0
            while not len(possibilities[row][col]) > 1:
                
                row += 1
                if row == self.squareRowCount*self.cellRowCount:
                    col += 1
                    row = 0

                if col == self.squareColCount*self.cellColCount:#solution found
                    return [possibilities]

            solutions = []
            for possible in possibilities[row][col]:

                newPossibilities = [[str(possibilities[row][col]) for col in range(self.squareRowCount*self.cellRowCount)] for row in range(self.squareColCount*self.cellColCount)]
                newPossibilities[row][col] = possible
                solutions.extend(self.recursiveSolve(self.applyRestrictions(newPossibilities)))

            return solutions

        return []

    def isSolved(self):
        for row in range(self.squareRowCount*self.cellRowCount):
            for col in range(self.squareColCount*self.cellColCount):
                if len(self.possibilities[row][col]) != 1:
                    return False

        return True

    def remove(self, pointers, possibilities):
        for pointer in pointers:
            if len(possibilities[pointer[0]][pointer[1]]) == 1:
                for pointerRemove in pointers:
                    if pointerRemove != pointer:
                        if possibilities[pointer[0]][pointer[1]] in possibilities[pointerRemove[0]][pointerRemove[1]]:
                            if len(possibilities[pointerRemove[0]][pointerRemove[1]]) == 2:
                                if self.firstChange == []:
                                    self.firstChange = [pointerRemove, pointers]
                            possibilities[pointerRemove[0]][pointerRemove[1]] = possibilities[pointerRemove[0]][pointerRemove[1]].replace(possibilities[pointer[0]][pointer[1]], '')

        return possibilities

    def insert(self, pointers, possibilities):
        for number in range(len(self.possible)):
            sole = False
            for pointer in pointers:
                if str(number) in possibilities[pointer[0]][pointer[1]]:
                    if sole == False:
                        sole = pointer
                    else:
                        sole = False
                        break

            if sole != False and len(possibilities[sole[0]][sole[1]]) != 1:
                if self.firstChange == []:
                    self.firstChange = [sole, pointers]
                possibilities[sole[0]][sole[1]] = str(number)

        return possibilities

    def rowRestriction(self):
        for row in range(self.squareRowCount*self.cellRowCount):
            yield [[row, col] for col in range(self.squareColCount*self.cellColCount)]

    def columnRestriction(self):
        for col in range(self.squareColCount*self.cellColCount):
            yield [[row, col] for row in range(self.squareRowCount*self.cellRowCount)]

    def squareRestriction(self):
        for row in range(0,self.squareRowCount*self.cellRowCount,self.cellRowCount):
            for col in range(0,self.squareColCount*self.cellColCount,self.cellColCount):
                yield [[row+row2, col+col2] for col2 in range(0,self.cellColCount,1) for row2 in range(0,self.cellRowCount,1)]

    def display(self):
        for row in self.grid:
            print([[row[col+col2] for col2 in range(0,self.cellColCount,1)] for col in range(0,self.squareColCount*self.cellColCount,self.cellColCount)])
        print()

    def squareToRow(self, grid):
        return [[grid[row+row2][col+col2] for row2 in range(0,self.cellRowCount,1) for col2 in range(0,self.cellColCount,1)] for row in range(0,self.squareRowCount*self.cellRowCount,self.cellRowCount) for col in range(0,self.squareRowCount*self.cellColCount,self.cellColCount)]

    def squareToRowIndividual(self, x, y):
        x2 = (x//self.squareRowCount)*self.squareColCount + y//self.cellRowCount
        y2 = (x%self.squareColCount)*self.squareRowCount + y%self.squareColCount
        return x2, y2

class sudokuX(sudoku):
    def __init__(self, rawData):
        super(sudokuX, self).__init__(rawData, 3, 3, 3, 3)

    def getRestrictionPointers(self):
        yield from super(sudokuX, self).getRestrictionPointers()
        yield self.diagonalRestriction()

    def diagonalRestriction(self):
        yield [[dia, dia] for dia in range(0, 9)]
        yield [[8-dia, dia] for dia in range(0, 9)]

def testImage():
    ds = np.load('dataset.npy', allow_pickle=True)[1:]
    for datum in range(len(ds)):
        imageFromPixles = Image.fromarray(np.uint8([ds[datum][0][i:i+28]*255 for i in range(0,784,28)]))
        imageFromPixles.save('testResults/testImage/' + str(ds[datum][1]) +  '/' + str(datum) + '.jpg')
        del imageFromPixles

def testNetwork():
    ds = np.load('dataset.npy', allow_pickle=True)[1:]
    results = {0: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
               1: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
               2: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
               3: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
               4: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
               5: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
               6: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
               7: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
               8: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}, 
               9: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}}
    for datum in range(len(ds)):
        guess = nt.recognise(ds[datum][0])[0]
        results[guess][ds[datum][1]] += 1
        imageFromPixles = Image.fromarray(np.uint8([ds[datum][0][i:i+28]*255 for i in range(0,784,28)]))
        imageFromPixles.save('testResults/testNetwork/' + str(guess) +  '/[' + str(ds[datum][1]) + ']' + str(datum) + '.jpg')
        del imageFromPixles

    [print(i, results[i]) for i in results]

def testSolver99():
    sudokuDataset = np.load('testData\sudokus99.npy', allow_pickle=True)[:100]
    result = {}

    for sudokuData in sudokuDataset:
        sudokuDict = {}
        for row in range(9):
            for col in range(9):
                if sudokuData[row][col] == '0':
                    sudokuDict[str(row)+str(col)] = ''
                else:
                    sudokuDict[str(row)+str(col)] = sudokuData[row][col]

        sud = sudoku(sudokuDict, 3, 3, 3, 3)

        sud.display()
        sud.solve()
        sud.display()

        print(sud.getNoOfSolutions())
        if sud.getNoOfSolutions() in result:
            result[sud.getNoOfSolutions()] += 1
        else:
            result[sud.getNoOfSolutions()] = 1

        del sud

    print(result)

def testSolver44():
    sudokuDataset = np.load('testData\sudokus44.npy', allow_pickle=True)
    result = {}

    for sudokuData in sudokuDataset:
        sudokuDict = {}
        for row in range(4):
            for col in range(4):
                if sudokuData[row][col] == '0':
                    sudokuDict[str(row)+str(col)] = ''
                else:
                    sudokuDict[str(row)+str(col)] = sudokuData[row][col]

        sud = sudoku(sudokuDict, 2, 2, 2, 2)

        sud.display()
        sud.solve()
        sud.display()

        print(sud.getNoOfSolutions())
        if sud.getNoOfSolutions() in result:
            result[sud.getNoOfSolutions()] += 1
        else:
            result[sud.getNoOfSolutions()] = 1

        del sud

    print(result)

def testSolverX():
    sudokuDataset = np.load('testData\sudokus99.npy', allow_pickle=True)[:100]
    result = {}

    for sudokuData in sudokuDataset:
        sudokuDict = {}
        for row in range(9):
            for col in range(9):
                if sudokuData[row][col] == '0':
                    sudokuDict[str(row)+str(col)] = ''
                else:
                    sudokuDict[str(row)+str(col)] = sudokuData[row][col]

        sud = sudokuX(sudokuDict)

        sud.display()
        sud.solve()
        sud.display()

        print(sud.getNoOfSolutions())
        if sud.getNoOfSolutions() in result:
            result[sud.getNoOfSolutions()] += 1
        else:
            result[sud.getNoOfSolutions()] = 1

        del sud

    print(result)

def testEdge():
   
    for a, b in [[[250, 50], [250, 450]], [[50, 50], [450, 450]], [[450, 50], [50, 450]], [[250, 250], [250, 250]]]:
        pixles = np.zeros((500, 500))

        corners = np.zeros((500, 500))
        corners[a[0]][a[1]] = 255
        corners[b[0]][b[1]] = 255
        imageFromPixles = Image.fromarray(corners)
        imageFromPixles.show()
        del imageFromPixles

        dx = (b[0] - a[0])
        dy = (b[1] - a[1])
        lengthXY = (dx**2+dy**2)**(1/2)
        if lengthXY == 0:
            print()
            imageFromPixles = Image.fromarray(corners)
            imageFromPixles.show()
        else:
            for row in range(500):
                for col in range(500):
                    if abs((a[0] - row)*dy - (a[1]- col)*dx)/lengthXY <= 3:#On same line
                        pixles[row][col] = 255

            imageFromPixles = Image.fromarray(pixles)
            imageFromPixles.show()
        del imageFromPixles

def connectedComponents(pixles):
        global pixlesTemp
        pixlesTemp = np.copy(pixles)
        components = []
        for row in range(pixles.shape[0]):
            for col in range(pixles.shape[1]):
                if pixlesTemp[row][col] == 255:
                    component, overflows = floodFill(row, col)

                    while len(overflows) > 0:
                        moreOverflows = []
                        print(len(overflows))
                        for row, col in overflows:
                            for r in range(-1,2):
                                for c in range(-1,2):
                                    if r != 0 or c != 0 :
                                        componentIn, overflowIn = floodFill(row+r, col+c)
                                        component += componentIn
                                        moreOverflows += overflowIn

                        overflows = [i for i in moreOverflows]

                        print('uo', len(moreOverflows))

                    components.append(component)

        return components

def floodFill(row, col, size = 0):
        if row >= 0 and row < pixlesTemp.shape[0] and col >= 0 and col < pixlesTemp.shape[1] and pixlesTemp[row][col] == 255:
            if size > 900:#overflow
                return [], [[row,col]]
            else:
                pixlesTemp[row][col] = 0
                component = [[row,col]] 
                overflow = []
                for r in range(-1,2):
                    for c in range(-1,2):
                        if r != 0 or c != 0:
                            componentIn, overflowIn = floodFill(row+r, col+c, size+1)
                            component += componentIn
                            overflow += overflowIn

                return component, overflow
        else:
            return [], []

def saveArray(fileName, array, pixles):
        print(fileName)
        validPixles = np.zeros(pixles.shape)
        for i in array:
            validPixles[i[0]][i[1]] = 255

        imageFromPixles = Image.fromarray(np.uint8(validPixles))
        imageFromPixles.save(fileName)

def testComponents():
    #pixles = np.asarray([[255 for i in range(20)] for j in range(20)])
    #pixles = np.asarray([[255 if (i+j)%2 == 0 else 0 for i in range(20)] for j in range(20)])
    pixles = np.asarray([[255 if (i+j)%5 == 0 else 0 for i in range(20)] for j in range(20)])

    imageFromPixles = Image.fromarray(pixles)
    imageFromPixles.show()
    del imageFromPixles

    components = connectedComponents(pixles)
    components = sorted(components, key = len)

    for i in range(len(components)):
        saveArray('images/components/' + str(i) + '.jpg', components[i], pixles)

#testImage()

#nt = network()
#testNetwork()

#testSolver99()

#testSolver44()

#testSolverX()

#testEdge()

testComponents()