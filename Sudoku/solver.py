import numpy as np
class sudoku():
    def __init__(self, rawData):
        #self.rawValues = rawData
        #self.grid = self.rowToSquare([[rawData[str(row)+str(col)] for row in range(9)] for col in range(9)])
        #self.grid = np.asarray([['', '', '3', '', '4', '6', '', '2', '5'],
        #                        ['', '8', '4', '1', '2', '', '', '', ''],
        #                        ['', '', '', '', '', '8', '', '1', '9'],
        #                        ['8', '', '7', '', '', '', '1', '5', ''],
        #                        ['', '4', '', '', '', '', '', '9', ''],
        #                        ['', '2', '9', '', '', '', '3', '', '7'],
        #                        ['7', '9', '', '6', '', '', '', '', ''],
        #                        ['', '', '', '', '1', '9', '2', '7', ''],
        #                        ['6', '1', '', '2', '3', '', '9', '', '']])

        self.grid = np.asarray([['', '', '', '', '', '', '', '', ''],
                                ['', '', '', '', '', '', '', '', ''],
                                ['', '', '', '', '', '', '', '', ''],
                                ['', '', '', '', '', '', '', '', ''],
                                ['', '', '', '', '', '', '', '', ''],
                                ['', '', '', '', '', '', '', '', ''],
                                ['', '', '', '', '', '', '', '', ''],
                                ['', '', '', '', '', '', '', '', ''],
                                ['', '', '', '', '', '', '', '', '']])

        self.possible = '123456789'
        self.possibilities = np.full(self.grid.shape, '123456789')
        for row in range(9):
            for col in range(9):
                if self.grid[row][col] != '':
                    self.possibilities[row][col] = str(self.grid[row][col])
            

    def squareToRow(self, grid):
        return [[grid[row+row2][col+col2] for row2 in range(0,3,1) for col2 in range(0,3,1)] for row in range(0,9,3) for col in range(0,9,3)]

    def rowToSquare(self, grid):
        return [[grid[row+row2][col+col2] for row2 in range(0,3,1) for col2 in range(0,3,1)] for row in range(0,9,3) for col in range(0,9,3)]

    def getValues(self):
        values = {}
        squareGrid = self.squareToRow(self.grid)
        for square in range(9):
            for cell in range(9):
                values[str(cell)+str(square)] = squareGrid[square][cell]
        return values

    def solve(self):
        print('solving')

        previousPossibilities = []
        while previousPossibilities != str(self.possibilities):
            previousPossibilities = str(self.possibilities)
            self.applyRestrictions()
            self.display()
            
        if len(previousPossibilities) != 351:
            print('solved')
        else:
            print('cannot solve')
            
        self.grid = self.possibilities

    def applyRestrictions(self):
        self.removeRow()
        self.removeCol()
        self.removeSqu()

    def remove(self, pointers):
        for pointer in pointers:
            if len(self.possibilities[pointer[0]][pointer[1]]) == 1:
                for pointerRemove in pointers:
                    if pointerRemove != pointer:
                        self.possibilities[pointerRemove[0]][pointerRemove[1]] = self.possibilities[pointerRemove[0]][pointerRemove[1]].replace(self.possibilities[pointer[0]][pointer[1]], '')

    def insert(self, pointers):
        for number in range(9):
            sole = False
            for pointer in pointers:
                if str(number) in self.possibilities[pointer[0]][pointer[1]]:
                    if sole == False:
                        sole = pointer
                    else:
                        sole = False
                        break

            if sole != False:
                self.possibilities[sole[0]][sole[1]] = number

    def removeRow(self):
        for row in range(9):
            pointers = [[row, col] for col in range(9)]
            self.remove(pointers)
            self.insert(pointers)

    def removeCol(self):
        for col in range(9):
            pointers = [[row, col] for row in range(9)]
            self.remove(pointers)
            self.insert(pointers)

    def removeSqu(self):
        for row in range(0,9,3):
            for col in range(0,9,3):
                pointers = [[row+row2, col+col2] for col2 in range(0,3,1) for row2 in range(0,3,1)]
                self.remove(pointers)
                self.insert(pointers)

    def display(self):
        for row in self.possibilities:
            print([[row[col], row[col+1], row[col+2]] for col in range(0,9,3)])
        print()

s = sudoku('p')
s.solve()
