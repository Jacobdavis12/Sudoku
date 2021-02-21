import numpy as np
class sudoku():
    def __init__(self, rawData, SquareRowCount, SquareColCount, CellRowCount, CellColCount):
        self.squareRowCount = SquareRowCount
        self.squareColCount = SquareColCount
        self.cellRowCount = CellRowCount
        self.cellColCount = CellColCount

        #self.grid = np.asarray(self.rowToSquare([[rawData[str(col)+str(row)] for row in range(self.squareRowCount*self.squareColCount)] for col in range(self.squareRowCount*self.squareColCount)]))
        #self.grid = np.asarray([['', '', '7', '5', '1', '', '', '', '3'],
        #                        ['', '', '', '', '', '6', '4', '', '7'],
        #                        ['', '', '', '', '3', '', '', '', ''],
        #                        ['', '', '9', '', '', '', '', '7', '8'],
        #                        ['', '', '6', '', '8', '', '5', '', ''],
        #                        ['2', '4', '', '', '', '', '6', '', ''],
        #                        ['', '', '', '', '7', '', '', '', ''],
        #                        ['3', '', '2', '6', '', '', '', '', ''],
        #                        ['8', '', '', '', '4', '3', '2', '', '']])

        self.grid = np.asarray([['2', '', '', '', '5', '6', '', '', ''],
                                ['5', '', '', '9', '7', '4', '2', '', ''],
                                ['', '', '', '', '3', '', '', '8', ''],
                                ['1', '', '7', '', '', '2', '', '9', ''],
                                ['', '4', '', '1', '', '7', '', '5', ''],
                                ['', '3', '', '4', '', '', '1', '', '7'],
                                ['', '2', '', '', '4', '', '', '', ''],
                                ['', '', '9', '5', '1', '8', '', '', '2'],
                                ['', '', '', '7', '2', '', '', '', '9']])

        self.firstChange = []
        self.noOfSolutions = 1

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

    def getNoOfOtherSolutions(self):
        return self.noOfSolutions - 1

    def solve(self):
        print('Solving')
        
        self.display()
        self.applyRestrictions(self.possibilities)
        self.display()
            
        if self.isSolved():
            print('Solved')
        else:
            print('Back tracking')
            solutions = self.recursiveSolve(self.possibilities)
            self.possibilities = solutions[0]
            self.noOfSolutions = len(solutions)
            if self.firstChange == []:
                for row in range(self.squareRowCount*self.cellRowCount):
                    for col in range(self.squareColCount*self.cellColCount):
                        if self.possibilities[row][col] != self.grid[row][col]:
                            self.firstChange = [[row, col], self.possibilities[row][col]]
            
        self.grid = self.possibilities
        
    def applyRestrictions(self, possibilities):
        previousPossibilities = []
        
        while previousPossibilities != str(possibilities):
            previousPossibilities = str(possibilities)
            for restriction in self.getRestrictionPointers():
                for pointers in restriction:
                    possibilities = self.remove(pointers, possibilities)
                    possibilities = self.insert(pointers, possibilities)

        return possibilities

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

    def getRestrictionPointers(self):
        yield self.rowRestriction()
        yield self.colRestriction()
        yield self.squareRestriction()

    def check(self, grid = self.possibilities):
        for restriction in self.getRestrictionPointers():
            for pointers in restriction:
                inRestriction = ''
                for pointer in pointers:
                    if len(grid[pointer[0]][pointer[1]]) == 1:
                        if grid[pointer[0]][pointer[1]] in inRestriction:
                            return pointer
                        inRestriction += grid[pointer[0]][pointer[1]]
                    if len(grid[pointer[0]][pointer[1]]) == 0:
                        return pointer

        return True

    def getHint(self):
        self.solve()
            
        return self.firstChange

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
                            if self.firstChange == []:
                                self.firstChange = [pointer, pointers]
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

            if sole != False:
                if self.firstChange == []:
                    self.firstChange = [sole, pointers]
                possibilities[sole[0]][sole[1]] = str(number)

        return possibilities

    def rowRestriction(self):
        for row in range(self.squareRowCount*self.cellRowCount):
            yield [[row, col] for col in range(self.squareColCount*self.cellColCount)]

    def colRestriction(self):
        for col in range(self.squareColCount*self.cellColCount):
            yield [[row, col] for row in range(self.squareRowCount*self.cellRowCount)]

    def squareRestriction(self):
        for row in range(0,self.squareRowCount*self.cellRowCount,self.cellRowCount):
            for col in range(0,self.squareColCount*self.cellColCount,self.cellColCount):
                yield [[row+row2, col+col2] for col2 in range(0,self.cellColCount,1) for row2 in range(0,self.cellRowCount,1)]

    def display(self):
        for row in self.possibilities:
            print([[row[col+col2] for col2 in range(0,self.cellColCount,1)] for col in range(0,self.squareColCount*self.cellColCount,self.cellColCount)])
        print()

    def squareToRow(self, grid):
        return [[grid[row+row2][col+col2] for row2 in range(0,self.cellRowCount,1) for col2 in range(0,self.cellColCount,1)] for row in range(0,self.squareRowCount*self.cellRowCount,self.cellRowCount) for col in range(0,self.squareRowCount*self.cellColCount,self.cellColCount)]

    def rowToSquare(self, grid):
        return [[grid[row+row2][col+col2] for row2 in range(0,self.cellRowCount,1) for col2 in range(0,self.cellColCount,1)] for row in range(0,self.squareRowCount*self.cellRowCount,self.cellRowCount) for col in range(0,self.squareRowCount*self.cellColCount,self.cellColCount)]

class xSudoku(sudoku):
    def __init__(self, rawData):
        super(xSudoku, self).__init__(rawData, 3, 3, 3, 3)

    def getRestrictionPointers(self):
        yield from super(xSudoku, self).getRestrictionPointers()
        yield self.diamondRestriction()

    def diamondRestriction(self):
        yield [[dia, dia] for dia in range(0, 9)]
        yield [[8-dia, dia] for dia in range(0, 9)]


#s = sudoku('p', 3, 3, 3, 3)
#s.solve()

s = xSudoku('p')
s.solve()