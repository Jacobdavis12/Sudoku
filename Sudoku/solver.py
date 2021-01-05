class sudoku():
    def __init__(self, rawData):
        self.rawValues = rawData
        self.grid = self.rowToSquare([[rawData[str(row)+str(col)] for row in range(9)] for col in range(9)])

        self.possible = '123456789'
        self.possibilities = np.empty(self.grid.shape)
        for row in range(9):
            for col in range(9):
                if self.grid[row][col] == 0:
                    self.possibilities[row][col] = str(possible)
                else:
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
        print('solved')
        self.applyRestrictions()

    def applyRestrictions(self):
        self.singularities()

    def singularities(self):
        for row in range(9):
            for col in range(9):
                if len(self.possibilities[row][col]) == 1:
                    self.removeFromRow()
                    self.removeFromCol()
                    self.removeFromSqu()


    def removeFromRow(self):
        for row in range(9):
            for col in range(9):
                if len(self.possibilities[row][col]) == 1:
                    for colRemove in range(9):
                        if colRemove != col:
                            self.possibilities[row][colRemove] = self.possibilities[row][colRemove].replace(self.possibilities[row][col],'')



