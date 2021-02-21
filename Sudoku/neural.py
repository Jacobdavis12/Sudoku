import numpy as np
from PIL import Image

class network:
    def __init__(self, Dimensions, fileName = False):
        self.dimensions = Dimensions

        self.biases = [2*np.random.rand(y)-1 for y in Dimensions[1:]]
        self.weights = [2*np.random.rand(x, y)-1 for x, y in zip(Dimensions[1:], Dimensions[:-1])]
        #self.biases  = np.copy(self.biases) * 0
        #self.weights  = np.copy(self.weights) * 0

    def loadNet(self, filename = 'network.npz'):
        data = np.load(filename, allow_pickle=True)
        self.biases = data['bias']
        self.weights = data['weight']
        self.dimensions = [784] + [len(i) for i in self.weights]

    def saveNet(self, filename = 'network.npz'):
        np.savez(filename, bias = self.biases, weight = self.weights)

    def train(self, learningRate, td):
        batch = 1
        while batch > 0:
            try:
                print(batch, ':', nt.test(2000, td))
                print('netCost:', self.SDG(50, td, learningRate))
                batch += 1
            except KeyboardInterrupt:
                if input('continue?: ') == 'n':
                    batch = 0
                    #d = self.demo(td)
                    #print(d)
                    #print([d[i][i]/sum([d[i][j] for j in d[i]]) for i in d])

    def run(self, activation):
        activations = [activation]
        for weight, bias in zip(self.weights, self.biases):
            activation = self.sigmoid(np.dot(weight, activation)+bias)
            activations.append(activation)
        return activations

    def SDG(self, size, td, lr):
        td.shuffleImages()
        netCost = 0
        #batchSize = len(td.images)//size

        for i in range(size, len(td.images), size):
            biasesChange = np.copy(self.biases) * 0
            weightsChange = np.copy(self.weights) * 0

            for j in range(size):
                imageIndex = i + j - size
                wanted = np.zeros(10)
                wanted[td.images[imageIndex][1]] = 1
                activations = self.run(td.images[imageIndex][0])
                c = self.cost(activations[-1], wanted)
                netCost += c
                #netCost += self.cost(activations[-1], wanted)

                weightsChange, biasesChange = self.backProp(activations, wanted, weightsChange, biasesChange)
            #input(c)

            self.weights = self.weights - weightsChange * lr / size
            self.biases -= self.biases - biasesChange * lr / size

        return netCost/len(td.images)

    def backProp(self, activations, wanted, weightsChange, biasesChange):
        dcda = self.costPrime(activations[-1], wanted)
        
        for i in range(1, len(self.dimensions)):
            dcdz = dcda * self.sigmoidPrime(activations[-i])
            weightsChange[-i] += np.outer(dcdz, activations[-i-1])
            biasesChange[-i] += dcdz
            
            dcda = np.dot(self.weights[-i].T, dcdz)

        return weightsChange, biasesChange

    def test(self, size, td):
        td.shuffleTestImages()
        correct = 0
        for imageIndex in range(size-1):
            activations = self.run(td.testImages[imageIndex][0])[-1]

            if np.argmax(activations) == td.testImages[imageIndex][1]:# and np.max(activations[np.argmax(activations)+1:]) != np.max(activations):
                correct+=1
            #else:
            #    print(td.testImages[imageIndex][1], imageIndex)
            #    imageFromPixles = Image.fromarray(np.uint8([td.testImages[imageIndex][0][i:i+28]*255 for i in range(0,784,28)]))

            #    imageFromPixles.show()
            #    print()

        return correct*10000//size/100

    def demo(self, td):
        results = {0: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}, 1: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}, 2: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}, 3: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}, 4: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}, 5: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}, 6: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}, 7: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}, 8: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}, 9: {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}}
        for imageI in td.images:
            results[imageI[1]][np.argmax(self.run(imageI[0])[-1])] +=1
        return results

    #Propogation functions
    def sigmoid(self, matrix):
        return 1/(1+np.exp(-matrix))

    def sigmoidPrime(self, sx):
        return sx*(1-sx)

    def cost(self, activation, wanted):
        return sum((activation - wanted)**2)/self.dimensions[-1]/2

    def costPrime(self, activation, wanted):
        return (activation - wanted)

class trainingData:
    
    def __init__(self, fileName):
        data = np.load(fileName, allow_pickle=True)[1:]
        #np.random.shuffle(data)
        self.images = data[:-2000]
        print(len(self.images))
        self.testImages = data[-2000:]


    def shuffleImages(self):
        np.random.shuffle(self.images)

    def shuffleTestImages(self):
        np.random.shuffle(self.testImages)

nt = network([784, 100, 50, 30, 10])
nt.loadNet('network.npz')
td = trainingData('ds.npy')
print('intialised')

print(nt.run(td.images[0][0]), td.images[0][1])#np.zeros(td.images[0][0].shape)))#

##for img in range(0, 1000, 1):
##    if np.argmax(nt.run(td.images[img][0])[-1]) != td.images[img][1]:
##        print(nt.run(td.images[img][0])[-1])
##        imageFromPixles = Image.fromarray(np.uint8([td.images[img][0][i:i+28]*255 for i in range(0,784,28)]))
##        imageFromPixles.show()
##        print(np.argmax(nt.run(td.images[img][0])[-1]))
##
##        input(td.images[img][1])
print(nt.test(100, td))


nt.train(3, td)
#print(nt.run(td.images[0][0]), td.images[0][1])
print(nt.test(1000, td))
print(nt.test(1000, td))
nt.saveNet('network.npz')

