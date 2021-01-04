import numpy as np
from PIL import Image

class network:
    def __init__(self, Dimensions, fileName = False):
        self.dimensions = Dimensions

        self.biases = [2*np.random.rand(y)-1 for y in Dimensions[1:]]
        self.weights = [2*np.random.rand(x, y)-1 for x, y in zip(Dimensions[1:], Dimensions[:-1])]
        self.biases  = np.copy(self.biases) * 0
        self.weights  = np.copy(self.weights) * 0

    def loadNet(self, filename = 'network.npz'):
        data = np.load(filename, allow_pickle=True)
        self.biases = data['bias']
        self.weights = data['weight']

    def saveNet(self, filename = 'network.npz'):
        np.savez(filename, bias = self.biases, weight = self.weights)

    def train(self, learningRate, td):
        batch = 1
        while batch > 0:
            try:
                print(batch, ':', nt.test(5000, td))
                print('netCost:', self.SDG(10, td, learningRate))
                batch += 1
            except KeyboardInterrupt:
                if input('continue?: ') == 'n':
                    batch = 0

    def run(self, activation):
        activations = [activation]
        for weight, bias in zip(self.weights, self.biases):
            activation = self.sigmoid(np.dot(weight, activation)+bias)
            activations.append(activation)
        return activations

    def SDG(self, size, td, lr):
        td.shuffleImages()
        netCost = 0

        for i in range(0, len(td.images)-size +2, size):
            biasesChange = np.copy(self.biases) * 0
            weightsChange = np.copy(self.weights) * 0

            for j in range(size):
                imageIndex = i + j
                wanted = np.zeros(10)
                wanted[td.images[imageIndex][1]] = 1
                activations = self.run(td.images[imageIndex][0])
                #c = self.cost(activations[-1], wanted)
                #netCost += c
                netCost += self.cost(activations[-1], wanted)

                weightsChange, biasesChange = self.backProp(activations, wanted, weightsChange, biasesChange)
            #input(c)

            self.weights -= weightsChange * lr / size
            self.biases -= biasesChange * lr / size

        return netCost/len(td.images)

    def backProp(self, activations, wanted, weightsChange, biasesChange):
        dcda = self.costPrime(activations[-1], wanted)
        
        for i in range(1, len(self.dimensions)):
            dcdz = dcda * self.sigmoidPrime(activations[-i])
            weightsChange[-i] += np.outer(dcdz, activations[-i-1])
            biasesChange[-i] += dcdz
            
            dcda = np.dot(self.weights[-i], dcdz)

        return weightsChange, biasesChange

    def test(self, size, td):
        td.shuffleTestImages()
        correct = 0
        for imageIndex in range(size-1):
            activations = self.run(td.testImages[imageIndex][0])[-1]
            if max(activations) == activations[td.testImages[imageIndex][1]]:
                correct += 1
        return correct

    #Propogation functions
    def sigmoid(self, matrix):
        return 1/(1+np.exp(-matrix))

    def sigmoidPrime(self, sx):
        return sx*(1-sx)

    def cost(self, activation, wanted):
        return sum((activation - wanted)**2)/self.dimensions[-1]

    def costPrime(self, activation, wanted):
        return 2*(activation - wanted)

class trainingData:
    
    def __init__(self):
        data = np.load('ds.npy', allow_pickle=True)[1:]
        #np.random.shuffle(data)
        self.images = data[:-5000]
        print(len(self.images))
        self.testImages = data[-5000:]


    def shuffleImages(self):
        pass
        #np.random.shuffle(self.images)

    def shuffleTestImages(self):
        pass
        #np.random.shuffle(self.testImages)

nt = network([784, 30, 30, 10])
nt.loadNet()
td = trainingData()
print('intialised')

#print(nt.run(td.images[0][0]), td.images[0][1])

#for img in range(100):
#    print(nt.run(td.images[img][0])[-1])
#    imageFromPixles = Image.fromarray(np.uint8([255*td.images[img][0][i:i+28] for i in range(0,784,28)]))
#    imageFromPixles.save('trala.jpg')
#    input(td.images[img][1])
print(nt.test(100, td))

nt.train(1, td)
#print(nt.run(td.images[0][0]), td.images[0][1])
print(nt.test(1000, td))
print(nt.test(1000, td))
#nt.saveNet()
