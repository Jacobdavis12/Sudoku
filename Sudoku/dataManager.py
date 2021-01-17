import numpy as np
from PIL import Image

def display(pixles):
    imageFromPixles = Image.fromarray(np.uint8([pixles[i:i+28]*255 for i in range(0,784,28)]))
    imageFromPixles.show()
    
#ds = np.load('dataSet.npy', allow_pickle=True)[1:]
#imageFromPixles = Image.fromarray(np.uint8([ds[0][0][i:i+28]*255 for i in range(0,784,28)]))
#imageFromPixles.show()

#dsNew = np.array([[2,1]])
#erroreneus = [8, 42, 43, 140, 141, 146, 153, 161, 166, 167, 170, 178, 180, 183, 184, 186, 187, 192, 193, 198, 207, 241, 242, 339, 340, 345, 352, 360, 365, 366, 369, 377, 379, 382, 383, 385, 386, 391, 392, 397]
#last = 0

#for e in erroreneus:
#    print(e)
#    #imageFromPixles = Image.fromarray(np.uint8([ds[e*81][0][i:i+28]*255 for i in range(0,784,28)]))
#    #imageFromPixles.show()
#    #imageFromPixles = Image.fromarray(np.uint8([ds[last*81][0][i:i+28]*255 for i in range(0,784,28)]))
#    #imageFromPixles.show()
#    dsNew = np.append(dsNew, ds[last*81:e*81], axis = 0)
#    last = int(e+1)

#dsNew = dsNew[len(dsNew)//2:]
#for g in range(len(dsNew)):
#    imageFromPixles = Image.fromarray(np.uint8([dsNew[g][0][i:i+28]*255 for i in range(0,784,28)]))
#    imageFromPixles.save('testData/' + str(g) + '.jpg')

#np.save('ds.npy', dsNew)

##y = [i[0] for i in ds]
##data = np.load('network.npz', allow_pickle=True)
##biases = data['bias']
##weights = data['weight']

##Mnist:
#def loadLabels():
#    with open('train-labels.idx1-ubyte', 'rb') as f:
#        content = f.read()[8:60008]
#    return list(content)

#images = np.load('imageFile.npy', allow_pickle=True)
#labels = loadLabels()
##display(images[0])

#ds = []
#for i in range(len(images)):
#    ds.append(np.asarray([images[i], labels[i]]))

#ds = np.asarray(ds)

#np.save('mnistds.npy', ds)

##Remove Blanks
#ds = np.load('ds.npy', allow_pickle=True)

#dsNew = []
#for i in ds:
#    if i[1] != 0:
#        dsNew.append(i)

#np.save('ds.npy', np.asarray(dsNew))

#invert
ds = np.load('ds.npy', allow_pickle=True)

for i in range(len(ds)):
    ds[i] = np.asarray([1-ds[i][0], ds[i][1]])

np.save('ds.npy', np.asarray(ds))
