import numpy as np
from PIL import Image

def display(pixles):
    imageFromPixles = Image.fromarray(np.uint8([pixles[i:i+28]*255 for i in range(0,784,28)]))
    imageFromPixles.show()
    
ds = np.load('mnistds.npy', allow_pickle=True)[1:]
imageFromPixles = Image.fromarray(np.uint8([ds[0][0][i:i+28]*255 for i in range(0,784,28)]))
imageFromPixles.show()
##for i in range(len(ds)):
##    print(ds[i][1])
##    if ds[i][1] != 0:
##        display(ds[i][0])
##        input()

for i in range(len(ds)):
    if ds[i][1] == 0:
        ds[i][0] = np.zeros(ds[i][0].shape)
        
    
np.save('mnist0sds.npy', ds)
##
##erroreneus = [162, 164, 170, 171, 172, 177, 180,185, 187, 188,190, 193, 196, 199]
##last = 0
##
##for e in erroreneus:
##    print(e)
##    #imageFromPixles = Image.fromarray(np.uint8([ds[e*81][0][i:i+28]*255 for i in range(0,784,28)]))
##    #imageFromPixles.show()
##    #imageFromPixles = Image.fromarray(np.uint8([ds[last*81][0][i:i+28]*255 for i in range(0,784,28)]))
##    #imageFromPixles.show()
##    dsNew = np.append(dsNew, ds[last*81:e*81], axis = 0)
##    last = int(e+1)

#dsNew = dsNew[len(dsNew)//2:]
#for g in range(len(dsNew)):
#    imageFromPixles = Image.fromarray(np.uint8([dsNew[g][0][i:i+28]*255 for i in range(0,784,28)]))
#    imageFromPixles.save('testData/' + str(g) + '.jpg')

##np.save('ds.npy', dsNew)

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

##invert
#ds = np.load('ds.npy', allow_pickle=True)

#for i in range(len(ds)):
#    ds[i] = np.asarray([1-ds[i][0], ds[i][1]])

#np.save('ds.npy', np.asarray(ds))
