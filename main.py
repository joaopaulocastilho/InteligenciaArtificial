import numpy as np
import math
import scipy.spatial.distance as dist

nRows = 3
nCols = 3
percTest = 0.7
niter = 30
alpha0 = 0.4
sigma0 = max(nRows, nCols)
T = niter/math.log(sigma0)

def kohonen(alpha, sigma, Xfull, grid, flag):
    it1 = int(grid.shape[0])
    it2 = int(grid.shape[1])
    Xtest = Xfull[:,:-1]
    ocorr = np.zeros((nRows, nCols, 3))
    for epoch in range(niter):
        print(epoch)
        for i in range(Xtest.shape[0]):
            menor = 9999999999999999999999
            for j in range(it1):
                for k in range(it2):
                    tmp = dist.euclidean(Xtest[i], grid[j][k])
                    if (tmp < menor):
                        menor = tmp
                        mline = j
                        mcol = k
            if flag:
                ocorr[mline][mcol][int(Xfull[i][7] - 1)] = ocorr[mline][mcol][int(Xfull[i][7] - 1)] + 1
            #print (menor)
            sigma = sigma0 * np.exp(((epoch/T)*-1))
            alpha = alpha0 * np.exp(((epoch/T)*-1))
            for j in range(it1):
                for k in range(it2):
                    dist_manh = abs(mline - j) + abs(mcol - k)
                    #print(dist_manh)
                    h = np.exp(((dist_manh/(2*sigma)) * -1))
                    tmp = 0
                    deltaW = alpha * h * (Xtest[i,:] - grid[j,k,:])
                    #print(deltaW.shape, grid.shape)
                    grid[j,k,:] += deltaW
    return grid, ocorr
try:
    fdata = open('seeds_dataset.txt')
except:
    print('Nao deu pra abrir, meu chapa!')
    exit(0)

data = np.array([d.split() for d in fdata.readlines()],dtype=float)
np.random.shuffle(data)

m = int(data.shape[0] * percTest)

Xtrain = data[:m,:]
Xtest = data[m:,:]

print(Xtest.shape, Xtrain.shape)

grid = np.random.rand(nRows, nCols, 7)

sigma = sigma0
alpha = alpha0

grid, ocorr = kohonen(alpha, sigma, Xtrain, grid, 0)

grid, ocorr = kohonen(alpha, sigma, Xtest, grid, 1)

print("Ocorrencias: ", ocorr)
print(grid.shape, grid)
