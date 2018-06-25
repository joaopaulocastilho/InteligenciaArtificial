import numpy as np
import matplotlib.pyplot as plt
## VALORES DE INICIALIZACAO:
nRows = 5 #Numero de linhas do grid
nCols = 5 #Numero de colunas do grid
nIter = 100 #Numero de iteracoes do SOM
alpha0 = 0.3 #Taxa de aprendizado
sigma0 = max(nRows, nCols) #Area dos vizinhos do nodo vencedor
percTrain = 0.7 #Quantos % do conjunto de dados vao para o treinamento

## INICIALIZAR VALORES FUNDAMENTAIS
T = (nIter / np.log(sigma0))

## FUNCOES FUNDAMENTAIS PARA EXECUCAO DO SOM

#Calcula a distancia euclidiana
def distances(a, b):
     return np.sqrt(np.sum((a-b)**2,2, keepdims=True))

#Calcula a posicao do no vencedor
def positionNode(dist):
    arg = dist.argmin()
    md = dist.shape[1]
    return arg//md, arg%md

#Distancia topologica entre dois nodos
def distanceNodes(n1, n2):
    n1 = np.asarray(n1)
    n2 = np.asarray(n2)
    return np.sqrt(np.sum((n1-n2)**2))

def som(Xfull, grid):
    Xdata = Xfull[:,:-1] #Tirar a coluna dos rotulos dos dados
    n = grid.shape[0] #Numero de linhas do grid
    m = grid.shape[1] #Numero de colunas do grid
    dataSize = Xdata.shape[0] #Tamanho do dataset passado por parametro
    for epoch in range(nIter):
        print('Epoch: ', epoch + 1)
        #Atualizar os valores da taxa de aprendizado e area dos vizinhos
        alpha = alpha0 * np.exp(-epoch/T)
        sigma = sigma0 * np.exp(-epoch/T)

        for k in range(dataSize):
            #Pegar a distancia do nodo vencedor
            dists = distances(Xdata[k,:], grid) #Distancia da entrada para todos os nodos
            winPos = positionNode(dists) #Posicao do vencedor

            deltaW = 0
            h = 0
            for i in range(n):
                for j in range(m):
                    #Calcular a distancia entre o nodo vencedor e o nodo atual
                    dNode = distanceNodes([i, j], winPos)
                    #Influencia ate o nodo vencedor
                    h = np.exp((-dNode**2)/(2*sigma**2))
                    #Atualizar os pesos
                    deltaW = (alpha * h * (Xdata[k,:] - grid[i,j,:]))
                    grid[i,j,:] += deltaW
    return grid

def executaTeste(Xfull, grid):
     Xdata = Xfull[:,:-1] #Tirar a coluna dos rotulos dos dados
     n = grid.shape[0] #Numero de linhas do grid
     m = grid.shape[1] #Numero de colunas do grid
     dataSize = Xdata.shape[0] #Tamanho do dataset passado por parametro
     ocorrencias = np.zeros((nRows, nCols, 3)) #Quantas vezes um rotulo ocorreu em um nodo
     for k in range(dataSize):
          #Pegar a distancia do nodo vencedor
          dists = distances(Xdata[k,:], grid) #Distancia da entrada para todos os nodos
          winPos = positionNode(dists) #Posicao do vencedor
          #Salvar a ocorrencia no neuronio
          ocorrencias[winPos[0]][winPos[1]][int(Xfull[k][7] - 1)] += 1
     return ocorrencias


#Preparar o conjunto de dados
try:
    fdata = open('seeds_dataset.txt')
except:
    print('Nao foi possivel abrir o arquivo de dados!')
    exit(0)

data = np.array([d.split() for d in fdata.readlines()],dtype=float)
np.random.shuffle(data) #"Baguncando" as linhas da entrada

m = int(data.shape[0] * percTrain) #Tamanho do conjunto de Treinamento
Xtrain = data[:m,:] #Conjunto de Treinamento
Xtest = data[m:,:] #Conjunto de Testes

grid = np.random.uniform(-1, 1, [nRows, nCols, 7]) #Iniciamamos o grid (rede neural) com valores aleatorios

grid = som(Xtrain, grid)

#print(grid)

ocorrencias = executaTeste(Xtest, grid)
#print(ocorrencias)

plt.imshow(ocorrencias)
plt.show()
