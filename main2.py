import numpy as np
import matplotlib.pyplot as plt
## VALORES DE INICIALIZACAO:
DEBUG = 0
nRows = 6 #Numero de linhas do grid
nCols = 6 #Numero de colunas do grid
nIter = 100 #Numero de iteracoes do SOM
alpha0 =0.3 #Taxa de aprendizado
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
        print('Executando Epoca: ', epoch + 1)
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

def analysis(ocorrencias):
     neuroInfo = np.zeros((nRows, nCols, 3))
     error = np.zeros(3)
     seeds = np.zeros(3)
     for i in range(nRows):
          for j in range(nCols):
               seeds += ocorrencias[i][j]
               tmp = np.argmax(ocorrencias[i][j])
               neuroInfo[i][j][0] = tmp #Semente que mais ocorre no neuronio
               neuroInfo[i][j][1] = np.sum(ocorrencias[i][j]) #Total de sementes
               neuroInfo[i][j][2] = neuroInfo[i][j][1] - ocorrencias[i][j][tmp] #Total de erros
               for k in range(3):
                    if k != tmp and ocorrencias[i][j][k] == ocorrencias[i][j][tmp]: #Em caso de empate
                         neuroInfo[i][j][2] = neuroInfo[i][j][1] #A quantidade de sementes erradas eh igual o total
                         neuroInfo[i][j][0] = -1 #Marca a semente predominante como -1
                         error[tmp] += ocorrencias[i][j][tmp]
                    if k != tmp:
                         error[k] += ocorrencias[i][j][k]
     return neuroInfo, error, seeds
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
ocorrencias = executaTeste(Xtest, grid)

## A partir de agora vamos calcular a taxa de acerto de cada neuronio

neuroInfo, error, seeds = analysis(ocorrencias)

if DEBUG == 1:
     print(ocorrencias)
     print("--------------")
     print(neuroInfo)
     print(error, seeds)

print("\n\n")
print("Taxa de acerto para sementes de tipo 1: {}%".format(round(((seeds[0]-error[0])/seeds[0]) * 100, 2)))
print("Taxa de acerto para sementes de tipo 2: {}%".format(round(((seeds[1]-error[1])/seeds[1]) * 100, 2)))
print("Taxa de acerto para sementes de tipo 3: {}%".format(round(((seeds[2]-error[2])/seeds[2]) * 100, 2)))

print("\n\nTaxa de acerto por neuronio:")
for i in range(nRows):
     for j in range(nCols):
          if neuroInfo[i][j][0] == -1:
               print("Neuronio [{},{}] descartado.".format(i, j))
          else:
               tmp = ((neuroInfo[i][j][1] - neuroInfo[i][j][2]) / neuroInfo[i][j][1]) * 100
               print("Neuronio [{},{}] (tipo {}): {}%".format(i, j, int(neuroInfo[i][j][0] + 1), round(tmp, 2)))

## Plotar grafico
plt.imshow(ocorrencias)
plt.show()
