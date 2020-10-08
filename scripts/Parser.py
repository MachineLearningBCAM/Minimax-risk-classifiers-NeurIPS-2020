import numpy as np


# Load the data in csv format
# Each row contains an instance (case)
# The values included in each row are separated by a string given by the parameter sep, e.g., ","
# Each column corresponds to the values of a (discrete) random variable
# name (string): file name containing the data
# sep (string): separates the different values of the data
# return numpy.array[instances, vars]
def loadCsv(name, sep, numInter=3, maxDiscVals=5):
    text = np.loadtxt(name, np.str, delimiter=sep)
    (N, n) = text.shape
    # Determine the nature of the variables (int or float)
    disc = [True for i in range(n)]
    for i in range(n):

        if str.isalpha(text[0, i]) or str.isdigit(
                text[0, i]):  # if all characters in the string are alphabetic or there is at least one character
            vals = np.unique(text[:, i])
            if vals.size > maxDiscVals:
                try:
                    for v in vals: np.float(v)
                    disc[i] = False
                except:
                    disc[i] = True
        else:
            try:
                np.float(text[0, i])
                disc[i] = False
            except:
                disc[i] = True

    data = [[] for i in range(n)]
    for i in range(n):
        if disc[i]:
            data[i] = np.unique(text[:, i], return_inverse=True)[1]
        else:
            varData = [np.float(x) for x in text[:, i]]
            if numInter != None:  # Discretize with equal frequency
                ordered = np.sort(varData)
                cut = [ordered[(j + 1) * N / numInter - 1] for j in range(numInter)]
                cut[numInter - 1] = ordered[N - 1]
                data[i] = [0 for j in range(N)]
                for j in range(N):
                    for k in range(numInter):
                        if varData[j] <= cut[k]:
                            break
                    data[i][j] = k
            else:  # Not discretize
                data[i] = varData

    return np.transpose(data)


def saveAsCSV(name, D, delimiter=','):
    np.savetxt(name, D, delimiter=delimiter, fmt='%1d')


def card(D):
    return np.max(D, 1) + 1


def sufficientTreeStatistics(r, D):
    N = [[np.zeros(shape=(r[u], r[v]), dtype=np.int) for u in range(len(r))] for v in range(len(r))]
    for x in D:
        for u in range(len(r)):
            for v in range(len(r)):
                N[u][v][x[u], x[v]] += 1

    return N


def sufficient3statistics(r, D):
    N = [[[np.zeros(shape=(r[u], r[v], r[w]), dtype=np.int) for w in range(len(r))] for u in range(len(r))] for v in
         range(len(r))]
    for x in D:
        for u in range(len(r)):
            for v in range(len(r)):
                for w in range(len(r)):
                    N[u][v][w][x[u], x[v], x[w]] += 1

    return N


def checkStatistics(stat, r, D):
    n = len(D)
    for a in stat:
        for s in a:
            if np.sum(s) != n:
                return False
    return True


def indToInst(ind, card):
    inst = np.zeros((len(card),))
    w = 1
    res = ind
    for i in range(len(card)):
        inst[i] = res % card[i]
        res = res / card[i]
        w *= card[i]
    return inst


def instToInd(inst, card):
    ind = inst[0]
    w = card[0]
    for i in range(1, len(card)):
        ind += w * inst[i]
        w *= card[i]
    return ind


'''
Generate a stratified partition of the data of (almost) the same size

indC: index of the class variable, int
D: data set, np.array((instances,variables))
k: number of partitions
'''


def stratifiedPartitions(indC, D, k, seed=None):
    if seed != None:
        np.random.seed(seed)

    C = D[:, indC]
    valsC, Nc = np.unique(C, return_counts=True)
    ordC = np.argsort(C)

    # Get the randomized indices to instances of each class
    indc = list()
    ini = 0
    for i, c in enumerate(valsC):
        indc.append(np.random.permutation(ordC[ini:(ini + Nc[i])]))
        ini += Nc[i]

    # Get the chunks
    deltaC = Nc / np.float(k)
    Dk = list()
    for j in range(k):
        Dk.append(np.row_stack(
            (D[indc[c][np.arange(int(deltaC[c] * j), int(deltaC[c] * (j + 1)))], :] for c in range(valsC.size))))

    return Dk


def stratifiedHoldoutTrainTest(indC, D, percTest=0.3, seed=None):
    if seed != None:
        np.random.seed(seed)

    C = D[:, indC]
    valsC, Nc = np.unique(C, return_counts=True)
    ordC = np.argsort(C)

    # Get the randomized indices to instances of each class
    indc = list()
    ini = 0
    for i, c in enumerate(valsC):
        indc.append(np.random.permutation(ordC[ini:(ini + Nc[i])]))
        ini += Nc[i]

    # Get the chunks
    Test = np.row_stack((D[indc[c][np.arange(int(Nc[c] * percTest))], :] for c in range(valsC.size)))
    Train = np.row_stack((D[indc[c][np.arange(int(Nc[c] * percTest), Nc[c])], :] for c in range(valsC.size)))

    return (Train, Test)


def stratifiedCVtrainingTest(indC, D, k, seed=None):
    if seed != None:
        np.random.seed(seed)

    Test = stratifiedPartitions(indC, D, k)

    Training = [np.row_stack((Test[j] for j in range(k) if j != i)) for i in range(k)]

    return (Training, Test)


####################
# Weak Supervision #
####################

'''
Creates a weak supervised dataset given in term of bags with label proportions
D: fully supervised data
indC: index of the class variable. If None is the last variable
minSizeBag: minimum size of a bag
maxSizeBag: maximum size of a bag
numBags: number of bags. If None a partition of the data is returned, else
each bag is obtained from a random set of indexes
seed: random seed

return (Bags,Props) where
Bags: is a set of Bags of instances given in terms of
the values of the predictor variables, list(np.array)
Props: is the proportion of the class labels in each bag, np.array(int)
'''


def createBagsWithProps(D, indC=None, minSizeBag=2, maxSizeBag=10, numBags=None, seed=None):
    if seed != None:
        np.random.seed(seed)

    (N, d) = D.shape

    if indC == None:
        indC = d - 1
    cardC = np.max(D[:, indC]) + 1

    Dc = D[:, indC]
    if indC == d - 1:
        Dx = D[:, range(indC)]
    else:
        Dx = np.hstack((D[:, range(indC)], D[:, range(indC + 1, d)]))

    if numBags == None:
        # Partition of the dataset
        indx = np.random.permutation(N)
        total = 0
        bagSize = list()
        while total < N:
            # Crea empty arrays
            bagSize.append(np.random.choice(
                range(np.min([maxSizeBag + 1, N - total, minSizeBag]), np.min([maxSizeBag + 1, N - total + 1]))))
            total += bagSize[-1]

        ini = 0
        Bags = list()
        Props = list()
        for s in bagSize:
            Bags.append(Dx[indx[range(ini, ini + s)], :])
            (c, Nc) = np.unique(Dc[indx[range(ini, ini + s)]], return_counts=True)
            prp = np.zeros(cardC, dtype=np.int)
            prp[c] = Nc
            Props.append(prp)

            ini += s
    else:
        Bags = list()
        Props = list()
        for b in range(numBags):
            indx = np.random.choice(N, np.random.choice(range(minSizeBag, maxSizeBag + 1)))
            Bags.append(Dx[indx, :])
            (c, Nc) = np.unique(Dc[indx], return_counts=True)
            prp = np.zeros(cardC, dtype=np.int)
            prp[c] = Nc
            Props.append(prp)

    return (Bags, Props)


def fromBagsToWeights(Bags, Props):
    '''
    Transform a dataset given in terms of Bags with label proportions into
    a dataset given in terms of replicated fully supervised instances with
    weights corresponding to the labels proportions of the associated class

    Return (D,w) where
    D: fully labeled data, with the class variable in the last column, np.array(instance,variable)
    '''

    cardC = len(Props[0])

    D = list()
    w = list()
    for ind, B in enumerate(Bags):
        for c in range(cardC):
            N = np.float(np.sum(Props[ind]))
            if Props[ind][c] > 0:
                D.append(np.hstack((B, c * np.ones(shape=(B.shape[0], 1), dtype=np.int))))
                w.append(Props[ind][c] / N * np.ones(shape=(B.shape[0], 1)))

    D = np.vstack(D)
    w = np.vstack(w)

    return (D, w)


def fromBagsToProbs(Bags, Props):
    '''
    Transform a dataset given in terms of Bags with label proportions into
    a dataset given in terms of probabilistic instances with class probabilities

    Return (Dx,pc) where
    Dx: unlabeled data, np.array(instance,variable)
    pc: probability of the class, npo.array(instance,class label)
    '''

    cardC = len(Props[0])

    Dx = np.vstack(Bags)
    pc = list()
    for ind, B in enumerate(Bags):
        N = np.float(np.sum(Props[ind]))
        p = np.hstack([Props[ind][c] / N * np.ones(shape=(B.shape[0], 1)) for c in range(cardC)])
        pc.append(p)

    pc = np.vstack(pc)

    return (Dx, pc)


def supervisedWTraining(D):
    return (np.ones(D.shape[0]), D)


def missingWTraining(M, ind, card):
    n = M.shape[0]
    D = np.vstack([np.hstack((M[:ind], np.ones(n) * x, M[ind:])) for x in range(card)])

    return (np.ones(n * card) / float(card), D)


def corruptedWTraining(C, rho, ind, card):
    if rho == 0:
        return supervisedWTraining(C)

    n = C.shape[0]
    D = np.vstack([np.hstack((C[:ind], np.ones(n) * x, C[ind:])) for x in range(card)])
    w = np.zeros(n * card)
    for i in range(n):
        for x in range(card):
            if x == D[i, ind]:
                w[i + x * n] = 1 - rho
            else:
                w[i + x * n] = rho / (card - 1.0)

    return (w, D)


def multilabelWTraining(U, L):
    n = U.shape[0]

    D = np.vstack([np.hstack((U[i, :], np.array([l])) for i in range(n) for l in L[i])])
    w = np.array([1.0 / len(L[i]) for i in range(n) for l in L[i]])

    return (w, D)


def missingTransform(D, ind):
    if ind == D.shape[1] - 1:
        return np.array(D[:, :ind])
    elif ind == 0:
        return np.array(D[:, ind + 1:])
    else:
        return np.hstack((D[:, ind], D[:, ind + 1:]))


def corruptedTransform(D, ind, rho, card):
    '''
    D: data, np.array
    ind: index of the corrupted feature
    rho: probability of corrupting each instance
    card: cardinality of the corrupted feature
    '''
    C = np.array(D)
    (n, d) = C.shape
    for i in range(n):
        rand = np.random.uniform()
        if rand < rho:
            C[i, ind] = np.random.choice([c for c in range(card) if c != D[i, ind]])

    return C


def multilabelTransform(D, alpha, beta, card):
    (n, d) = D.shape
    U = D[:, :(d - 1)]

    L = list()
    for i in range(n):
        S = list()
        for c in range(card):
            rand = np.random.uniform()
            if c == D[i, -1]:
                if rand < alpha:
                    S.append(c)
            elif rand < beta:
                S.append(c)
        L.append(S)

    return (U, L)