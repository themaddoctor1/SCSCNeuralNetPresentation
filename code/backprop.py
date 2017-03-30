from sys import argv as args

import random
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import math
import random

# TRAINING DATA
# =============

data = []
X = []
Y = []
Z = []
alpha = 1 # Learning rate

# User interface stuff
delta = 0.05
grid_x = np.arange(-0.5, 1.5, delta)
grid_y = np.arange(-0.5, 1.5, delta)
grid_X, grid_Y = np.meshgrid(grid_x, grid_y)

def andSetup():
    print("Loading the AND demo")
    for pair in [[0, 0, 0],[0, 1, 0], [1, 0, 0], [1, 1, 1]]:
        data.append(pair)
        X.append(pair[0])
        Y.append(pair[1])
        Z.append(pair[2])

def xorSetup():
    for pair in [[0, 0, 0],[0, 1, 1], [1, 0, 1], [1, 1, 0]]:
        data.append(pair)
        X.append(pair[0])
        Y.append(pair[1])
        Z.append(pair[2])

"""
Creates a random setup with the given number of points
"""
def randSetup(points, mutex):
    print("Loading the random demo")
    for i in range(points):
        x = random.random()
        y = random.random()
        z = 1 if random.random() > 0.5 else 0
        if z == 1:
            x = 1 - x / 2.0**mutex
        else:
            x /= 2.0**mutex
        print("(" + str(x) + "," + str(y) + ") -> " + str(z))
        X.append(x)
        Y.append(y)
        Z.append(z)
        data.append([x, y, z])
    alpha = 0.5 / points




# MATRIX IMPLEMENTATION
# =====================
def mul(A, B):
    C = []
    for i in range(len(A)):
        C.append([])
        for j in range(len(B[0])):
            v = 0
            for k in range(len(B)):
                v += A[i][k] * B[k][j]
            C[i].append(v)
    return C

def add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[i]))] for i in range(len(A))]

def sub(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[i]))] for i in range(len(A))]

def mulByConst(A, c):
    return [[v * c for v in A[i]] for i in range(len(A))]

def transpose(M):
    return [[M[i][j] for i in range(len(M))] for j in range(len(M[0]))]

# Linear function
# x - Input vector
# d - Whether function is non-derivative (same for other functions)
def linear(x, d):
    if d:
        return [v for v in x]
    else:
        return [[1 if i == j else 0 for i in range(len(x))] for j in range(len(x))]

# Sigmoid function
def sigmoid(x, d):
    if d:
        return [[1 / (1 + math.exp(-v[0]))] for v in x]
    else:
        s = sigmoid(x, True)
        return [[s[i][0] * (1 - s[i][0]) if i == j else 0 for j in range(len(x))] for i in range(len(x))]

# Rectified linear unit function
def relu(x, d):
    if d:
        return [[x[i][0] if x[i][0] > 0 else 0] for i in range(len(x))]
    else:   
        return [[(1 if x[i][0] > 0 else 0) if i == j else 0 for j in range(len(x))] for i in range(len(x))]

def unitStep(x, d):
    if d:
        return [[1 if x[i][0] > 0 else 0] for i in range(len(x))]
    else:
        return [[1 if i == j else 0 for j in range(len(x))] for i in range(len(x))]

def computeLayer(W, x, f):
    """Computes network output
    W - Weight matrix
    x - Input vector
    f - Activation function

    return - f(W*x)
    """
    s = [[0] for i in range(len(W))]
    
    return f(mul(W, x), True)

def computeNet(Ws, x, fs):
    """Computes the output of a neural network.
    Ws - A set of weight matrices (the weights of each layer)
    x  - The input vector
    fs - The activation functions
    """
    
    y = x
    for i in range(len(Ws)):
        y = computeLayer(Ws[i], y, fs[i])

    return y

alpha = 0.1

def trainOnPair(Ws, fs, x, t):
    # Step one: Compute network output (feed-forward).
    xs = [x]
    ss = []
    for i in range(len(Ws)):
        # Compute the linear combination, then the function of the combo.
        ss.append(mul(Ws[i], xs[-1]))
        xs.append(fs[i](ss[-1], True))

    # Step two: Compute derivatives.
    # Part a: Derivative of error (y - t).
    dE = sub(xs[-1], t)

    # Part b: Gradients of functions
    dfs = [fs[i](ss[i], False) for i in range(len(Ws))]

    # Part c: Gradients of linear combinations
    dss = [transpose(W) for W in Ws]

    # Part d: Derivative of neurons as function of weights
    dws = [transpose(x) for x in xs[:-1]]

    #Step three: Apply derivatives
    error = mul(dfs[-1], mulByConst(dE, alpha))

    for i in range(len(Ws)-1, -1, -1):
        # Translate error (currently dE/dy) to dE/ds

        # Apply the new error to the matrix
        dW = mul(error, dws[i])
        Ws[i] = sub(Ws[i], dW)

        # If we are not on the first layer, we need to keep propagating.
        if i > 0:
            error = mul(mul(dfs[i-1], dws[i]), error)

# GRAPH PLOTTING
# ==============
plt.get_current_fig_manager().full_screen_toggle()

def resetPlot(Ws, fs):

    plt.clf()

    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    
    plt.autoscale(False)
    
    my_Z = [[computeNet(Ws, [[grid_X[i][j]],[grid_Y[i][j]],[1]], fs)[0][0] for j in range(len(grid_X[i]))] for i in range(len(grid_X))]

    try:
        CS = plt.contour(grid_X, grid_Y, my_Z, 6, colors = 'k')
        plt.clabel(CS, inline=1, fontsize=20)
    except ValueError:
        pass
    
    # Plot each set
    plt.scatter(X0, Y0, color='red')
    plt.scatter(X1, Y1, color='blue')



# ARGUMENT PARSING
# ================

validFuncs = {
    'linear'   : linear,
    'sigmoid'  : sigmoid,
    'relu'     : relu,
    'unitStep' : unitStep
}

demo = 'and'
# The size of each layer
sizes = [3, 3, 1]
fs = [sigmoid, linear]
steps = 20
i = 1;

if __name__ == '__main__':
    while i < len(args):
        arg = args[i]
        if arg == "-d":
            demo = args[i+1]
            i += 1
        elif arg == "-s":
            sizes = []
            for l in range(int(args[i+1])+1):
                sizes.append(int(args[i+2+l]))
            fs = [validFuncs[args[i + 3 + int(args[i+1]) + k]] for k in range(int(args[i+1]))]

            for f in fs:
                for k in validFuncs.keys():
                    if validFuncs[k] == f:
                        print(k)

            i += 2 * int(args[i+1]) + 2
        elif arg == "-a":
            alpha = float(args[i+1])
            i += 1
        elif arg == "-i":
            steps = int(args[i+1])
            i += 1
        elif arg == "-h":
            print("Flag | Description")
            print(" -a  | Sets training alpha value")
            print(" -d  | Sets the demo name")
            print(" -h  | Shows help listing")
            print(" -i  | Sets number of training iterations between UI redraws")
            print(" -s  | Sets layer sizes and functions")
            exit()
        else:
            print("Illegal argument '" + arg + "' detected")
            exit()
        i += 1


    if demo == "randsep":
        randSetup(30, 1)
    elif demo == "randtog":
        randSetup(30, 0)
    elif demo == "and":
        andSetup()
    elif demo == "xor":
        xorSetup()
    else:
        print("Demo '" + str(demo) + "' is not available.")
        exit()

    print("Creating network with " + str(len(sizes) - 1) + " layers")
    print("Demo " + demo + " selected")
    print("Training will run " + str(steps) + " iterations between UI redraws")

    # MAIN EXECUTION
    # ==============

    # Build the scatter points
    X0 = []
    Y0 = []
    X1 = []
    Y1 = []
    for i in range(len(X)):
        if Z[i] == 1:
            X1.append(X[i])
            Y1.append(Y[i])
        else:
            X0.append(X[i])
            Y0.append(Y[i])

    # Neural net weights are randomly selected.
    Ws = [
        [[random.random() for x in range(sizes[i])] for y in range(sizes[i+1])] for i in range(len(sizes) - 1)
    ]


    iterCnt = 0
    while True:
        # Redraw the plot
        resetPlot(Ws, fs)
        plt.gcf().canvas.set_window_title("Iteration " + str(iterCnt))
        plt.pause(0.03125)
        
        # Redraw each pair
        for j in range(steps):
            for i in range(len(data)):
                pair = data[i]
                trainOnPair(Ws, fs, transpose([pair[:-1] + [1]]), [[pair[-1]]])
                #resetPlot(betas[0], betas[1], betas[2])
                #plt.gcf().canvas.set_window_title("Iteration " + str(iterCnt) + ": " + str(i) + " of " + str(len(data)))
                #plt.pause(1.0 / len(data))
            iterCnt += 1

        print("Completed iteration " + str(iterCnt))



