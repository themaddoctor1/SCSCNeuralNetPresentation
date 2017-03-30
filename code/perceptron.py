from sys import argv as args

import matplotlib.pyplot as plt
import pylab
import random

data = []
X = []
Y = []
Z = []
alpha = 0.1

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
            x += mutex
        print("(" + str(x) + "," + str(y) + ") -> " + str(z))
        X.append(x)
        Y.append(y)
        Z.append(z)
        data.append([x, y, z])
    alpha = 0.5 / points

"""
Resets the matplotlib plot
"""
def resetPlot(a, b, c):
    plt.clf()
    
    # The data, split by category
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
    
    # Plot each set
    plt.scatter(X0, Y0, color='red')
    plt.scatter(X1, Y1, color='blue')

    axes = plt.axis()

    # The regressed line
    plt.plot([1000, -1000], [-(c + 1000*a) / b, (1000*a - c) / b])
    plt.xlim([axes[0], axes[1]])
    plt.ylim([axes[2], axes[3]])

"""
Performs the equivalent operation to perceptron on a single data pair
"""
def perceptronDriver(coeffs, pair, alpha):
    # y = c_0 + c_1 x_1 + c_2 x_2
    y = coeffs[0] * pair[0] + coeffs[1] * pair[1] + coeffs[2];
    
    y = 1 if y > 0 else 0

    # Derivative of error for output is y - t
    err = y - pair[2]

    # Derivative of output as function of coefficients
    coDer = [pair[0], pair[1], 1]

    for i in range(3):
        coeffs[i] -= alpha * coDer[i] * err;

def adalineDriver(coeffs, pair, alpha):
    # y = c_0 + c_1 x_1 + c_2 x_2
    y = coeffs[0] * pair[0] + coeffs[1] * pair[1] + coeffs[2];
    
    # Derivative of error for output is y - t
    err = (2*y-1) - pair[2]

    # Derivative of output as function of coefficients
    coDer = [pair[0], pair[1], 1]

    for i in range(3):
        coeffs[i] -= alpha * coDer[i] * err;



"""
DOCS:
random <pts> <exc> - Randomly generates pts points with one of the classes offset by exc on the x-axis.
and                - Generates the AND gate sample.
xor                - Generates the XOR gate sample.
"""
if len(args) == 1:
    print("Insufficient command line arguments")
if args[1] == "random" and len(args) == 4:
    randSetup(int(args[2]), int(args[3]))
elif args[1] == "and":
    andSetup()
elif args[1] == "xor":
    xorSetup()
else:
    print("Invalid command line argument(s)")
    exit()

iterCnt = 0
betas = [random.random(), random.random(), -random.random()]
while True:
    iterCnt += 1
    
    # Redraw the plot
    resetPlot(betas[0], betas[1], betas[2])
    plt.gcf().canvas.set_window_title("Iteration " + str(iterCnt) + ": z = " + str(betas[0]) + "x + " + str(betas[1]) + "y + " + str(betas[2]))
    plt.pause(2)
    
    # Redraw each pair
    for i in range(len(data)):
        pair = data[i]
        # Train a value pair
        perceptronDriver(betas, pair, alpha)
        resetPlot(betas[0], betas[1], betas[2])
        plt.gcf().canvas.set_window_title("Iteration " + str(iterCnt) + ": " + str(i) + " of " + str(len(data)))
        plt.pause(1.0 / len(data))



