import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt
# for inset color bars
import mpl_toolkits as toolkits
import mpl_toolkits.axes_grid.inset_locator as insetter

def activation_jump(z_val): #z_val is a vector
    #jump activation
    py_results = []
    #first way 
    # for z in z_val:
    #     if z>0:
    #         py_results.append(1)
    #     else:
    #         py_results.append(0)
    
    #second way
    py_results = z_val>0 #becomes an array of [true,true,false.....] values
    res = np.array(py_results,dtype=float)
    return res

def get_z(y_in,w,b):
    z_val = np.dot(y_in,w)+b
    return z_val

def apply_layer(y_in, w, b, activation):
    return apply_layer_get_z(y_in, w, b, activation)[0]

def apply_layer_get_z(y_in, w, b, activation):
    """
    go from one layer to the next given 
    w = weigth matrix shape [nuerons_in x neurons out]
    b = bias vector of y_in length
    y_in = values of inputs of shape [bach_size,n_nerouns_in]

    returns the values of the output neoruons in the next layer as matrix
    shape [batch_size,n_nerouns_out]
    """
    y_in = np.array(y_in)
    w = np.array(w)
    ##helpful inside of visualization code
    if len(w.shape) == 1 and w.shape[0] != y_in.shape[0]:
        #w shape = 1, means it is a vector but it must be a matrix
        #if we are here, so we flipped the inputting of (y_in, w) so we try flipping them
        #rather than crashing
        #print("Before :: ",w.shape,"x",y_in.shape)
        c = y_in
        y_in = w
        w = c
        #print("After :: ",w.shape,"x",y_in.shape)


    z = np.dot(w,y_in) + b
    #print(z)

    y_next_layer_out = np.empty(len(z))
    if callable(activation):
        y_next_layer_out = activation(z)
    elif activation == "sig" or activation == "sigmoid":
        y_next_layer_out = (1.0/(1+np.exp(-z)))
    elif activation == "jump":
        y_next_layer_out = np.array(z > 0, dtype="float")
    elif activation == "lin" or activation == "linear":
        y_next_layer_out = z
    elif activation.lower() == "relu":
        y_next_layer_out = ((z > 0)*z)
    
    return [y_next_layer_out,z]

def apply_net(y_in, weigths, biases, activations):
    """
    apply a whole network of multiple layers 
    biases are a collection of biases, 
    each bias is a Vector of same dimensions as the layer of the same index
    """
    y = y_in
    for layerIndex in range(len(biases)):
        y = apply_layer(y, weigths[layerIndex], biases[layerIndex], activations[layerIndex])
    return y

# routines for plotting the network


def plot_connection_line(ax, X, Y, W, vmax=1.0, line_width=3.0, spaces=20, col=[1, 0.3, 0,1]):
    t = np.linspace(0, 1, spaces)
    if W > 0:
        col = [0, 0.4, 0.8,1.0]

    xPos = X[0]+(3*t**2-2*t**3)*(X[1]-X[0])
    yPos = Y[0]+t*(Y[1]-Y[0])
    alph = np.abs(W)/vmax
    ax.plot(xPos, yPos, alpha=alph, color=col, linewidth=line_width)
    pass


def plot_neuron_alpha(ax, X, Y, B, size=75, vmax=1.0, col=[1, 0.3, 0,1], z_order=15):
    if B > 0:
        col = [0, 0.4, 0.8,1.0]
    alph = np.abs(B)/vmax

    ax.scatter([X], [Y], marker='o', color=col, alpha=alph, zorder=z_order)
    pass


def plot_neuron(ax, X, Y, B, size=50, vmax=1.0, col=[1, 0.3, 0,1], z_order=15):
    if B > 0:
        col = [0, 0.4, 0.8,1.0]
    ax.scatter([X], [Y], marker='o',s=size, color=col, zorder=z_order)
    pass


##################################################
#Region Start
#Helper Functions Enclosed By This Region 
#Have been taken from https://github.com/FlorianMarquardt
#and have begen modified by Mo'tasem Zakarneh to fit the project accordingly
##################################################

def visualize_network_2_in(weigths, biases, activations, M=100, y0range=[-1, 1], y1range=[-1, 1], size=500, line_width=5.0):
    """
    Visualize a neural network with 2 input neurons and 1 output neuron 
    (plot output vs input in a 2D plot)

    weights is a list of the weight matrices for the layers, 
    where weights[j] is the matrix for the connections
    from layer j to layer j+1 (where j==0 is the input)

    weights[j][m,k] is the weight for input neuron k going to output neuron m
    (note: internally, m and k are swapped, see the explanation of
    batch processing in lecture 2)

    biases[j] is the vector of bias values for obtaining the neurons in layer j+1
    biases[j][k] is the bias for neuron k in layer j+1

    activations is a list of the activation functions for
    the different layers: choose 'linear','sigmoid',
    'jump' (i.e. step-function), and 'reLU'

    M is the resolution (MxM grid)

    y0range is the range of y0 neuron values (horizontal axis)
    y1range is the range of y1 neuron values (vertical axis)
    """
    swapped_weights = []
    for j in range(len(weigths)):
        swapped_weights.append(np.transpose(weigths[j]))

    y0, y1 = np.meshgrid(np.linspace(
        y0range[0], y0range[1], M), np.linspace(y1range[0], y1range[1], M))
    y_in = np.zeros([M*M, 2])
    y_in[:, 0] = y0.flatten()
    y_in[:, 1] = y1.flatten()
    y_out = apply_net(y_in, swapped_weights, biases, activations)

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 6))

    # plot the network itself:

    # positions of neurons on plot:
    posX = [[-0.5, +0.5]]
    posY = [[0, 0]]
    vmax = 0.0  # for finding the maximum weight
    vmaxB = 0.0  # for maximum bias
    for j in range(len(biases)):
        n_neurons = len(biases[j])
        posX.append(np.array(range(n_neurons))-0.5*(n_neurons-1))
        posY.append(np.full(n_neurons, j+1))
        vmax = np.maximum(vmax, np.max(np.abs(weigths[j])))
        vmaxB = np.maximum(vmaxB, np.max(np.abs(biases[j])))

    # plot connections
    for j in range(len(biases)):
        for k in range(len(posX[j])):
            for m in range(len(posX[j+1])):
                plot_connection_line(ax[0], [posX[j][k], posX[j+1][m]],
                                     [posY[j][k], posY[j+1][m]],
                                     swapped_weights[j][k, m], vmax=vmax,
                                     line_width=line_width)

    # plot neurons
    for k in range(len(posX[0])):  # input neurons (have no bias!)
        plot_neuron(ax[0], posX[0][k], posY[0][k],
                    vmaxB, vmax=vmaxB, size=size)
    for j in range(len(biases)):  # all other neurons
        for k in range(len(posX[j+1])):
            plot_neuron(ax[0], posX[j+1][k], posY[j+1][k],
                        biases[j][k], vmax=vmaxB, size=size)

    ax[0].axis('off')

    # now: the output of the network
    img = ax[1].imshow(np.reshape(y_out, [M, M]), origin='lower',
                       extent=[y0range[0], y0range[1], y1range[0], y1range[1]])
    ax[1].set_xlabel(r'$y_0$')
    ax[1].set_ylabel(r'$y_1$')
    
    axins1 = insetter.inset_axes(ax[1],
                        width="40%",  # width = 50% of parent_bbox width
                        height="5%",  # height : 5%
                        loc='upper right')

    imgmin = np.min(y_out)
    imgmax = np.max(y_out)
    color_bar = fig.colorbar(
        img, cax=axins1, orientation="horizontal", ticks=np.linspace(imgmin, imgmax, 3))
    cbxtick_obj = plt.getp(color_bar.ax.axes, 'xticklabels')
    plt.setp(cbxtick_obj, color="white")
    axins1.xaxis.set_ticks_position("bottom")

    plt.show()
    pass

##################################################
#Region End
##################################################