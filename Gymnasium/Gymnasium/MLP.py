# multilayer perceptron of any architecture

import numpy as np
import matplotlib.pyplot as plt

class MLP:
    '''
    Class to define Multilayer Perceptrons.
    Declare instance with MLP(layers).
    '''
    def __init__(self, layers):
        '''
        layers: a tuple with (ninputs, nhidden1, nhidden2, ... noutput)
        '''
        self.layers = layers
        self.trace = False
        self.threshold = 5.0
        self.labels = None # text for the labels [input-list, output-list]
        
        self.size = 0
        self.W = [] # list of numpy matrices
        self.b = [] # list of numpy vectors
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i],layers[i+1])-0.5
            b = np.random.rand(layers[i+1])-0.5
            self.W.append(w)
            self.b.append(b)
            self.size += layers[i] * layers[i+1] + layers[i+1]
        
        self.lRMS = [] # hold all traced RMSs to draw graph
        self.laccuracy = [] # hold all traced accuracies to draw graph
        
    def sigm (self, neta):
        return 1.0 / (1.0 + np.exp(-neta))
    
    def forward (self, x): # propagate x vector and return output
        '''
        Get network outputs from inputs in x
        x must be a vector or a list of vectors
        '''
        self.s = [x] # save all outputs for the update stage
        for w,b in zip(self.W, self.b):
            net = np.dot(self.s[-1],w) + b
            self.s.append(self.sigm(net))
        return self.s[-1]
    
    def fast_forward (self, x): # fast forward (optimized in time, but not use to train!)
        for i in range(len(self.b)):
            net = np.dot(x,self.W[i]) + self.b[i]
            x = self.sigm(net)
        return x
    
    def print (self, v, msg=''):
        if self.trace:
            print(msg)
            print(v)
    
    def update (self, x, d, alpha): # do a learning step
        self.forward(x) # propagate
        
        self.incW = [] # saved in attributes for possible batch updates
        self.incb = []
        for k in range(1,len(self.layers)): # crea matrices de actualización
            self.incW.append(np.zeros((self.layers[k-1],self.layers[k])))
            self.incb.append(np.zeros(self.layers[k]))
        
        # updates of output layer
        s = self.s[-1] # output layer
        delta = (d - s) * s * (1.0 - s) # derivada sigmoidal f'(x) = f(x) * (1 - f(x))
        self.incW[-1] = alpha * np.dot(self.s[-2].reshape(self.layers[-2],1), delta.reshape(1,self.layers[-1])) # reshape coloca los vectores en forma de matriz
        self.incb[-1] = alpha * delta
        
        # updates of hidden layers
        for k in range(len(self.layers)-3,-1,-1):
            R = np.dot(self.W[k+1], delta)
            delta = R.flatten() * self.s[k+1] * (1.0 - self.s[k+1])
            self.incW[k] = alpha * np.dot(self.s[k].reshape(self.layers[k],1), delta.reshape(1,self.layers[k+1])) # reshape coloca los vectores en forma de matriz
            self.incb[k] = alpha * delta
        
        self.print(self.incW)
        self.print(self.incb)
        
        # actualiza
        for i in range(0,len(self.layers)-1):
            self.W[i] += self.incW[i]
            self.b[i] += self.incb[i]
    
        self.print(self.W)
        self.print(self.b)
        
    def RMS (self, X, D):
        S = self.forward(X)
        return np.mean(np.sqrt(np.mean(np.square(S-D),axis=1)))
    
    def accuracy (self, X, D):
        S = self.forward(X)
        S = np.round(S)
        errors = np.mean(np.abs(D-S))
        return 1.0 - errors
    
    def info (self, X, D):
        '''
        Print evaluation of an MLP.
        X: inputs to test.
        D: desired outputs.
        '''
        self.lRMS.append(self.RMS(X,D))
        self.laccuracy.append(self.accuracy(X,D))
        print('     RMS: %6.5f' % self.lRMS[-1])
        print('Accuracy: %6.5f' % self.laccuracy[-1])
        
    def train (self, X, D, alpha, epochs, trace=0, observer=None):
        '''
        Train a MLP.
        X: inputs.
        D: desired outputs.
        alpha: learning rate.
        epoch: number of epoch to train.
        trace: epoch number to print progress (0 = no trace).
        observer: call back after each trace.
        '''
        self.lRMS = [] # save all traced RMSs to draw graph
        self.laccuracy = [] # save all traced accuracies to draw graph

        for e in range(1,epochs+1):
            for i in range(len(X)):
                self.update(X[i],D[i], alpha)
            if trace!=0 and e%trace == 0:
                print('\n   Epoch: %d' % e)
                self.info(X,D)
                if observer:
                    observer()
        print()
            
    def to_chromosome (self):
        '''
        Convert weights and biases to a flatten list to use in AG.
        '''
        ch = []
        for w,b in zip(self.W,self.b):
            ch += w.flatten().tolist()
            ch += b.flatten().tolist()
        return ch

    def from_chromosome (self, ch):
        '''
        Convert a flatten list (chromosome from a GA) to internal weights and biases.
        '''
        if len(ch) != self.size:
            print(self.size)
            raise ValueError("Chromosome legnth doesn't match architecture")
        self.W = []
        self.b = []
        flat = np.array(ch)
        pos = 0
        for i in range(len(self.layers)-1): # for each layer
            to = self.layers[i]*self.layers[i+1] # number of weights
            w = np.array(flat[pos:pos+to]).reshape(self.layers[i],self.layers[i+1])
            pos += to
            to = self.layers[i+1] # number of bias
            b = np.array(flat[pos:pos+to]).reshape(self.layers[i+1])
            pos += to
            
            self.W.append(w)
            self.b.append(b)
    
    def save (self, name):
        f = open(name, 'wb')
        pickle.dump(self.layers,f)
        pickle.dump(self.W,f)
        pickle.dump(self.b,f)
        f.close()
        
    def load (self, name):
        f = open(name, 'rb')
        self.layers = pickle.load(f)
        self.W = pickle.load(f)
        self.b = pickle.load(f)
        f.close()
    
    def regions (self, xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, size=200, init=True, outputs=(0,1,2), colors=(0,1,2)):
        '''
        Draw classification regions.
        outputs: how to map each output to a color
        colors: how colorize each output (0: red, 1: green; 2: blue)
        '''
        if self.layers[-1]==1: # if only 1 output
            outputs = outputs[:1]
        if self.layers[-1]==2: # if only 2 outputs
            outputs = outputs[:2]

        fig = plt.figure()
        if init: # more efficiente if used only the first time
            self.valx = np.arange(xmin, xmax, (xmax-xmin)/size)
            self.valy = np.arange(ymin, ymax, (ymax-ymin)/size)
        
        X = np.zeros((len(self.valx), len(self.valy), 3), np.uint8)
        for j,x in enumerate(self.valx):
            for i,y in enumerate(self.valy):
                s = self.forward([x,y])
                X[i,j,colors[0]] = int(255*s[outputs[0]])
                if len(outputs)>1:
                    X[i,j,colors[1]] = int(255*s[outputs[1]])
                if len(outputs)>2:
                    X[i,j,colors[2]] = int(255*s[outputs[2]])
                
        plt.imshow(X, origin='lower', cmap='gray', extent=[xmin,xmax,ymin,ymax])
        
    def color (self, w):
        # color depends of excitatory or inhibitory weight, strength depends on abs value
        # return (color, alpha, linewidth)
        
        level = w/self.threshold
        
        color = 'red'
        if w>0:
            color = 'green'
            
        linewidth = 1.0
        alpha = 1.0
        if abs(level)>1.0:
            linewidth = level
        else:
            alpha = 1.0 - abs(level)
            
        return (color, alpha, linewidth)
        
    def paint (self, threshold=1.0):
        '''
        Draw a graphical representation of an MLP
        '''
        #plt.style.use('dark_background')
        fig = plt.figure(figsize=(8, 6))
        ax = fig.gca()
        ax.axis('off')
        ax.axis('equal') # avoid deforming
        ax.invert_yaxis() # set the origin in top
        
        ww = self.W

        relleno = '#C0C0FF' # #8080FF

        maxx = 1.4
        maxy = 1
        size = 0.03 # neuron size
        size2 = 2*size
        marginx = 0.1
        marginy = 0.1

        # draw connections
        inclayer = (maxx-marginx) / len(ww)
        x = marginx/2
        xant = x
        x += inclayer

        for w in ww:
            sw = w.shape

            if sw[1]==1:
                incneuron = 0
                y = maxy/2
            else:
                incneuron = (maxy-marginy) / (sw[1]-1)
                y = marginy/2

            for i in range(sw[1]): # each neuron
                if sw[0]==1: # center if only one neuron (special case)
                    yant = maxy/2
                    incyant = 0
                else:
                    yant = marginy/2
                    incyant = (maxy-marginy) / (sw[0]-1)

                for j in range(sw[0]): # each previous layer neuron
                    par = self.color(w[j,i])
                    plt.plot([xant, x], [yant, y], color=par[0], alpha=par[1], linewidth=par[2], solid_capstyle='round')
                    yant += incyant

                y += incneuron
                
            xant = x
            x += inclayer

        # inputs
        inclayer = (maxx-marginx) / float(len(ww))
        x = marginx/2

        if self.layers[0]<2: # center if only one neuron (special case)
            y = maxy/2
            incneuron = 0
        else:
            y = marginy/2
            incneuron = (maxy-marginy) / float(self.layers[0]-1)

        incinput = incneuron
        xinput = x
        if incneuron==0:
            self.margininput = y

        # draw rectangles and labels
        for i in range(self.layers[0]):
            shape = plt.Rectangle((x-size,y-size), size2, size2, color='w', ec='k', zorder=4)
            ax.add_artist(shape)
            if self.labels: # if there are labels defined
                plt.text(x-1.5*size, y+0.01, self.labels[0][i], color='k', zorder=4, ha='right'), 
            y += incneuron

        xant = x
        x += inclayer

        # draw neurons
        for b in self.b:
            if len(b)==1: # center if only one neuron (special case)
                incneuron = 0
                y = maxy/2
            else:
                incneuron = (maxy-marginy) / float(len(b)-1)
                y = marginy/2

            for i in range(len(b)):
                # bias
                par = self.color(b[i])
                plt.plot([x, x], [y-size2, y], color=par[0], alpha=par[1], linewidth=par[2], solid_capstyle='round', zorder=4)
                
                # neuron
                shape = plt.Circle((x, y), size, color=relleno, ec='k', zorder=4)
                ax.add_artist(shape)
                
                # labels in output layer
                if self.labels and b is self.b[-1]:
                    plt.text(x+1.5*size, y+0.01, self.labels[1][i], color='k', zorder=4, ha='left'), 

                y += incneuron
            xant = x
            x += inclayer
                
def one_hot (d):
    num_classes = len(set(d))
    rows = d.shape[0]
    labels = np.zeros((rows, num_classes), dtype='float32')
    labels[np.arange(rows),d.T] = 1
    return labels
