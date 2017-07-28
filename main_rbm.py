import numpy as np
from main_explore_data import shuffle_datasets, plot_examples

"""When the comments refer to PG_RBM, it refers to
A Practical Guide to Training Restricted Boltzmann Machines, version 1, Geoffrey Hinton
https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
"""

def sigm(X):
    return 1./(1.+np.exp(-X))

class RBM():
    def __init__(self, num_hidden, num_visible):
        self.num_hidden = num_hidden
        self.num_visible = num_visible

        # Initialize the weights with Xavier Glorot initialization
        # http://proceedings.mlr.press/v9/glorot10a.html
        scale = 0.1*np.sqrt(6. /(num_hidden + num_visible ))
        self.weights = np.random.uniform(-scale, scale, (num_visible, num_hidden))

        # Append zeros for the biases
        self.weights = np.insert(self.weights, 0, 0, axis = 0)
        self.weights = np.insert(self.weights, 0, 0, axis = 1)
        self.delta_weights = np.zeros_like(self.weights)

    def train(self, data, num_epochs=1000, learning_rate=0.1, momentum = 0.5, CD_steps=1, batch_size=100):
        self.momentum = momentum
        self.learning_rate = learning_rate

        #Add a bias column
        data = np.insert(data, 0, 1, axis = 1)

        num_samples, num_visible = data.shape

        for n in range(num_epochs):
            subsample = data[np.random.choice(num_samples, batch_size)]
            visible_sample = subsample.copy()

            #Make CD steps of Gibbs samples
            hidden_activations, hidden_sample, visible_activations, visible_sample = self.gibbs_sample(visible_sample, CD_steps)

            # Now calculate both parts of eq.6 in PG_RBM
            E_data = np.dot(subsample.T, hidden_activations )
            E_model = np.dot(visible_activations.T, hidden_activations)

            # Update the weights
            gradient = (E_data - E_model) / batch_size
            self.weights += learning_rate*gradient

            if n%10 == 0:
                cross_entropy = np.mean(-1*(subsample[:,1:]*np.log(visible_activations[:,1:]) + (1-subsample[:,1:])*np.log(1-visible_activations[:,1:])))
                print("At step %3i/%5i, the gradient norm is %5.3f and the cross entropy %5.3f "%(n, num_epochs, np.linalg.norm(gradient), cross_entropy))

    def gibbs_sample(self, visible_sample, steps=3):
        """
        Performs block Gibbs sampling. Alternatively samples the hidden states
        and the visible states
        :param visible_sample:
        :param steps:
        :return: None
        """
        batch_size = visible_sample.shape[0]
        for step in range(steps):
            # Updating the hidden states
            # Section 3.1 PG_RBM
            hidden_activations = sigm(np.dot(visible_sample, self.weights))
            hidden_activations[:,0] = 1. #Biases are always 1
            hidden_sample = hidden_activations > np.random.rand(batch_size, self.num_hidden + 1)

            # Updating the visible states
            # Section 3.2 PG_RBM
            visible_activations = sigm(np.dot(hidden_sample, self.weights.T))
            visible_activations[:,0] = 1. #Biases are always 1
            visible_sample = visible_activations > np.random.rand(batch_size, self.num_visible + 1)
        return hidden_activations, hidden_sample, visible_activations, visible_sample


    def update_weights(self, gradient, n):
        """
        Updates the weights according to the gradient. Implementing both learning rate decay
        and momentum
        :param gradient: the gradient wrt the weights
        :param n: step
        :return: None
        """
        lr_decay = self.learning_rate*(1/2)*(n/100)
        self.delta_weights = lr_decay * gradient + self.momentum*self.delta_weights
        self.weights += self.delta_weights


    def generate_samples(self, num_samples = 25):
        visible_sample = np.random.rand(1,785)
        visible_sample[0,0] = 1
        samples = []
        for n in range(num_samples):
            hidden_activations, hidden_sample, visible_activations, visible_sample = self.gibbs_sample(visible_sample, 10)
            samples.append(visible_activations)
        return samples






if __name__ == "__main__":
    X = shuffle_datasets(10000)
    rbm = RBM(100, 784)
    rbm.train(X)

    samples = rbm.generate_samples(25)
    plot_examples(np.vstack(samples)[:,1:], 5)





