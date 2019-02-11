import numpy as np
import matplotlib.pyplot as plt
import mnist
import tqdm
import os


class Dataset(object):
    def __init__(self):
        pass


    @staticmethod
    def shuffle(X, Y):
        # This is probably easier with zipping and unzipping.
        rng = np.random.get_state()
        np.random.shuffle(X)
        np.random.set_state(rng)
        np.random.shuffle(Y)


    def train_val_split(self, X, Y, val_percentage):
        """
        Selects samples from the dataset randomly to be in the
        validation set. Also, shuffles the train set.
        --
        X: [N, num_features] numpy vector,
        Y: [N, 1] numpy vector
        val_percentage: amount of data to put in validation set
        """
        dataset_size = X.shape[0]
        idx = np.arange(0, dataset_size)
        np.random.shuffle(idx)

        train_size = int(dataset_size * (1 - val_percentage))
        idx_train = idx[:train_size]
        idx_val = idx[train_size:]
        X_train, Y_train = X[idx_train], Y[idx_train]
        X_val, Y_val = X[idx_val], Y[idx_val]
        return X_train, Y_train, X_val, Y_val


    def onehot_encode(self, Y, n_classes=10):
        onehot = np.zeros((Y.shape[0], n_classes))
        onehot[np.arange(0, Y.shape[0]), Y] = 1
        return onehot


    def bias_trick(self, X):
        return np.concatenate((X, np.ones((len(X), 1))), axis=1)


class MNIST(Dataset):
    # For passing MNIST around nicely.
    def __init__(self):
        if not os.path.isdir(mnist.SAVE_PATH):
            mnist.init()

        # Load the data set -- in this case MNIST.
        X_train, Y_train, X_test, Y_test = mnist.load()

        # Pre-process data by adding 1 to the input set for bias trick,
        # one-hot encoding the target values, and splitting the training
        # set to include a validation set.
        X_train, X_test = X_train / 127.5 - 1, X_test / 127.5 - 1
        X_train = self.bias_trick(X_train)
        X_test = self.bias_trick(X_test)
        Y_train, Y_test = self.onehot_encode(Y_train), self.onehot_encode(Y_test)
        X_train, Y_train, X_val, Y_val = self.train_val_split(X_train, Y_train, 0.1)

        self.X_train, self.Y_train = X_train, Y_train
        self.X_val, self.Y_val = X_val, Y_val
        self.X_test, self.Y_test = X_test, Y_test


class Activations(object):
    """
    These are really just namespaces to keep track of stuff, as we
    want to deliver just a single file for the assignment.
    """
    class softmax(object):
        @staticmethod
        def f(X, w=None):
            """
            Computes softmax for input X and weights w. If no weights
            are given, the input is assumed to be already computed
            zs. This is done to avoid having to pass overloaded
            functions as attributes to layers.
            """
            if w is None:
                a_exp = np.exp(X)
                return a_exp / a_exp.sum()
            else:
                a = X.dot(w.T)
                a_exp = np.exp(a)
                return a_exp / a_exp.sum(axis=1, keepdims=True)


        @classmethod
        def df(cls, z):
            raise NotImplementedError


    class sigmoid(object):
        @staticmethod
        def f(X, w=None):
            if w is None:
                return 1.0/(1.0 + np.exp(-X))
            else:
                z = X.dot(w.T)
                return 1.0/(1.0 + np.exp(-z))


        @classmethod
        def df(cls, z):
            return cls.f(z)*(1-cls.f(z))


    class tanh(object):
        @staticmethod
        def f(X, w=None):
            if w is None:
                return 1.7159*np.tanh(2/3*X)
            else:
                z = X.dot(w.T)
                return 1.7159*np.tanh(2/3*z)


        @classmethod
        def df(cls, z):
            # Use 1/cosh^2 instead of sech^2.
            return 1.14393 * (1 / np.cosh(2/3*z))**2


class Loss(object):
    def cross_entropy(outputs, targets):
        """
        Calculates the cross entropy loss between a set of given
        outputs from a model, and the corresponding target values.
        """
        log_y = np.log(outputs)
        cross_entropy = -(targets * log_y)
        return cross_entropy.mean()


class Metrics(object):
    """
    Wrapper for several metrics that may be useful for evaluating the
    performance of a network.
    """
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.test_loss = []

        self.train_acc = []
        self.val_acc = []
        self.test_acc = []


class Model(object):
    def __init__(self):
        self.layers = []
        self.metrics = Metrics()


    def add_layer(self, neurons, activation, input_size=0):
        """
        Add layers to Model. When the initial layer is added, the
        input size to the network has to be specified as well.
        """
        if not self.layers:
            self.input_size = input_size
            self.layers.append(Layer(neurons, activation, self.input_size))
        else:
            self.layers.append(Layer(neurons, activation, self.layers[-1].neurons))


    def forward(self, X):
        if len(X.shape) == 1:
            X = np.array([X])

        for layer in self.layers:
            X = layer.evaluate(X)

        return X


    def evaluate(self, X, Y):
        """
        Evaluates a the performance of the model on a set of inputs X
        and targets Y.
        """
        outputs = self.forward(X)

        # Calculate loss.
        cross_entropy_loss = Loss.cross_entropy(outputs, Y)

        # Calculate accuracy.
        predictions = outputs.argmax(axis=1)
        targets = Y.argmax(axis=1)
        acc = (predictions == targets).mean()

        return cross_entropy_loss, acc*100


    def plot_metrics(self):
        """
        Plots all available metrics in a standard manner. Only metrics
        that have available data are considered.
        """
        # Plot loss data.
        plt.figure(figsize=(12, 8))
        plt.ylim([0, 0.6])
        if self.metrics.train_loss:
            plt.plot(self.metrics.train_loss, label='Training loss')
        if self.metrics.val_loss:
            plt.plot(self.metrics.val_loss, label='Validation loss')
        if self.metrics.test_loss:
            plt.plot(self.metrics.test_loss, label='Test loss')
        plt.legend()
        plt.show()
        plt.clf()

        # Plot accuracy data.
        plt.figure(figsize=(12, 8))
        plt.ylim([0, 100])
        if self.metrics.train_acc:
            plt.plot(self.metrics.train_acc, label='Training accuracy')
        if self.metrics.val_acc:
            plt.plot(self.metrics.val_acc, label='Validation accuracy')
        if self.metrics.test_acc:
            plt.plot(self.metrics.test_acc, label='Test accuracy')
        plt.legend()
        plt.show()
        plt.clf()


    def train(self, dataset, epochs, batch_size, lr, evaluate=False,
              val_set=None, test_set=None):
        X, Y = dataset.X_train, dataset.Y_train
        batches = X.shape[0] // batch_size

        # There is currently no error handling for when no
        # validation/test sets are supplied -- but we don't care for
        # now.
        if val_set:
            X_val, Y_val = val_set
        if test_set:
            X_train, Y_train = val_set

        for t in range(epochs):
            # Shuffle training data here.
            Dataset.shuffle(X, Y)

            # SGD over the training set. For each training example,
            # perform the backpropagation algorithm and update the
            # weights of the network by the gradient descent rule
            # (i.e. move in the opposite direction of the gradient
            # that is output).
            for i in tqdm.trange(batches):
                X_batch = X[i*batch_size:(i+1)*batch_size]
                Y_batch = Y[i*batch_size:(i+1)*batch_size]

                for x, y in zip(X_batch, Y_batch):
                    x = np.expand_dims(x, axis=1)
                    y = np.expand_dims(y, axis=1)
                    gradients = self.backprop(x, y)

                    # Update weights according to gradients.
                    for j, layer in enumerate(self.layers):
                        layer_gradients = gradients[j]
                        update = (lr/batch_size) * layer_gradients
                        layer.weights = layer.weights - update

                # Evaluate the model according to given metrics.
                if evaluate and i % (batches // 20) == 0:
                    # train_loss, train_acc = self.evaluate(dataset.X_train, dataset.Y_train)
                    # self.metrics.train_loss.append(train_loss)
                    # self.metrics.train_acc.append(train_acc)

                    val_loss, val_acc = self.evaluate(dataset.X_val, dataset.Y_val)
                    self.metrics.val_loss.append(val_loss)
                    self.metrics.val_acc.append(val_acc)

                    # test_loss, test_acc = self.evaluate(dataset.X_test, dataset.Y_test)
                    # self.metrics.test_loss.append(test_loss)
                    # self.metrics.test_acc.append(test_acc)

            print('Val acc:', self.metrics.val_acc[-1])


    def backprop(self, x, t):
        """
        The backpropagation algorithm consists of (in order):

        * Feedforward, compute activations for the entire network for
          input X.

        * Compute the output error of the last layer of the network.

        * Backpropagate the error layer by layer by d_j =
          f'(z_j)*sum(w_kj*d_k), where we are computing the error of
          layer j by the means of the next layer k.

        * The output is the gradient of the cost function for each
          layer. For each individual weight this is given by the
          activation of the previous layer multiplied by the error of
          the output of the current layer.
        """
        # Hold the actual gradients as they are computed.
        gradients = []

        # We need the activations of each layer to produce the
        # gradient, and the zs for each layer to produce the
        # derivative required for the delta.
        activations = [x]
        zs = []

        # Compute activations for each layer.
        for i, layer in enumerate(self.layers):
            z = np.dot(layer.weights, activations[i])
            zs.append(z)
            activations.append(layer.activation.f(z))

        # Calculate the first error (delta) for the output layer, as
        # this is the only one that uses the actual cost derivative,
        # which in our case is -(t_k - y_k) from log cross entropy.
        d = -(t - activations[-1])
        gradients.insert(0, np.dot(d, activations[-2].T))

        # Backpropagate errors and add gradients as we go. Note that
        # we reuse the delta for each iteration.
        for i in range(-2, -(len(self.layers)+1), -1):
            layer = self.layers[i]
            next_layer = self.layers[i+1]
            derivative = layer.activation.df(zs[i])
            d = derivative * np.dot(next_layer.weights.T, d)
            gradients.insert(0, np.dot(d, activations[i-1].T))
        return gradients


    def save():
        """
        Serialize and save the weights of the network.
        """
        pass


    def load():
        """
        Deserialize and load already trained weights into the network.
        """
        pass


class Layer(object):
    def __init__(self, neurons, activation, input_size):
        self.neurons = neurons
        self.activation = activation
        self.input_size = input_size

        # Initalize weights from a normal distribution with mean 0 and
        # std 1/sqrt(fan-in). The fan-in of a neuron is the number of
        # inputs it has.

        # Uniform distribution.
        # self.weights = np.random.uniform(-1, 1, (self.neurons, self.input_size))

        # Using fan-in.
        sigma = 1 / np.sqrt(input_size)
        self.weights = np.random.normal(loc=0, scale=sigma,
                                        size=(self.neurons, self.input_size))


    def evaluate(self, X):
        return self.activation.f(X, self.weights)


def main():
    # Load dataset.
    mnist = MNIST()

    # Train model on dataset.
    model = Model()
    model.add_layer(64, Activations.tanh, mnist.X_train.shape[1])
    model.add_layer(10, Activations.softmax)
    model.train(mnist, epochs=5, batch_size=128, lr=0.5, evaluate=True)
    # model.plot_metrics()


if __name__ == '__main__':
    main()
