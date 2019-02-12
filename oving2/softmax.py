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


    class relu(object):
        @staticmethod
        def f(X, w=None):
            if w is None:
                Xc = np.array(X)
                Xc[Xc<0] = 0
                return Xc
            else:
                z = X.dot(w.T)
                zc = np.array(z)
                zc[zc<0] = 0
                return zc


        @classmethod
        def df(cls, z):
            zc = np.array(z)
            zc[zc<0] = 0
            zc[zc>0] = 1
            return zc


    class dropout(object):
        @staticmethod
        def f(z):
            raise NotImplementedError

        @classmethod
        def df(cls, z):
            return z


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


    def add_dropout(self, rate, input_size=0):
        if not self.layers:
            self.input_size = input_size
            self.layers.append(Dropout(rate, self.input_size))
        else:
            self.layers.append(Dropout(rate, self.layers[-1].neurons))


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


    def train(self, dataset, epochs, batch_size, lr, evaluate=False, momentum=0.0, decay=0.0):
        X, Y = dataset.X_train, dataset.Y_train
        batches = X.shape[0] // batch_size
        lr0 = lr

        for t in range(epochs):
            # Shuffle training data here.
            Dataset.shuffle(X, Y)

            # If we are using momentum, initialize weighted average
            # sum of previous gradients.
            weighted_avg_gradients = \
                [np.zeros((l.neurons, l.input_size)) for l in self.layers]

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
                        if momentum:
                            update = update + (momentum/batch_size)*weighted_avg_gradients[j]
                            weighted_avg_gradients[j] = update
                        layer.weights = layer.weights - update

                # Evaluate the model according to given metrics.
                if evaluate and i % (batches // 2) == 0:
                    train_loss, train_acc = self.evaluate(dataset.X_train, dataset.Y_train)
                    self.metrics.train_loss.append(train_loss)
                    self.metrics.train_acc.append(train_acc)

                    val_loss, val_acc = self.evaluate(dataset.X_val, dataset.Y_val)
                    self.metrics.val_loss.append(val_loss)
                    self.metrics.val_acc.append(val_acc)

                    test_loss, test_acc = self.evaluate(dataset.X_test, dataset.Y_test)
                    self.metrics.test_loss.append(test_loss)
                    self.metrics.test_acc.append(test_acc)

            # Anneal learning rate (exponential decay).
            lr = lr0 * np.exp(-decay*t)

            print('Train acc:', self.metrics.train_acc[-1])
            print('Val acc:', self.metrics.val_acc[-1])
            print('Test acc:', self.metrics.test_acc[-1])


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
            z = layer.z(activations[i])
            zs.append(z)
            activations.append(layer.evaluate_z(z))

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

            # This assumes no abuse of layer building within
            # models. This is obviously not very sturdy or general,
            # but should work for our small use case.
            if type(layer) == Dropout:
                gradients.insert(0, np.array([]))
            elif type(next_layer) == Dropout:
                derivative = layer.activation.df(zs[i])
                masked_ds = d * next_layer.mask
                delta_sum = np.dot(self.layers[i+2].weights.T, d)
                d = derivative * np.multiply(delta_sum, next_layer.mask[:, None])
                gradients.insert(0, np.dot(d, activations[i-1].T))
                next_layer.remask()
            else:
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


    def z(self, X):
        return np.dot(self.weights, X)


    def evaluate_z(self, z):
        return self.activation.f(z)


    def evaluate(self, X):
        return self.activation.f(X, self.weights)


class Dropout(Layer):
    def __init__(self, rate, size):
        self.rate = 1-rate
        self.neurons = size
        self.input_size = size
        self.activation = Activations.dropout
        self.mask = np.random.binomial(1, self.rate, size=self.input_size) / self.rate
        self.weights = 1


    def remask(self):
        self.mask = np.random.binomial(1, self.rate, size=self.input_size) / self.rate


    def z(self, X):
        return X


    def evaluate_z(self, z):
        activations = z * self.mask[:, None]
        return activations


    def evaluate(self, X):
        return X


def main():
    # Load MNIST.
    mnist = MNIST()

    # Train model on dataset (MNIST in this case).
    # model = Model()
    # model.add_dropout(0.25, mnist.X_train.shape[1])
    # model.add_layer(64, Activations.relu)
    # model.add_dropout(0.25)
    # model.add_layer(64, Activations.relu)
    # model.add_dropout(0.20)
    # model.add_layer(10, Activations.softmax)
    # model.train(mnist, epochs=100, batch_size=128, lr=0.5,
    #             evaluate=True)
    # model.plot_metrics()

    # Pretty good this one.
    model = Model()
    model.add_dropout(0.15, mnist.X_train.shape[1])
    model.add_layer(128, Activations.relu)
    model.add_dropout(0.15)
    model.add_layer(10, Activations.softmax)
    model.train(mnist, epochs=100, batch_size=128, lr=0.5,
                evaluate=True, decay=0.005)

    # model = Model()
    # model.add_layer(60, Activations.tanh, mnist.X_train.shape[1])
    # model.add_layer(60, Activations.tanh)
    # model.add_layer(10, Activations.softmax)
    # model.train(mnist, epochs=15, batch_size=128, lr=0.5,
    #             evaluate=True)

    # model = Model()
    # model.add_dropout(0.4, mnist.X_train.shape[1])
    # model.add_layer(64, Activations.relu)
    # model.add_dropout(0.25)
    # model.add_layer(10, Activations.softmax)
    # model.train(mnist, epochs=15, batch_size=128, lr=0.5,
    #             evaluate=True, decay=0.03)


if __name__ == '__main__':
    main()
