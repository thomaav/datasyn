import numpy as np
import matplotlib.pyplot as plt
import mnist
import tqdm
import os


class Model:
    def __init__(self):
        self.layers = []


    def add_layer(self, neurons, activation, input_size=0):
        """Add layers to Model. First added layer is the input layer."""
        if not self.layers:
            self.input_size = input_size
            self.layers.append(Layer(neurons, activation, self.input_size))
        else:
            self.layers.append(Layer(neurons, activation, self.layers[-1].neurons))


    def evaluate(self, X):
        if len(X.shape) == 1:
            X = np.array([X])
            
        for layer in self.layers:
            X = layer.evaluate(X)

        return X


class Layer:
    def __init__(self, neurons, activation, input_size):
        self.neurons = neurons
        self.activation = activation
        self.input_size = input_size
        self.weights = np.zeros((self.neurons, self.input_size))


    # def forward(self, X):
    #     return np.dot(self.weights, X.T)


    def evaluate(self, X):
        return self.activation(X, self.weights)


def should_early_stop(validation_loss, num_steps=3):
    if len(validation_loss) < num_steps + 1:
        return False

    is_increasing = [validation_loss[i] <= validation_loss[i + 1]
                     for i in range(-num_steps - 1, -1)]
    return sum(is_increasing) == len(is_increasing)


def train_val_split(X, Y, val_percentage):
    """
      Selects samples from the dataset randomly to be in the validation set. Also, shuffles the train set.
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


def onehot_encode(Y, n_classes=10):
    onehot = np.zeros((Y.shape[0], n_classes))
    onehot[np.arange(0, Y.shape[0]), Y] = 1
    return onehot


def bias_trick(X):
    return np.concatenate((X, np.ones((len(X), 1))), axis=1)


def check_gradient(X, targets, w, epsilon, computed_gradient):
    print("Checking gradient...")
    dw = np.zeros_like(w)
    for k in range(w.shape[0]):
        for j in range(w.shape[1]):
            new_weight1, new_weight2 = np.copy(w), np.copy(w)
            new_weight1[k, j] += epsilon
            new_weight2[k, j] -= epsilon
            loss1 = cross_entropy_loss(X, targets, new_weight1)
            loss2 = cross_entropy_loss(X, targets, new_weight2)
            dw[k, j] = (loss1 - loss2) / (2 * epsilon)
    maximum_abosulte_difference = abs(computed_gradient - dw).max()
    assert maximum_abosulte_difference <= epsilon**2, "Absolute error was: {}".format(
        maximum_abosulte_difference)


def softmax(X, w):
    a = X.dot(w.T)
    a_exp = np.exp(a)
    return a_exp / a_exp.sum(axis=1, keepdims=True)


# c)
def sigmoid(X, w):
    z = X.dot(w.T)
    return 1.0/(1.0 + np.exp(-z))
    

def forward(X, w):
    return softmax(X.T, w)


def calculate_accuracy(X, targets, w):
    output = forward(X, w)
    predictions = output.argmax(axis=1)
    targets = targets.argmax(axis=1)
    return (predictions == targets).mean()


def cross_entropy_loss(X, targets, w):
    output = forward(X, w)
    assert output.shape == targets.shape
    #output[output == 0] = 1e-8
    log_y = np.log(output)
    cross_entropy = -targets * log_y
    # print(cross_entropy.shape)
    return cross_entropy.mean()


def gradient_descent(X, targets, w, learning_rate, should_check_gradient):
    normalization_factor = X.shape[0] * \
        targets.shape[1]  # batch_size * num_classes
    outputs = forward(X, w)
    delta_k = - (targets - outputs)

    dw = delta_k.T.dot(X)
    dw = dw / normalization_factor  # Normalize gradient equally as loss normalization
    assert dw.shape == w.shape, "dw shape was: {}. Expected: {}".format(
        dw.shape, w.shape)

    if should_check_gradient:
        check_gradient(X, targets, w, 1e-2, dw)

    w = w - learning_rate * dw
    return w


# Global ftw
if not os.path.isdir(mnist.SAVE_PATH):
    mnist.init()

X_train, Y_train, X_test, Y_test = mnist.load()

# Pre-process data
# b) normalize [-1, 1]
X_train, X_test = X_train / 127.5 - 1, X_test / 127.5 - 1
X_train = bias_trick(X_train)
X_test = bias_trick(X_test)
Y_train, Y_test = onehot_encode(Y_train), onehot_encode(Y_test)

X_train, Y_train, X_val, Y_val = train_val_split(X_train, Y_train, 0.1)


# Hyperparameters
batch_size = 64
learning_rate = 0.5
num_batches = X_train.shape[0] // batch_size
should_gradient_check = False
check_step = num_batches // 10
max_epochs = 20

# Tracking variables
TRAIN_LOSS = []
TEST_LOSS = []
VAL_LOSS = []
TRAIN_ACC = []
TEST_ACC = []
VAL_ACC = []


def train_loop():
    w = np.zeros((Y_train.shape[1], X_train.shape[1]))
    for e in range(max_epochs):  # Epochs
        for i in tqdm.trange(num_batches):
            X_batch = X_train[i * batch_size:(i + 1) * batch_size]
            Y_batch = Y_train[i * batch_size:(i + 1) * batch_size]

            w = gradient_descent(
                X_batch,
                Y_batch,
                w,
                learning_rate,
                should_gradient_check)
            print(cross_entropy_loss(X_batch, Y_batch, w))
            if i % check_step == 0:
                # Loss
                TRAIN_LOSS.append(cross_entropy_loss(X_train, Y_train, w))
                TEST_LOSS.append(cross_entropy_loss(X_test, Y_test, w))
                VAL_LOSS.append(cross_entropy_loss(X_val, Y_val, w))

                TRAIN_ACC.append(calculate_accuracy(X_train, Y_train, w))
                VAL_ACC.append(calculate_accuracy(X_val, Y_val, w))
                TEST_ACC.append(calculate_accuracy(X_test, Y_test, w))
                if should_early_stop(VAL_LOSS):
                    print(VAL_LOSS[-4:])
                    print("early stopping.")
                    return w
    return w


def main():
    # print(X_val[0])
    # exit()

    model = Model()
    model.add_layer(64, sigmoid, 785)
    model.add_layer(10, softmax)
    model.evaluate(X_train)


    exit()


    w = train_loop()


    plt.plot(TRAIN_LOSS, label="Training loss")
    plt.plot(TEST_LOSS, label="Testing loss")
    plt.plot(VAL_LOSS, label="Validation loss")
    plt.legend()
    plt.ylim([0, 0.05])
    plt.show()
    
    plt.clf()
    plt.plot(TRAIN_ACC, label="Training accuracy")
    plt.plot(TEST_ACC, label="Testing accuracy")
    plt.plot(VAL_ACC, label="Validation accuracy")
    plt.ylim([0.8, 1.0])
    plt.legend()
    plt.show()
    
    plt.clf()
    
    w = w[:, :-1]  # Remove bias
    w = w.reshape(10, 28, 28)
    w = np.concatenate(w, axis=0)
    plt.imshow(w, cmap="gray")
    plt.show()


if __name__ == '__main__':
    main()
