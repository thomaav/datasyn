# Code modified from: https://github.com/hsjeong5/MNIST-for-Numpy


import numpy as np
from urllib import request
import gzip
import pickle
import os
import random
import matplotlib.pyplot as plt


SAVE_PATH = "data"
filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]


def download_mnist():
    os.makedirs(SAVE_PATH, exist_ok=True)
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading " + name[1] + "...")
        save_path = os.path.join(SAVE_PATH, name[1])
        request.urlretrieve(base_url+name[1], save_path)
    print("Download complete.")


def save_mnist():
    mnist = {}
    for name in filename[:2]:
        path = os.path.join(SAVE_PATH, name[1])
        with gzip.open(path, 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        path = os.path.join(SAVE_PATH, name[1])
        with gzip.open(path, 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    save_path = os.path.join(SAVE_PATH, "mnist.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")


def init_mnist():
    download_mnist()
    save_mnist()


def load_mnist():
    save_path = os.path.join(SAVE_PATH, "mnist.pkl")
    with open(save_path,'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


debug = 0
def run_times(n):
    global debug
    debug = debug + 1
    if debug == n:
        exit()


def sigmoid(X, w):
    z = np.dot(X, w)
    return 1.0/(1.0 + np.exp(-z))


def softmax(X, w):
    a = np.dot(X, np.transpose(w))
    ea = np.exp(a)
    sigmaea = np.ndarray.flatten(np.sum(ea, axis=1))
    return ea / sigmaea[:, None]


def gradient_descent(X, t, w, lr, lmbd=0):
    s = sigmoid(X, w)
    Ewdw = -(X * (t - s))
    Ewdw = Ewdw.mean(axis=0).reshape(-1, 1)
    reg = 2*lmbd * w
    return w - lr*Ewdw - lr*reg


def gradient_descent_softmax(X, t, w, lr, lmbd=0):
    y = softmax(X, w)
    Ewdw = -np.einsum('ij,jk->jki', X.T, t - y)
    Ewdw = Ewdw.mean(axis=0)
    reg = 2*lmbd * w
    return w - lr*Ewdw - lr*reg


def logreg_loss(t, o):
    # ln(<=0) is undefined and will net you -nan, so take care to add
    # an epsilon to work around it.
    epsilon = 1e-12
    o = np.clip(o, epsilon, 1.0-epsilon)
    Ew = -(t*np.log(o) + (1-t)*np.log(1-o))
    return Ew.mean()


def softmax_loss(t, o):
    # ln(<=0) is undefined and will net you -nan, so take care to add
    # an epsilon to work around it. Flattening is unneeded.
    epsilon = 1e-12
    o = np.clip(o, epsilon, 1.0-epsilon)
    Ew = -(t*np.log(o))
    return np.mean(np.sum(Ew, axis=1))


def l2_norm(w):
    return np.sum(np.square(w))


def round_guesses(y):
    y[y > 0.5] = 1
    y[y < 0.5] = 0


def onehot_encode_guesses(y):
    guesses = np.argmax(y, axis=1)
    onehot = np.zeros((guesses.shape[0], 10))
    onehot[np.arange(guesses.shape[0]), guesses] = 1
    return onehot


def percentage_wrong(x, y):
    return 100 * np.count_nonzero(x - y) / x.shape[0]


def percentage_wrong_onehot(x, y):
    # We need to divide by 2 here, as the one hot encoding will give
    # _two_ nonzero elements in their difference, as we have a false
    # positive.
    return percentage_wrong(x, y) / 2.0


def remove_unwanted_classes(X, Y, classes):
    unwanted_categories = []
    for i in range(len(X)):
        if Y[i] not in classes:
            unwanted_categories.append(i)

    X = np.delete(X, unwanted_categories, 0)
    Y = np.delete(Y, unwanted_categories, 0)
    Y = np.vectorize(lambda i: 1 if i == 2 else 0)(Y)
    return X, Y


def shuffle(X, Y):
    # This is probably easier with zipping and unzipping.
    rng = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng)
    np.random.shuffle(Y)


def evaluate_logreg(X, Y, w):
    train_y = sigmoid(X, w)
    train_loss = logreg_loss(Y, train_y)
    round_guesses(train_y)
    return train_loss, percentage_wrong(train_y, Y)


def evaluate_softmax(X, Y, w):
    y = softmax(X, w)
    loss = softmax_loss(Y, y)
    y = onehot_encode_guesses(y)
    return loss, percentage_wrong_onehot(y, Y)


def horizontal_subplots(weights):
    fig = plt.figure(figsize=(12, 8))
    print(len(weights))
    for nplot in range(len(weights)):
        fig.add_subplot()
        subplot = plt.subplot(1, 5, nplot+1)
        subplot.set_title("λ = " + str(10**(-nplot)))
        subplot.axis('off')
        plt.imshow(np.reshape(weights[nplot-1][:-1], (28, 28)), cmap='gray')
    plt.show()


def logistic_regression(X_train, Y_train, X_test, Y_test):
    # Setup hyperparameters. There are better ways of initializing the
    # weights, but this does just okay for this classification task.
    epochs = 200
    batch_size = 32
    lr = lr0 = 0.0005
    k = 0.005
    w = np.zeros((len(X_train[0]), 1))

    # If we want overfitting, we need to reduce the size of the
    # training set.
    # X_train = X_train[:10000]
    # Y_train = Y_train[:10000]
    # X_test = X_test[-2000:]
    # Y_test = Y_test[-2000:]

    # Remove non- 2s and 3s. t-values should be set to 1 for 2s and 0
    # for 3s.
    X_train, Y_train = remove_unwanted_classes(X_train, Y_train, [2, 3])
    X_test, Y_test = remove_unwanted_classes(X_test, Y_test, [2, 3])

    # Normalize inputs (as opposed to standardizing, normalizing
    # should be just fine here).
    X_train = X_train / 255
    X_test = X_test / 255
    Y_train = Y_train[:, np.newaxis]
    Y_test = Y_test[:, np.newaxis]

    # We want a validation set, as we don't want to 'snoop' on our
    # test set. There are probably prettier ways to do this than just
    # shuffling and picking the first 10% as validation set.
    validation_size = int(X_train.shape[0]*0.10)
    shuffle(X_train, Y_train)

    X_val = X_train[:validation_size]
    Y_val = Y_train[:validation_size]
    X_train = X_train[validation_size:]
    Y_train = Y_train[validation_size:]

    # We need some counters to compute statistics. This is shamefully
    # ugly, but they were added as we were going along.
    batches = X_train.shape[0] // batch_size
    train_losses = []
    train_percentages = []
    val_losses = []
    val_percentages = []
    test_losses = []
    test_percentages = []
    val_reg_accuracies = []
    weight_lengths = []
    trained_weights = []

    # Actual training. Note that we are doing five sessions of
    # training to create plots of different lambdas for
    # regularization.
    training_sessions = 5
    for i in range(training_sessions):
        print("Starting training run", str(i+1) + "..")

        # Reset from last training run.
        val_reg_accuracies.append([])
        weight_lengths.append([])
        lr = 0.0005
        w = np.zeros((len(X_train[0]), 1))

        for t in range(epochs):
            # Shuffle the training data to be able to bounce out of local
            # minima (coupled with mini batches, that is).
            shuffle(X_train, Y_train)

            # Actual gradient descent work.
            for j in range(batches):
                X_batch = X_train[j*batch_size:(j+1)*batch_size]
                Y_batch = Y_train[j*batch_size:(j+1)*batch_size]
                w = gradient_descent(X_batch, Y_batch, w, lr, lmbd=10**(-i))

                # Training statistics. Woeful.
                if j % (batches // 20) == 0:
                    train_loss, train_percentage_wrong = evaluate_logreg(X_train, Y_train, w)
                    val_loss, val_percentage_wrong = evaluate_logreg(X_val, Y_val, w)
                    test_loss, test_percentage_wrong = evaluate_logreg(X_test, Y_test, w)

                    train_losses.append(train_loss)
                    train_percentages.append(100 - train_percentage_wrong)

                    val_losses.append(val_loss)
                    val_percentages.append(100 - val_percentage_wrong)

                    test_losses.append(test_loss)
                    test_percentages.append(100 - test_percentage_wrong)

                    val_reg_accuracies[i].append(100 - val_percentage_wrong)
                    weight_lengths[i].append(l2_norm(w))

            # Debugging method of visualizing data as we go. This
            # would be factored out if we were to use this any
            # further.
            # if t == output_epoch:
            #     plt.figure(figsize=(12, 8))
            #     plt.ylim([50,100])
            #     plt.plot(val_losses, label="Validation loss")
            #     plt.plot(val_percentages, label="Validation percentages")
            #     plt.legend()
            #     plt.show()

            # Print some statistics for each epoch. Carriage return
            # could make this easier to look at.
            print("Training loss:\t", round(train_loss, 6), " -- ", round(train_percentage_wrong, 4), "% wrong", end="      ")
            print("Validation loss:", round(val_loss, 6), " -- ", round(val_percentage_wrong, 4), "% wrong")

            # If the last four test losses are strictly incrasing we stop
            # early.
            previous_losses = val_losses[-4:]
            if len(previous_losses) > 4 \
               and all(x < y for x, y in zip(previous_losses, previous_losses[1:])):
                print("Stopping early after %d epochs." % t)
                break

            # Anneal the learning rate (exponential decay).
            lr = lr0 * np.exp(-k*t)

            # Quit out after a fitting amount of epochs for now.
            if t == 20:
                trained_weights.append(w)
                break

    # Plots are usually shown from here.
    # horizontal_subplots(trained_weights)


def softmax_regression(X_train, Y_train, X_test, Y_test):
    # Setup hyperparameters. There are better ways of initializing the
    # weights, but this does just okay for this classification
    # task. Note that the layer now has multiple arrays of weights.
    epochs = 200
    batch_size = 32
    lr = lr0 = 0.01
    k = 0.0005
    w = np.array([np.zeros((len(X_train[0])))]*10)

    # Normalize inputs (as opposed to standardizing, normalizing
    # should be just fine here).
    X_train = X_train / 255
    X_test = X_test / 255

    X_train = X_train[:10000]
    Y_train = Y_train[:10000]

    # One-hot encode outputs.
    onehot = np.zeros((Y_train.shape[0], 10))
    onehot[np.arange(Y_train.shape[0]), Y_train] = 1
    Y_train = onehot

    onehot = np.zeros((Y_test.shape[0], 10))
    onehot[np.arange(Y_test.shape[0]), Y_test] = 1
    Y_test = onehot

    # Statistics.
    batches = X_train.shape[0] // batch_size
    train_losses = []
    val_losses = []
    test_losses = []
    train_percentages = []
    val_percentages = []
    test_percentages = []

    # We want a validation set, as we don't want to 'snoop' on our
    # test set. There are probably prettier ways to do this than just
    # shuffling and picking the first 10% as validation set.
    validation_size = int(X_train.shape[0]*0.10)
    shuffle(X_train, Y_train)

    X_val = X_train[:validation_size]
    Y_val = Y_train[:validation_size]
    X_train = X_train[validation_size:]
    Y_train = Y_train[validation_size:]

    for t in range(epochs):
        # Shuffle the training data to be able to bounce out of local
        # minima (coupled with mini batches, that is).
        shuffle(X_train, Y_train)

        # Actual gradient descent work.
        for j in range(X_train.shape[0] // batch_size):
            X_batch = X_train[j*batch_size:(j+1)*batch_size]
            Y_batch = Y_train[j*batch_size:(j+1)*batch_size]
            w = gradient_descent_softmax(X_batch, Y_batch, w, lr, lmbd=0)

            # Training statistics.
            if j % (batches // 20) == 0:
                train_loss, train_percentage_wrong = evaluate_softmax(X_train, Y_train, w)
                val_loss, val_percentage_wrong = evaluate_softmax(X_val, Y_val, w)
                test_loss, test_percentage_wrong = evaluate_softmax(X_test, Y_test, w)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                test_losses.append(test_loss)
                train_percentages.append(100 - train_percentage_wrong)
                val_percentages.append(100 - val_percentage_wrong)
                test_percentages.append(100 - test_percentage_wrong)

        # Print some evaluation for each epoch.
        print(evaluate_softmax(X_train, Y_train, w))

        # Anneal the learning rate (exponential decay).
        lr = lr0 * np.exp(-k*t)

        # Early stopping was implemented for logistic regression, just
        # stop after an appropriate amount of epochs here.
        if t == 60:
            break

        if t == 50:
            plt.figure(figsize=(12, 8))
            plt.ylim([10,100])
            plt.plot(train_percentages, label="Training percentage")
            plt.plot(val_percentages, label="Validation percentage")
            plt.plot(test_percentages, label="Test percentage")
            plt.legend()
            plt.show()


def main():
    if not os.path.isdir(SAVE_PATH):
        init_mnist()

    # Use _all_ the images. Might as well.
    X_train, Y_train, X_test, Y_test = load_mnist()

    # Append zeroes for bias trick.
    X_train = np.c_[X_train, np.ones(len(X_train))]
    X_test = np.c_[X_test, np.ones(len(X_test))]

    # logistic_regression(X_train, Y_train, X_test, Y_test)
    softmax_regression(X_train, Y_train, X_test, Y_test)


if __name__ == '__main__':
    main()
