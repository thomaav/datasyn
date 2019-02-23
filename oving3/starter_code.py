import os
import matplotlib.pyplot as plt
import torch
import tqdm
import dataloaders
from torch import nn
from dataloaders import load_cifar10
from utils import to_cuda, compute_loss_and_accuracy


def init_xavier(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


class ExampleModel(nn.Module):
    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        num_filters = 32  # Set number of filters in first conv layer

        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        # The output of feature_extractor will be [batch_size, num_filters, 16, 16]
        self.num_output_features = 32*16*16
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, num_classes),
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """

        # Run image through convolutional layers
        x = self.feature_extractor(x)
        # Reshape our input to (batch_size, num_output_features)
        x = x.view(-1, self.num_output_features)
        # Forward pass through the fully-connected layers.
        x = self.classifier(x)
        return x


class SimpleModel(nn.Module):
    def __init__(self, image_channels, num_classes):
        super().__init__()

        # Conv layers for feature extraction.
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # FF layer for classification.
        self.classifier = nn.Sequential(
            nn.Linear(128*4*4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )


    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """

        # Run image through convolutional layers
        x = self.feature_extractor(x)
        # Reshape our input to (batch_size, num_output_features)
        x = x.view(-1, 128*4*4)
        # Forward pass through the fully-connected layers.
        x = self.classifier(x)
        return x


class GoodModel(nn.Module):
    def __init__(self, image_channels, num_classes):
        super().__init__()

        # Conv for feature extraction.
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),


            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),



            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),



            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),



            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # self.num_output_features = 128*29*29
        self.num_output_features = 128*4*4

        # Xavier init all weights.
        # self.apply(init_xavier)

        # FF for classification.
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, num_classes),
            nn.Dropout(0.15),
        )


    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """

        # Run image through convolutional layers
        x = self.feature_extractor(x)
        # Reshape our input to (batch_size, num_output_features)
        x = x.view(-1, self.num_output_features)
        # Forward pass through the fully-connected layers.
        x = self.classifier(x)
        return x


class GoodestModel(nn.Module):
    def __init__(self, image_channels, num_classes):
        super().__init__()

        # Conv for feature extraction.
        self.feature_extractor = nn.Sequential(
            # 64
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),



            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.1),


            # 128
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),



            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2),


            # 256
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),



            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),



            # 512
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),



            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
        )

        self.num_output_features = 512*2*2

        # FF for classification. Size avg. of out from CNN and out for
        # classification, (2048 + 10) / 2.
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )


    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """

        # Run image through convolutional layers
        x = self.feature_extractor(x)
        # Reshape our input to (batch_size, num_output_features)
        x = x.view(-1, self.num_output_features)
        # Forward pass through the fully-connected layers.
        x = self.classifier(x)
        return x


class Trainer:
    def __init__(self):
        """
        Initialize our trainer class.
        Set hyperparameters, architecture, tracking variables etc.
        """
        # Define hyperparameters
        self.epochs = 100
        self.batch_size = 64
        self.learning_rate = 5e-2
        self.early_stop_count = 4

        # Task1
        # self.batch_size = 64
        # self.learning_rate = 5e-2

        # Task2 first good model
        # self.batch_size = 64
        # Adam with lr=5e-4, decay=0.001 or so
        # Adding xavier is fine
        # Reached 78% on epoch 10.

        # Architecture

        # Since we are doing multi-class classification, we use the CrossEntropyLoss
        self.loss_criterion = nn.CrossEntropyLoss()
        # Initialize the mode
        self.model = GoodestModel(image_channels=3, num_classes=10)
        # Transfer model to GPU VRAM, if possible.
        self.model = to_cuda(self.model)

        # Init xavier weights
        self.model.apply(init_xavier)

        # Define our optimizer. SGD = Stochastich Gradient Descent
        # self.optimizer = torch.optim.SGD(self.model.parameters(),
        #                                  self.learning_rate)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=6e-4)

        # Load our dataset
        self.dataloader_train, self.dataloader_val, self.dataloader_test = load_cifar10(self.batch_size)

        self.validation_check = len(self.dataloader_train)

        # Tracking variables
        self.VALIDATION_LOSS = []
        self.TEST_LOSS = []
        self.TRAIN_LOSS = []
        self.TRAIN_ACC = []
        self.VALIDATION_ACC = []
        self.TEST_ACC = []

    def validation_epoch(self):
        """
            Computes the loss/accuracy for all three datasets.
            Train, validation and test.
        """
        self.model.eval()

        # Compute for training set
        train_loss, train_acc = compute_loss_and_accuracy(
            self.dataloader_train, self.model, self.loss_criterion
        )
        self.TRAIN_ACC.append(train_acc)
        self.TRAIN_LOSS.append(train_loss)

        # Compute for validation set
        validation_loss, validation_acc = compute_loss_and_accuracy(
            self.dataloader_val, self.model, self.loss_criterion
        )
        self.VALIDATION_ACC.append(validation_acc)
        self.VALIDATION_LOSS.append(validation_loss)
        # Compute for testing set
        test_loss, test_acc = compute_loss_and_accuracy(
            self.dataloader_test, self.model, self.loss_criterion
        )
        self.TEST_ACC.append(test_acc)
        self.TEST_LOSS.append(test_loss)

        print("Current training loss:", train_loss, " Accuracy:", train_acc)
        print("Current validation loss:", validation_loss, " Accuracy:", validation_acc)
        print("Current test loss:", test_loss, " Accuracy:", test_acc)

        self.model.train()

    def should_early_stop(self):
        """
        Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss list.
        if len(self.VALIDATION_LOSS) < self.early_stop_count:
            return False
        # We only care about the last [early_stop_count] losses.
        relevant_loss = self.VALIDATION_LOSS[-self.early_stop_count:]
        previous_loss = relevant_loss[0]
        for current_loss in relevant_loss[1:]:
            # If the next loss decrease, early stopping criteria is not met.
            if current_loss < previous_loss:
                return False
            previous_loss = current_loss
        return True

    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """
        # Track initial loss/accuracy
        # self.validation_epoch()
        for epoch in range(self.epochs):
            print("Epoch: {e}".format(e=epoch+1))
            # Perform a full pass through all the training samples
            for batch_it, (X_batch, Y_batch) in enumerate(tqdm.tqdm(self.dataloader_train)):
                # X_batch is the CIFAR10 images. Shape: [batch_size, 3, 32, 32]
                # Y_batch is the CIFAR10 image label. Shape: [batch_size]
                # Transfer images / labels to GPU VRAM, if possible
                X_batch = to_cuda(X_batch)
                Y_batch = to_cuda(Y_batch)

                # Perform the forward pass
                predictions = self.model(X_batch)
                # Compute the cross entropy loss for the batch
                loss = self.loss_criterion(predictions, Y_batch)

                # Backpropagation
                loss.backward()

                # Gradient descent step
                self.optimizer.step()

                # Reset all computed gradients to 0
                self.optimizer.zero_grad()

            # Compute loss/accuracy for all three datasets.
            self.validation_epoch()
            # Check early stopping criteria.
            if self.should_early_stop():
                print("Early stopping.")
                return


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()

    os.makedirs("plots", exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(12, 8))
    plt.title("Cross Entropy Loss")
    plt.plot(trainer.VALIDATION_LOSS, label="Validation loss")
    plt.plot(trainer.TRAIN_LOSS, label="Training loss")
    plt.plot(trainer.TEST_LOSS, label="Testing Loss")
    plt.legend()
    plt.savefig(os.path.join("plots", "final_loss.png"))
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.title("Accuracy")
    plt.plot(trainer.VALIDATION_ACC, label="Validation Accuracy")
    plt.plot(trainer.TRAIN_ACC, label="Training Accuracy")
    plt.plot(trainer.TEST_ACC, label="Testing Accuracy")
    plt.legend()
    plt.savefig(os.path.join("plots", "final_accuracy.png"))
    plt.show()

    print("Final test accuracy:", trainer.TEST_ACC[-trainer.early_stop_count])
    print("Final validation accuracy:", trainer.VALIDATION_ACC[-trainer.early_stop_count])
