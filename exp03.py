import numpy as np
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics



class CNN(nn.Module):

    def __init__(self, input_size, output_size):
        """
        The constructor. This is where you should initialize the architecture and structural
        components of the training algorithm (e.g., which loss function and optimization strategy
        to use)
        :param input_size: The number of inputs to the neural network.
        :param output_size: The number of outputs of the neural network.
        """
        super().__init__()
        self.architecture = self._initialize_architecture(input_size=input_size, output_size=output_size)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.architecture.parameters())

    @staticmethod
    def _initialize_architecture(input_size, output_size):
        """
        This private method instantiates the overarching architecture of the neural network.
        :param input_size: The number of inputs to the neural network.
        :param output_size: The number of outputs of the neural network.
        :return: The overarching architecture of the network.
        """

        # working architecture
        architecture = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(20,50,kernel_size=5),
            nn.MaxPool2d(kernel_size=2,stride= 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=800,out_features=250),
            nn.ReLU(),
            nn.Linear(250,10),
            nn.LogSoftmax(dim= 1)
            )

        return architecture

    def fit(self, x, y, epochs=25, batch_size=None, verbose=True):
        """
        Train the model to predict y when given x.
        :param x: The input/features data
        :param y: The the output/target data
        :param epochs: The number of times to iterate over the training data.
        :param batch_size: How many samples to consider at a time before updating network weights.
        sqrt(# of training instances) by default.
        :param verbose: Print a lot of info to the console about training.
        :return: None. Internally the weights of the network are updated.
        """
        if batch_size is None:
            batch_size = int(np.sqrt(len(x)))

        # Convert from Numpy to PyTorch tensor.
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        # Train for the given number of epochs.
        for epoch in range(epochs):
            if verbose:
                print("Working on Epoch", epoch, "of", epochs)

            # Let the model know we're starting another round of training
            self.architecture.train()

            # Organize into helper methods which automatically put things into
            # well formed batches for us.
            train_ds = TensorDataset(x, y)
            train_dl = DataLoader(train_ds, batch_size=batch_size)

            # Iterate over each minibatch
            for x_minibatch, y_minibatch in train_dl:
                # Feed through the architecture
                pred_minibatch = self.architecture(x_minibatch)
                #pred_minibatch = torch.reshape(pred_minibatch,(-1,))

                # Calculate the loss
                loss = self.loss_function(pred_minibatch, y_minibatch)

                # Apply backpropegation using the specified optimizer
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            if verbose:
                print("Loss at end of epoch:", loss)

    @staticmethod
    def _convert_to_one_hot_encoding(y):
        """
        Converts digit to a 1 hot encoded value.

        The mapping is as follows:
        0 -> [1, 0, 0, 0, ..., 0]
        1 -> [0, 1, 0, 0, ..., 0]
        2 -> [0, 0, 1, 0, ..., 0]
        ...
        9 -> [0, 0, 0, 0, ..., 1]

        :param y: The scalar value(s).
        :return: The 1 hot encoded vector(s).
        """
        y_labels = []
        y = y.astype("int32")
        for item in y:
            one_hot_encoder = np.zeros(10)
            one_hot_encoder[item] = 1
            y_labels.append(np.array(one_hot_encoder))

        return np.array(y_labels)




    def predict(self, x):
        """
        Predict an outcome given a sample of features.
        :param x: The features to use for a prediction.
        :return: The predicted outcome based on the features.
        """
        x = torch.from_numpy(x).float()
        # Let the model know we're in "evaluate" mode (as opposed to training)
        self.architecture.eval()
        # Feed through the architecture.
        result = self.architecture(x)
        # Convert back to numpy
        result = result.max(dim=1)[1].detach().numpy()

        return result


class Exp03:

    @staticmethod
    def load_train_test_data(file_path_prefix=""):
        """
        This method loads the training and testing data
        :param file_path_prefix: Any prefix needed to correctly locate the files.
        :return: x_train, y_train, x_test, y_test, which are to be numpy arrays.
        """

        minst_train = np.loadtxt(file_path_prefix + "mnist_train.csv", delimiter=',', dtype=np.int)
        minst_test = np.loadtxt(file_path_prefix + "mnist_test.csv", delimiter=',', dtype=np.int)
        x_train = []
        y_train = np.zeros(len(minst_train))

        x_test = []
        y_test = np.zeros(len(minst_test))

        for i in range(len(minst_train)):
            # training data block
            temp_x_train = minst_train[i][1:]
            x_train.append(np.reshape(temp_x_train,(28,28)))
            y_train[i] = minst_train[i][0]

        for i in range(len(minst_test)):
            temp_x_test = minst_test[i][1:]
            x_test.append(np.reshape(temp_x_test,(28,28)))
            y_test[i] = minst_test[i][0]

        x_train, y_train, x_test, y_test = np.array(x_train), y_train, np.array(x_test), y_test
        x_train = np.expand_dims(x_train,1)
        x_test = np.expand_dims(x_test,1)
        print(x_train.shape)

        return x_train, y_train, x_test, y_test


    @staticmethod
    def compute_mean_absolute_error(true_y_values, predicted_y_values):
        list_of_errors = []
        for true_y, pred_y in zip(true_y_values, predicted_y_values):
            error = abs(true_y - pred_y)
            list_of_errors.append(error)
        mean_abs_error = np.mean(list_of_errors)
        return mean_abs_error

    @staticmethod
    def compute_mean_absolute_percentage_error(true_y_values, predicted_y_values):
        list_of_perc_errors = []
        for true_y, pred_y in zip(true_y_values, predicted_y_values):
            error = abs((true_y - pred_y) / true_y)
            list_of_perc_errors.append(error)
            list_of_perc_errors.append(error)
        mean_abs_error = np.mean(list_of_perc_errors)
        return mean_abs_error

    @staticmethod
    def print_error_report(trained_model, x_train, y_train, x_test, y_test):
        print("\tEvaluating on Training Data")
        # Evaluating on training data is a less effective as an indicator of
        # accuracy in the wild. Since the model has already seen this data
        # before, it is a less realistic measure of error when given novel/unseen
        # inputs.
        #
        # The utility is in its use as a "sanity check" since a trained model
        # which preforms poorly on data it has seen before/used to train
        # indicates underlying problems (either more data or data preprocessing
        # is needed, or there may be a weakness in the model itself.

        y_train_pred = trained_model.predict(x_train)

        mean_absolute_error_train = Exp03.compute_mean_absolute_error(y_train, y_train_pred)
        mean_absolute_perc_error_train = Exp03.compute_mean_absolute_percentage_error(y_train, y_train_pred)

        print("\tMean Absolute Error (Training Data):", mean_absolute_error_train)
        # print("\tMean Absolute Percentage Error (Training Data):", mean_absolute_perc_error_train)
        print()

        print("\tEvaluating on Testing Data")
        # Is a more effective as an indicator of accuracy in the wild.
        # Since the model has not seen this data before, so is a more
        # realistic measure of error when given novel inputs.

        y_test_pred = trained_model.predict(x_test)

        mean_absolute_error_test = Exp03.compute_mean_absolute_error(y_test, y_test_pred)
        mean_absolute_perc_error_test = Exp03.compute_mean_absolute_percentage_error(y_test, y_test_pred)

        print("\tMean Absolute Error (Testing Data):", mean_absolute_error_test)
        # print("\tMean Absolute Percentage Error (Testing Data):", mean_absolute_perc_error_test)
        print()

    def run(self):
        start_time = datetime.now()
        print("Running Exp: ", self.__class__, "at", start_time)

        print("Loading Data")
        x_train, y_train, x_test, y_test = Exp03.load_train_test_data()



        print("Training Model...")

        #######################################################################
        # Complete this 2-step block of code using the variable name 'model' for
        # the linear regression model.
        # You can complete this by turning the given psuedocode to real code
        #######################################################################

        # (1) Initialize model;


        model = CNN(input_size = 20, output_size=10)

        # (2) Train model using the function 'fit' and the variables 'x_train'
        # and 'y_train'\

        # Reshape x_train and x_test to fit the model
        # convert copy of ytrain to a one hot encoding to use cross entopy loss
        y_train_copy = model._convert_to_one_hot_encoding(y_train)


        model.fit(x_train, y_train_copy)


        print("Training complete!")
        print()


        print("Evaluating Model")
        y_train_pred = model.predict(x_train)

        y_test_pred = model.predict(x_test)

        print("-- SciKit Learn Classification Report: Training Data")
        report_train = metrics.classification_report(y_train, y_train_pred)
        print(report_train)
        print("-- SciKit Learn Classification Report: Testing Data")
        report_test = metrics.classification_report(y_test, y_test_pred)
        print(report_test)
        # End and report time.
        end_time = datetime.now()
        print("Exp is over; completed at", datetime.now())
        total_time = end_time - start_time
        print("Total time to run:", total_time)
