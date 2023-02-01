import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, array_of_neurons, array_of_activation_functions, learning_rate, momentum):
        super(Net, self).__init__()
        self.array_of_neurons = array_of_neurons
        self.array_of_activation_functions = array_of_activation_functions
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.running_loss = 0.0
        self.lambd = 0.01
        self.array_of_fc = nn.ModuleList()
        for i in range(len(array_of_neurons)-1):
            self.array_of_fc.append(nn.Linear(array_of_neurons[i], array_of_neurons[i+1]))
        self.array_of_activation_functions = nn.ModuleList()
        for i in range(len(array_of_activation_functions)):
            if array_of_activation_functions[i] == "sigmoid":
                self.array_of_activation_functions.append(nn.Sigmoid())
            elif array_of_activation_functions[i] == "relu":
                self.array_of_activation_functions.append(nn.ReLU())
            elif array_of_activation_functions[i] == "leaky_relu":
                self.array_of_activation_functions.append(nn.LeakyReLU())
            elif array_of_activation_functions[i] == "tanh":
                self.array_of_activation_functions.append(nn.Tanh())
            elif array_of_activation_functions[i] == "softmax":
                self.array_of_activation_functions.append(nn.Softmax(dim=-1))
            elif array_of_activation_functions[i] == "identity":
                self.array_of_activation_functions.append(nn.Identity())
            else:
                raise Exception("Activation function not found")
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.02, momentum=0.9)

    def forward(self, x):
        for i in range(len(self.array_of_fc)):
            x = self.array_of_fc[i](x)
            x = self.array_of_activation_functions[i](x)
        return x

    def set_weights_and_biases(self, flattend_list_of_weights_and_biases):
        modules = self.array_of_fc.modules()
        for module in modules:
            if isinstance(module, nn.Linear):
                module.weight.data = (flattend_list_of_weights_and_biases[:module.weight.shape[0]*module.weight.shape[1]].reshape(module.weight.shape))
                flattend_list_of_weights_and_biases = flattend_list_of_weights_and_biases[module.weight.shape[0]*module.weight.shape[1]:]
                module.bias.data = (flattend_list_of_weights_and_biases[:module.bias.shape[0]].reshape(module.bias.shape))
                flattend_list_of_weights_and_biases = flattend_list_of_weights_and_biases[module.bias.shape[0]:]

    def train(self, number_of_epoch, train_X: list, train_y: list, batch_size, print_loss=False, print_acc=False, l2=False, l1=False, test_X=None, test_y=None):
        for epoch in range(number_of_epoch):
            self.running_loss = 0.0
            for i in range(0, len(train_X), batch_size):
                # Get the mini-batch
                inputs = train_X[i:i+batch_size]
                labels = train_y[i:i+batch_size]

                # zero the parameter gradients
                self.optimizer.zero_grad()
                inputs = inputs.to(torch.float32)
                # forward + backward 
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                if l2 and loss < 0.1:
                    # Add L2 regularization to the loss
                    l2_reg = sum(p.pow(2).sum() for p in self.parameters())
                    loss += self.lambd * l2_reg * 0
                if l1 and loss < 0.1:
                    # Add L1 regularization to the loss
                    l1_reg = sum(p.abs().sum() for p in self.parameters())
                    loss += self.lambd * l1_reg
                loss.backward()
                self.optimizer.step()
                self.running_loss += loss.item()
            if self.running_loss / batch_size < 0.1:
                self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
            if self.running_loss / batch_size < 0.01:
                self.optimizer = optim.SGD(self.parameters(), lr=0.002, momentum=0.9)
            if print_loss and epoch % 20 == 0:
                print(f'Epoch [{epoch}/{number_of_epoch}], Loss: {self.running_loss / batch_size}')
            if print_acc and epoch % 100 == 0:
                if test_X is None:
                    self.print_acc(train_X, train_y)
                else:
                    self.print_acc(test_X, test_y)
            # shuffle the data
            zippe_lists = list(zip(train_X.tolist(), train_y.tolist()))
            random.shuffle(zippe_lists)
            train_X, train_y = zip(*zippe_lists)
            train_X, train_y = torch.tensor(train_X), torch.tensor(train_y)


    def print_acc(self, test_X, test_y):
        with torch.no_grad():
            correct = 0
            total = 0
            list_1 = []
            list_2 = []
            for x, y in zip(test_X, test_y):
                correct += torch.equal(torch.argmax(self(x)), torch.argmax(y))
                total += 1
                if torch.argmax(self(x)) == 0:
                    list_1.append(x)
                else:
                    list_2.append(x)
            print(f'Accuracy: {100 * correct / total}')
            plt.scatter([x[0] for x in list_1], [x[1] for x in list_1], color="blue")
            plt.scatter([x[0] for x in list_2], [x[1] for x in list_2], color="red")
            plt.pause(0.001)
