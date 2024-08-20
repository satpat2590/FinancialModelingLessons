import mnist as yf
import ctypes
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matmul
import timeit

# Load the shared library
mylib = ctypes.CDLL('./libmylib.so')
mylib.say_hello.argtypes = [ctypes.c_char_p]


class NeuralNet():
    def __init__(self):
        # Define a transformation to normalize the data
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        # Download and load the training and test datasets
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        # Create data loaders
        self.trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
        self.testloader = DataLoader(testset, batch_size=64, shuffle=False)

        # Instantiate the model, define the loss function and the optimizer
        self.model = SimpleNN()
        self.criterion = nn.NLLLoss()  # Negative log-likelihood loss
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)  # Stochastic Gradient Descent

    def train_model(self):
        # Number of epochs (how many times we pass through the entire dataset)
        epochs = 5

        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in self.trainloader:
                self.optimizer.zero_grad()  # Zero the parameter gradients
                output = self.model(images)  # Forward pass
                loss = self.criterion(output, labels)  # Compute loss
                loss.backward()  # Backward pass
                self.optimizer.step()  # Update weights

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs} - Loss: {running_loss/len(self.trainloader)}")

        print("Training completed!")

    def evaluate_model(self):
        correct = 0
        total = 0

        with torch.no_grad():  # No need to calculate gradients for evaluation
            for images, labels in self.testloader:
                output = self.model(images)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy on the test set: {100 * correct / total:.2f}%")



class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Input layer to hidden layer
        self.fc2 = nn.Linear(128, 64)       # Hidden layer to another hidden layer
        self.fc3 = nn.Linear(64, 10)        # Hidden layer to output layer
        self.relu = nn.ReLU()               # ReLU activation function
        self.log_softmax = nn.LogSoftmax(dim=1)  # Log-Softmax for output layer

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input (batch_size, 28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.log_softmax(self.fc3(x))
        return x


# Function to test torch.mm
def test_torch_mm(a, b):
    return torch.mm(a, b)

# Function to test custom C++ matrix multiplication
def test_custom_mm(a, b):
    return matmul.matmul(a, b)

def torch_mm_comparison():
    # Create random matrices of a larger size for the test
    size = 1000
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    # Warm-up to avoid including any overhead in timing
    test_torch_mm(a, b)
    test_custom_mm(a, b)

    # Measure the time taken by torch.mm
    torch_mm_time = timeit.timeit(lambda: test_torch_mm(a, b), number=10)
    print(f"torch.mm average time over 10 runs: {torch_mm_time / 10:.6f} seconds")

    # Measure the time taken by the custom C++ matrix multiplication
    custom_mm_time = timeit.timeit(lambda: test_custom_mm(a, b), number=10)
    print(f"Custom C++ matrix_mul average time over 10 runs: {custom_mm_time / 10:.6f} seconds")

    # Verify that the results are the same
    result_torch = test_torch_mm(a, b)
    result_custom = test_custom_mm(a, b)
    print("Difference between results:", torch.abs(result_torch - result_custom).sum().item())


if __name__=="__main__":
    neuneu = NeuralNet()
    neuneu.train_model()
    neuneu.evaluate_model()