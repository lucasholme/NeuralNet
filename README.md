🧠 Neural Network from Scratch with NumPy
=========================================

This project implements a **fully connected neural network** from scratch using **NumPy** to classify handwritten digits from the **MNIST dataset**. The model is trained using the **softmax activation** for multi-class classification and **backpropagation** for learning.

🚀 Features
-----------

-   ✅ **Fully Connected Neural Network** -- Supports customizable layer sizes.
-   ✅ **Backpropagation & Gradient Descent** -- Implements weight updates manually.
-   ✅ **Softmax Activation** -- Suitable for multi-class classification tasks.
-   ✅ **ReLU Activation** -- Used for hidden layers.
-   ✅ **MNIST Dataset** -- Classifies handwritten digits (0-9).
-   ✅ **Training & Accuracy Evaluation** -- Prints loss and accuracy over epochs.

📥 Installation
---------------

Ensure you have **Python 3.x** installed on your system.

### 1️⃣ Clone the Repository

```
git clone https://github.com/your-username/nn-mnist.git
```

### 2️⃣ Install Dependencies

```
pip install numpy tensorflow
```

🛠️ Usage
---------

### 1️⃣ Run the Neural Network

```
python neural_network.py
```

### 2️⃣ How It Works

-   **Loads the MNIST dataset** -- Prepares training and testing data.
-   **Preprocesses data** -- Normalizes pixel values and one-hot encodes labels.
-   **Initializes network parameters** -- Randomly initializes weights and biases.
-   **Performs forward propagation** -- Computes activations for each layer.
-   **Computes loss** -- Uses categorical cross-entropy.
-   **Performs backward propagation** -- Updates weights using gradients.
-   **Trains the model** -- Runs for a specified number of epochs.
-   **Evaluates test accuracy** -- Prints final accuracy after training.

🔢 Neural Network Architecture
------------------------------

The model is a **3-layer neural network** with:

-   **Input Layer** -- 784 neurons (28×28 flattened pixels).
-   **Hidden Layer** -- 128 neurons, ReLU activation.
-   **Output Layer** -- 10 neurons, Softmax activation.

Example setup in `neural_network.py`:

```
layer_sizes = [784, 128, 10]
learning_rate = 0.01
epochs = 1000

nn = NeuralNetwork(layer_sizes, learning_rate)
nn.train(train_data, train_labels, test_data, test_labels, epochs)
```

🔥 Example Output
-----------------

During training, the network prints progress:

```
Epoch 0, Loss: 2.307, Accuracy: 11.35%
Epoch 100, Loss: 0.525, Accuracy: 88.75%
Epoch 500, Loss: 0.195, Accuracy: 95.43%
Epoch 1000, Loss: 0.103, Accuracy: 97.12%
Final Test Accuracy: 97.32%
```

🛑 Notes
--------

⚠️ **Manual Training** -- This is a pure NumPy implementation without TensorFlow/Keras optimizers.\
⚠️ **Performance** -- Training may take longer than GPU-accelerated frameworks.\
⚠️ **Customizable Layers** -- Modify `layer_sizes` to experiment with different architectures.

📜 License
----------

This project is open-source under the MIT License.
