import numpy as np

def ReLU(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def sigmoid(x):
    """Sigmoid activation function for binary classification."""
    return 1 / (1 + np.exp(-x))

def softmax(x):
    """Softmax activation function for multiclass classification."""
    exps = np.exp(x - np.max(x))  # Shift for numerical stability
    return exps / np.sum(exps)

def FFNN(input_vector, mode="binary"):
    """
    Feedforward Neural Network.
    Parameters:
        input_vector (np.array): Input vector with four components.
        mode (str): "binary" for single-output, "multiclass" for multi-output.
    Returns:
        np.array or float: Probability output(s) depending on mode.
    """
    # Define weights and biases based on the uploaded networks
    
    # Input to Hidden Layer 1
    W1 = np.array([[0.5, 1.0, -0.8, -0.1], 
                   [0.9, -0.1, 0.4, 0.01]])
    b1 = np.array([0.1, -0.05])

    # Hidden Layer 1 to Hidden Layer 2
    W2 = np.array([[0.8, -0.7], 
                   [0.2, 0.3]])
    b2 = np.array([-0.05, 0.3])

    # Hidden Layer 2 to Output Layer (differs for binary vs multiclass)
    if mode == "binary":
        W3 = np.array([[0.9, -0.4]])
        b3 = np.array([-0.01])
    elif mode == "multiclass":
        W3 = np.array([[0.9, -0.3], 
                       [-0.3, 0.1], 
                       [0.4, -0.2]])
        b3 = np.array([0.01, 0.8, 0.05])
    else:
        raise ValueError("Mode must be either 'binary' or 'multiclass'")
    
    # Forward pass through layers with ReLU and final activation
    h1 = ReLU(np.dot(W1, input_vector) + b1)        # Hidden Layer 1
    h2 = ReLU(np.dot(W2, h1) + b2)                  # Hidden Layer 2
    output = np.dot(W3, h2) + b3                    # Output Layer (before activation)

    # Apply final activation function based on mode
    if mode == "binary":
        return sigmoid(output[0])  # Return a single probability for binary
    elif mode == "multiclass":
        return softmax(output)     # Return a vector of probabilities for multiclass

# Test the FFNN function
input_vector = np.array([0.5, -1.0, 0.3, 0.8])
binary_output = FFNN(input_vector, mode="binary")
multiclass_output = FFNN(input_vector, mode="multiclass")

binary_output, multiclass_output
v = np.array([10, 0, 12, 7])
print("Test oder so", FFNN(v , "multiclass"))