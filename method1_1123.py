import numpy as np
# Set precision
np.set_printoptions(precision=4, suppress=True)

# --- 1. Activation function: Sigmoid ---
def activation(A):
    # Clamping A to prevent overflow in np.exp()
    A = np.clip(A, -500, 500)
    return 1.0 / (1.0 + np.exp(-A))

# Sigmoid derivative
def activation_derivative(Y):
    return Y * (1.0 - Y)

def calculate_overall_E(weights_biases, training_data):
    w1, w2, w3, w4, w5, w6, b1, b2, b3 = weights_biases
    
    total_E = 0.0
    
    for data in training_data:
        x1, x2, d = data[0], data[1], data[2]
        
        # Forward Propagation (FFBP)
        # Hidden Layer 1 (h1)
        activity1 = w1 * x1 + w2 * x2 + b1
        y1 = activation(activity1)
        # Hidden Layer 2 (h2)
        activity2 = w3 * x1 + w4 * x2 + b2
        y2 = activation(activity2)
        # Output Layer (y3)
        activity3 = w5 * y1 + w6 * y2 + b3
        y3 = activation(activity3)
        
        # Calculate single sample error E_p
        error_output = d - y3
        E_p = 0.5 * error_output**2
        total_E += E_p
    
    return total_E

# --- 2. Training Function ---
def train_and_test_method1():
    
    training_data = [
        [0.90, 0.87, 1.0], 
        [1.31, 0.75, 1.0], 
        [2.48, 1.14, 0.0], 
        [0.41, 1.87, 0.0], 
        [2.45, 0.52, 0.0], 
        [2.54, 2.97, 1.0], 
        [0.07, 0.09, 1.0], 
        [1.32, 1.96, 0.0], 
        [0.94, 0.34, 1.0], 
        [1.75, 2.21, 0.0]
    ]
    # --- 2.1 Initialization Function ---
    
    # Weight initialization
    w1, w2, w3, w4 = 0.3, 0.3, 0.3, 0.3  # w1:x1->h1, w2:x2->h1, w3:x1->h2, w4:x2->h2
    w5, w6 = 0.8, 0.8  # w5:h1->out, w6:h2->out
    
    # Bias initialization
    b1, b2, b3 = 0.0, 0.0, 0.0 # b1:h1 bias, b2:h2 bias, b3:Output bias
    
    eta = 1.0  # Learning rate
    
    NUM_CYCLES = 30

    # --- 2.2 Training Loop (30 Cycles) ---
    for cycle in range(1, NUM_CYCLES + 1):
        for data in training_data:
            x1, x2, d = data[0], data[1], data[2]
            
            # -------------------
            # Forward Propagation
            # -------------------
            # Hidden Layer 1 (h1)
            activity1 = w1 * x1 + w2 * x2 + b1
            y1 = activation(activity1)
            # Hidden Layer 2 (h2)
            activity2 = w3 * x1 + w4 * x2 + b2
            y2 = activation(activity2)
            # Output Layer (y3)
            activity3 = w5 * y1 + w6 * y2 + b3
            y3 = activation(activity3)
            
            # ---------------
            # Backpropagation
            # ---------------
            
            # Output Layer Delta (delta3)
            error_output = d - y3
            delta3 = activation_derivative(y3) * error_output
            
            # Hidden Layer Delta (delta1, delta2)
            error_hidden1 = delta3 * w5
            delta1 = activation_derivative(y1) * error_hidden1
            
            error_hidden2 = delta3 * w6
            delta2 = activation_derivative(y2) * error_hidden2
        
            # -------------------------
            # Weight and Bias Updates
            # -------------------------
            
            # w5, w6, b3 (Hidden to Output)
            w5 = w5 + eta * delta3 * y1
            w6 = w6 + eta * delta3 * y2
            b3 = b3 + eta * delta3 # Output bias update
            
            # w1, w2, b1 (Input to Hidden 1)
            w1 = w1 + eta * delta1 * x1
            w2 = w2 + eta * delta1 * x2
            b1 = b1 + eta * delta1 # Hidden 1 bias update
            
            # w3, w4, b2 (Input to Hidden 2)
            w3 = w3 + eta * delta2 * x1
            w4 = w4 + eta * delta2 * x2
            b2 = b2 + eta * delta2 # Hidden 2 bias update

        current_weights = (w1, w2, w3, w4, w5, w6, b1, b2, b3)
        overall_E = calculate_overall_E(current_weights, training_data)
    
    print("-" * 70)
    print(f"  w1={w1:.7f}, w2={w2:.7f}, w3={w3:.7f}, w4={w4:.7f}")
    print(f"  w5={w5:.7f}, w6={w6:.7f}")
    print(f"  b1={b1:.7f}, b2={b2:.7f}, b3={b3:.7f}")
    print(f"Overall Big E: {overall_E:.6f}")
    print("="*70)
    

train_and_test_method1()