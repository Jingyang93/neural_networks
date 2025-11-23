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

# --- 2. Training Function ---
def train_and_test_method1():
    
    # --- 2.1 Initialization Function ---
    
    # Weight initialization
    w1, w2, w3, w4 = 0.3, 0.3, 0.3, 0.3  # w1:x1->h1, w2:x2->h1, w3:x1->h2, w4:x2->h2
    w5, w6 = 0.8, 0.8  # w5:h1->out, w6:h2->out
    
    # Bias initialization
    b1, b2, b3 = 0.0, 0.0, 0.0 # b1:h1 bias, b2:h2 bias, b3:Output bias
    
    eta = 1.0  # Learning rate
    
    X1, D1 = [1.0, 1.0], 0.9  # Pair 1
    X2, D2 = [-1.0, -1.0], 0.05 # Pair 2
    
    NUM_CYCLES = 15
    
    print("="*70)
    print("Method 1 training")
    print(f"P1: {X1} -> {D1}, P2: {X2} -> {D2}")
    print("="*70)

    # --- 2.2 Training Loop (15 Cycles) ---
    for cycle in range(1, NUM_CYCLES + 1):
        
        # --- Step1: Train on P1 ---
        x1, x2, d = X1[0], X1[1], D1
        
        # ------------------------------------
        # Forward Propagation (FFBP Step 1: P1)
        # ------------------------------------
        # Hidden Layer 1 (h1)
        activity1 = w1 * x1 + w2 * x2 + b1
        y1 = activation(activity1)
        # Hidden Layer 2 (h2)
        activity2 = w3 * x1 + w4 * x2 + b2
        y2 = activation(activity2)
        # Output Layer (y3)
        activity3 = w5 * y1 + w6 * y2 + b3
        y3 = activation(activity3)
        
        # ------------------------------------
        # Backpropagation (FFBP Step 1: P1)
        # ------------------------------------
        
        # Output Layer Delta (delta3)
        error_output = d - y3
        delta3 = activation_derivative(y3) * error_output
        
        # Hidden Layer Delta (delta1, delta2)
        error_hidden1 = delta3 * w5
        delta1 = activation_derivative(y1) * error_hidden1
        
        error_hidden2 = delta3 * w6
        delta2 = activation_derivative(y2) * error_hidden2
        
        # ------------------------------------
        # Weight and Bias Updates (Update Step 1: P1)
        # ------------------------------------
        
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


        # --- Step2: Train on P2 ---
        x1, x2, d = X2[0], X2[1], D2

        # ------------------------------------
        # Forward propagation (FFBP Step 2: P2)
        # ------------------------------------
        # Hidden Layer 1 (h1)
        activity1 = w1 * x1 + w2 * x2 + b1
        y1 = activation(activity1)
        # Hidden Layer 2 (h2)
        activity2 = w3 * x1 + w4 * x2 + b2
        y2 = activation(activity2)
        # Output Layer (y3)
        activity3 = w5 * y1 + w6 * y2 + b3
        y3 = activation(activity3)
        
        # ------------------------------------
        # Backprpagation (FFBP Step 2: P2)
        # ------------------------------------
        
        # Output Layer Delta (delta3)
        error_output = d - y3
        delta3 = activation_derivative(y3) * error_output
        
        # Hidden Layer Delta (delta1, delta2)
        error_hidden1 = delta3 * w5
        delta1 = activation_derivative(y1) * error_hidden1
        
        error_hidden2 = delta3 * w6
        delta2 = activation_derivative(y2) * error_hidden2
        
        # ------------------------------------
        # Weights and Bias Updates (Update Step 2: P2)
        # ------------------------------------
        
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

    print("-" * 70)
    print(f"  w1={w1:.7f}, w2={w2:.7f}, w3={w3:.7f}, w4={w4:.7f}")
    print(f"  w5={w5:.7f}, w6={w6:.7f}")
    print(f"  b1={b1:.7f}, b2={b2:.7f}, b3={b3:.7f}")
    print("="*70)
    
    
    results = {}
    
    # --- 1: X=[1.0, 1.0], D=0.9  ---
    x1, x2, d = X1[0], X1[1], D1
    
    activity1 = w1 * x1 + w2 * x2 + b1
    y1 = activation(activity1)
    activity2 = w3 * x1 + w4 * x2 + b2
    y2 = activation(activity2)
    activity3 = w5 * y1 + w6 * y2 + b3
    y3 = activation(activity3) # Q1: Output
    
    error_output = d - y3
    E = 0.5 * error_output**2 # Q2: Big E
    
    results["Q1_Output"] = y3
    results["Q2_Big_E"] = E

    # --- 2: X=[-1.0, -1.0], D=0.05  ---
    x1, x2, d = X2[0], X2[1], D2
    
    activity1 = w1 * x1 + w2 * x2 + b1
    y1 = activation(activity1)
    activity2 = w3 * x1 + w4 * x2 + b2
    y2 = activation(activity2)
    activity3 = w5 * y1 + w6 * y2 + b3
    y3 = activation(activity3) # Q3: Output
    
    error_output = d - y3
    E = 0.5 * error_output**2 # Q4: Big E
    
    results["Q3_Output"] = y3
    results["Q4_Big_E"] = E
    
    print(f"1.  [1.0, 1.0] output (target 0.9): {results['Q1_Output']:.4f}")
    print(f"2.  [1.0, 1.0] Big E (target 0.9): {results['Q2_Big_E']:.4f}")
    print(f"3.  [-1.0, -1.0] output (target 0.05): {results['Q3_Output']:.4f}")
    print(f"4.  [-1.0, -1.0] Big E (target 0.05): {results['Q4_Big_E']:.4f}")

train_and_test_method1()