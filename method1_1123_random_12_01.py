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

# Compute Big E over all training samples
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


# Hard-code full dataset (Data Item 1–20)
full_data = [
    [0.90, 0.87, 1.0],   # 1
    [1.81, 1.02, 0.0],   # 2
    [1.31, 0.75, 1.0],   # 3
    [2.36, 1.60, 0.0],   # 4
    [2.48, 1.14, 0.0],   # 5
    [2.17, 2.08, 1.0],   # 6
    [0.41, 1.87, 0.0],   # 7
    [2.85, 2.91, 1.0],   # 8
    [2.45, 0.52, 0.0],   # 9
    [1.05, 1.93, 0.0],   #10
    [2.54, 2.97, 1.0],   #11
    [2.32, 1.73, 0.0],   #12
    [0.07, 0.09, 1.0],   #13
    [1.86, 1.31, 0.0],   #14
    [1.32, 1.96, 0.0],   #15
    [1.45, 2.19, 0.0],   #16
    [0.94, 0.34, 1.0],   #17
    [0.28, 0.71, 1.0],   #18
    [1.75, 2.21, 0.0],   #19
    [2.49, 1.52, 0.0],   #20
]

# odd index → training, even index → testing
training_data = [full_data[i] for i in range(0, len(full_data), 2)]
testing_data  = [full_data[i] for i in range(1, len(full_data), 2)]

# ============
# Testing Eval
# ============
def evaluate_testing_data(weights_biases):
    w1, w2, w3, w4, w5, w6, b1, b2, b3 = weights_biases

    total_E = 0.0
    print("\n--- Testing Predictions ---")

    for idx, (x1, x2, d) in enumerate(testing_data):
        a1 = w1 * x1 + w2 * x2 + b1
        y1 = activation(a1)

        a2 = w3 * x1 + w4 * x2 + b2
        y2 = activation(a2)

        a3 = w5 * y1 + w6 * y2 + b3
        y3 = activation(a3)

        E_p = 0.5 * (d - y3)**2
        total_E += E_p

        print(f"Test Item {idx+1}: x1={x1:.2f}, x2={x2:.2f}, d={d} --> y={y3:.6f}")

    print(f"\nTesting Big E = {total_E:.9f}")
    return total_E

# --- Training Function ---
def train_and_test_method1():
    # --- Random Initialization ---
    w1, w2, w3, w4 = np.random.uniform(-5, 5, 4)
    w5, w6 = np.random.uniform(-5, 5, 2)
    b1, b2, b3 = np.random.uniform(-5, 5, 3)

    params = [w1, w2, w3, w4, w5, w6, b1, b2, b3]
    step = 0.1

    best_E = calculate_overall_E(params, training_data)

    # Hill-Climbing: try to improve weights one-by-one
    for _ in range(50):   # 50 rounds is enough
        improved = False

        for idx in range(len(params)):
            original = params[idx]

            # Try +step
            params[idx] = original + step
            E_plus = calculate_overall_E(params, training_data)

            # Try -step
            params[idx] = original - step
            E_minus = calculate_overall_E(params, training_data)

            # Choose direction
            if E_plus < best_E and E_plus < E_minus:
                params[idx] = original + step
                best_E = E_plus
                improved = True

            elif E_minus < best_E:
                params[idx] = original - step
                best_E = E_minus
                improved = True

            else:
                params[idx] = original  # revert

        # If no improvement, shrink step size
        if not improved:
            step /= 10
            if step < 1e-6:
                break

    # After hill climbing, return best found
    return best_E, tuple(params)


# ======================================
# Repeat N random initializations (main)
# ======================================
def repeat_random_initializations(N=10000):

    best_E = float("inf")
    best_weights = None

    for i in range(N):
        E, weights = train_and_test_method1()
        if E < best_E:
            best_E = E
            best_weights = weights
        if i % 20 == 0:
            print(f"Run {i}: E={E:.6f} (best so far={best_E:.6f})")

    print("\n========= BEST RESULT =========")
    print(f"Best E = {best_E:.9f}")
    print("Best weights =")
    print(best_weights)

    # ---- NEW: Evaluate on Testing Data ----
    print("\n========= TESTING PERFORMANCE =========")
    test_E = evaluate_testing_data(best_weights)
    print(f"Testing Big E = {test_E:.9f}")

    return best_E, best_weights


train_and_test_method1()

best_E, best_weights = repeat_random_initializations(N=200)
