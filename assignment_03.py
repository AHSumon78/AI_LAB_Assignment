import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# -------------------------------
# 1. Generate dataset
# -------------------------------
x = np.linspace(-10, 10, 300).reshape(-1, 1)

# Equations
y_linear = 5*x + 10
y_quadratic = 3*x**2 + 5*x + 10
y_cubic = 4*x**3 + 3*x**2 + 5*x + 10

# -------------------------------
# 2. Split data
# -------------------------------
def split_data(x, y):
    X_train, X_temp, Y_train, Y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

data_sets = {}
for name, y in zip(['Linear','Quadratic','Cubic'], [y_linear, y_quadratic, y_cubic]):
    data_sets[name] = split_data(x, y)

# -------------------------------
# 3. Build and train models
# -------------------------------
def build_and_train_model(X_train, Y_train, X_val, Y_val, hidden_layers=1, neurons=10, epochs=500):
    model = Sequential()
    model.add(Dense(neurons, input_dim=1, activation='relu'))
    for _ in range(hidden_layers-1):
        model.add(Dense(neurons, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, verbose=0)
    return model

models = {}
for name, data in data_sets.items():
    X_train, Y_train, X_val, Y_val, X_test, Y_test = data
    if name=='Linear':
        hidden, neurons = 1, 5
    elif name=='Quadratic':
        hidden, neurons = 2, 10
    else:
        hidden, neurons = 3, 20
    model = build_and_train_model(X_train, Y_train, X_val, Y_val, hidden, neurons, 500)
    models[name] = {'model': model, 'X_test': X_test, 'Y_test': Y_test}

# -------------------------------
# 4. Plot results
# -------------------------------
plt.figure(figsize=(18,5))

for i, (name, info) in enumerate(models.items(), 1):
    model = info['model']
    X_test = info['X_test']
    Y_test = info['Y_test']
    Y_pred = model.predict(X_test).flatten()  # flatten to 1D

    plt.subplot(1,3,i)
    plt.scatter(X_test, Y_test, color='blue', label='Original y')
    plt.scatter(X_test, Y_pred, color='red', label='Predicted y')
    plt.title(f'{name} Equation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

plt.tight_layout()
plt.show()
