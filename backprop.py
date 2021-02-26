import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat
import sys

np.set_printoptions(threshold=sys.maxsize)


def nn_hypothesis_CostFunction(X, Y, theta1, theta2, m, Lambda, classes):

    a_2 = propagate_forward(X, theta1)
    a_2 = np.column_stack((np.ones(m), a_2))

    hypothesis = propagate_forward(a_2, theta2)

    cost_function = 0

    # My cost function, seems better
    cost_function = (-1/m)*np.sum((Y * np.log(hypothesis)) +
                                  ((1-Y) * np.log(1-hypothesis))) + (Lambda/(2*m))*(np.sum((theta2**2)) + np.sum((theta1**2)))

    return hypothesis, cost_function


def propagate_forward(a_in, theta):
    a_out = 1/(1+np.exp(-1*(np.dot(a_in, theta.T))))

    return a_out


def back_prop(X, Y, theta1, theta2, m, Lambda):
    # The two lines below should be in the gradient descent function
    D1 = np.zeros(theta1.shape)
    delta1 = np.zeros(theta1.shape)
    D2 = np.zeros(theta2.shape)
    delta2 = np.zeros(theta2.shape)

    a_2 = propagate_forward(X, theta1)
    a_2 = np.column_stack((np.ones(m), a_2))

    hypothesis = propagate_forward(a_2, theta2)

    small_delta3 = hypothesis - Y
    small_delta2 = np.zeros((m, a_2.shape[1]))

    sigmoid_derivative2 = a_2*(1-a_2)
    theta_dot_derivative = np.zeros((m,  a_2.shape[1], classes))

    for i in range(0, m):
        theta_dot_derivative[i] = theta2.T * \
            sigmoid_derivative2[i][:, np.newaxis]
        small_delta2[i] = small_delta3[i] @ theta_dot_derivative[i].T

        delta1 += np.dot(small_delta2[i]
                         [1:, np.newaxis], X[i][:, np.newaxis].T)
        delta2 += np.dot(small_delta3[i]
                         [:, np.newaxis], a_2[i][:, np.newaxis].T)

    D1[:, 0] = (1/m)*delta1[:, 0]
    D1[:, 1:] = (1/m)*(delta1[:, 1:] + Lambda*theta1[:, 1:])

    D2[:, 0] = (1/m)*delta2[:, 0]
    D2[:, 1:] = (1/m)*(delta2[:, 1:] + Lambda*theta2[:, 1:])

    return D1, D2


def GradientDescent(X, Y, theta1, theta2, Lambda, classes, m, alpha, iterations):
    for i in range(0, iterations):
        hypo, cost_function = nn_hypothesis_CostFunction(
            X, Y, theta1, theta2, m, Lambda, classes)

        print("Iterations: {} | Cost: {}".format(i+1, cost_function))

        D1, D2 = back_prop(X, Y, theta1, theta2, m, Lambda)

        theta1 = theta1 - alpha*D1
        theta2 = theta2 - alpha*D2

    return theta1, theta2


theta_data = loadmat("ex4weights.mat")
theta1, theta2 = theta_data['Theta1'], theta_data['Theta2']

data = loadmat("ex3data1.mat")
X, y = data['X'], data['y']

classes = 10

# Changing the y value of the number "0" to 0 from 10
y[y == 10] = 0

m = np.size(y)
y = y.reshape(m,)
y = y.astype('float64')

# Display 100 random images from the dataset
rand_indices = np.random.choice(m, 100, replace=False)

X_display = X.copy()
X = np.column_stack((np.ones(m), X))
m, n = np.shape(X)

Y = np.zeros((m, classes))

for i in range(0, m):
    Y[i][int(y[i])] = 1

theta2 = np.roll(theta2, 1, axis=0)

epsilon_init = 0.12
my_theta1 = np.random.uniform(
    low=-epsilon_init, high=epsilon_init, size=theta1.shape)
my_theta2 = np.random.uniform(-epsilon_init, epsilon_init, theta2.shape)

Lambda = 0
alpha = 1
iterations = 400

my_theta1, my_theta2 = GradientDescent(
    X, Y, my_theta1, my_theta2, Lambda, classes, m, alpha, iterations)

max_indices = [None]*9
for i in range(0, 100):
    a2 = np.zeros(25)
    a2 = propagate_forward(X[rand_indices[i]], my_theta1)
    a2 = np.hstack((np.ones(1), a2))
    h = propagate_forward(a2, my_theta2)
    index_max = np.argmax(h)
    if(i<9):
        max_indices[i] = index_max
    print("Random index: {} | Predicted number: {} | Actual number: {}".format(
        rand_indices[i], index_max, int(y[rand_indices[i]])))
    print("Probabilities:")
    for j in range(0, classes):
        print("{} : {:.2f}%".format(j, h[j]*100))

fig = plt.figure(figsize=(8, 8))
rows = 3
columns = 3
axis = [None]*9
for i in range(1, rows*columns+1):
    axis[i-1] = fig.add_subplot(rows, columns, i)
    axis[i-1].set_title("Predicted number: {}".format(max_indices[i-1]), fontsize=10)
    plt.imshow(X_display[rand_indices[i-1]].reshape(20, 20).T, cmap='gray')
fig.tight_layout()
plt.show()
