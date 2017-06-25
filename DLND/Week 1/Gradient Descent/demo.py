from numpy import *

# this guy compares the actual value and predicted value
# and reports sum of squared error
def compute_error_for_given_points(b, m, points):
    totalError = 0

    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        totalError += (y - (m * x + b)) ** 2
    return totalError/ float(len(points))

# this guy calculates step wise gradient (direction) and updates the
#current slope and intercept. This new value of slope and intercept
# is then fed to gradient_descent_runner.
def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i,0]
        y = points[i,1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - (m_current * x + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]

# this runs calls step_gradient function iteratively again and again
# to achieve local minima.

# takes m, b = 0 feeds in step_gradient to get current value of m, b.
# Feeds the current value of m, b again to step_gradient again and inside
# step_gradient the m, b are again updated to give best result.
def gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations):
    b = initial_b
    m = initial_m

    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def run():
    points = genfromtxt("data.csv", delimiter=",")
    # hyperparameters
    learning_rate = 0.0001
    # y = mx + b
    initial_m = 0
    initial_b = 0
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_given_points(initial_b, initial_m, points)))
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_given_points(b, m, points)))

if __name__ == '__main__':
    run()
