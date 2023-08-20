import math

from sklearn.datasets import load_diabetes
import numpy as np
import matplotlib.pyplot as plt


def gradientDescent(xerr1, yerr1, data_matrix, target):
    x1 = np.random.rand(10)

    epsilon1 = 0.01
    res_vec1 = target
    grad1 = data_matrix.T @ data_matrix @ x1 - data_matrix.T @ res_vec1
    delta1 = 0.05
    # yerr1 = []
    # xerr1 = []
    i = 0
    r1 = data_matrix @ x1 - res_vec1
    while (2 * grad1.T @ grad1) > delta1:
        yerr1.append(r1.T @ r1)
        xerr1.append(i)
        x1 = x1 - (2 * epsilon1 * grad1)
        grad1 = data_matrix.T @ data_matrix @ x1 - data_matrix.T @ res_vec1
        r1 = data_matrix @ x1 - res_vec1
        i += 1

    return x1


def gradient_descent(data, target, epsilon=0.1, delta=0.05):
    x_k = np.random.rand(10)
    grad1 = data.T @ data @ x_k - data.T @ target
    xerr = []
    yerr = []

    i = 0
    r_train = data @ x_k - target
    while (2 * grad1.T @ grad1) > delta:
        xerr.append(i)
        yerr.append((r_train.T @ r_train) / len(target))
        x_k = x_k - (2 * epsilon * grad1)
        grad1 = data.T @ data @ x_k - data.T @ target
        r_train = data @ x_k - target
        i += 1

    return x_k, xerr, yerr


def double_gradient_descent(train_data, train_target, test_data, test_target, epsilon=0.1, delta=0.08):
    """
    preforms a gradient descent over the train data, returns the error for both test and train data
    :param train_data: train data set
    :param train_target: target column for train set
    :param test_data: test data set
    :param test_target: target column for test set
    :param epsilon: step size
    :param delta: limit for stopping
    :return: the final x_k, number of steps, the train err and the test err plots
    """
    x_k = np.random.rand(10)
    grad1 = train_data.T @ train_data @ x_k - train_data.T @ train_target
    xerr = []
    train_err_y = []
    test_err_y = []

    i = 0
    r_train = train_data @ x_k - train_target
    r_test = test_data @ x_k - test_target
    while (2 * grad1.T @ grad1) > delta:
        xerr.append(i)
        train_err_y.append((r_train.T @ r_train) / len(train_target))
        test_err_y.append((r_test.T @ r_test) / len(test_target))
        x_k = x_k - (2 * epsilon * grad1)
        grad1 = train_data.T @ train_data @ x_k - train_data.T @ train_target
        r_train = train_data @ x_k - train_target
        r_test = test_data @ x_k - test_target
        i += 1

    return x_k, xerr, train_err_y, test_err_y


def Q1():
    # load the diabetes dataset
    diabetes = load_diabetes()

    # gradient descent on all
    diabetes_matrix = np.array(diabetes.data)
    res_vec = np.array(diabetes.target)
    x_res, xerr, yerr = gradient_descent(diabetes_matrix, res_vec)

    plt.plot(xerr, yerr)
    # add labels to the x and y axes
    plt.xlabel('iteration')
    plt.ylabel('err')
    # add a title to the plot
    plt.title('Example Plot')
    # display the plot
    plt.show()


def Q2():
    # splitting data:
    # load the diabetes dataset
    diabetes = load_diabetes()

    # shuffle the columns of the dataset
    num_cols = diabetes.data.shape[0]
    col_indices = np.random.permutation(num_cols)
    shuffled_data = diabetes.data[col_indices]
    shuffled_target = np.array(diabetes.target)[col_indices]

    # select the first 354 columns of the shuffled dataset
    train_data = shuffled_data[:354]
    train_target = shuffled_target[:354]

    test_data = shuffled_data[354:]
    test_target = shuffled_target[354:]

    xres, xerr1, train_err_y, test_err_y = double_gradient_descent(train_data, train_target, test_data, test_target)

    plt.plot(xerr1, train_err_y, label="train error")
    plt.plot(xerr1, test_err_y, label="test error")
    plt.legend()
    # # add labels to the x and y axes
    plt.xlabel('iteration')
    plt.ylabel('err')
    # # add a title to the plot
    plt.title('task 2: train vs test error')
    # # display the plot
    plt.show()

def Q3():
    diabetes = load_diabetes()
    train_size = 4 * len(diabetes.data) // 5
    train_errs = []
    test_errs = []
    for i in range(10):
        # shuffle the columns of the dataset
        num_cols = diabetes.data.shape[0]
        col_indices = np.random.permutation(num_cols)
        shuffled_data = diabetes.data[col_indices]
        shuffled_target = np.array(diabetes.target)[col_indices]

        # select the first 354 columns of the shuffled dataset
        train_data = shuffled_data[:train_size]
        train_target = shuffled_target[:train_size]

        test_data = shuffled_data[train_size:]
        test_target = shuffled_target[train_size:]
        xres, xerr1, train_err_y, test_err_y = double_gradient_descent(train_data, train_target, test_data, test_target)
        train_err = train_data @ xres - train_target
        test_err = test_data @ xres - test_target
        train_errs.append(train_err.T @ train_err / len(train_data))
        test_errs.append(test_err.T @ test_err / len(test_data))
    train_avg = np.average(train_errs)
    test_avg = np.average(test_errs)

    X = [str(i) for i in range(10)] + ["Average"]
    X_axis = np.arange(len(X))
    bar_width = 0.4
    plt.bar(X_axis - 0.2, train_errs + [train_avg], bar_width, label='train')
    plt.bar(X_axis + 0.2, test_errs + [test_avg], bar_width, label='test')

    plt.xticks(X_axis, X)
    plt.legend()
    plt.show()


Q1()
