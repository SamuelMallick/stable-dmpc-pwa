import matplotlib.pyplot as plt


def plot_system(X, U):
    for i in range(3):
        plt.plot(X[0, 2 * i], X[0, 2 * i + 1], "o", color=f"C{i}")
        plt.plot(X[:, 2 * i], X[:, 2 * i + 1], color=f"C{i}")
        plt.plot(X[-1, 2 * i], X[-1, 2 * i + 1], "x", color=f"C{i}")
    plt.xlim(-20.1, 20.1)
    plt.ylim(-20.1, 20.1)
    plt.figure()
    plt.plot(U)
    plt.show()
