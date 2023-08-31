import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def main():
    max_temp = 39
    x = range(max_temp)  # C
    y = [round(1.8 * F, 1) + 32 for F in x]  # F
    # plt.plot(x, y, '-*r')
    # plt.show()
    x0 = np.array(x).reshape(-1, 1)
    y0 = np.array(y).reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(x0, y0, test_size=0.2)
    model = LinearRegression()
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    accuracy = round(accuracy * 100, 2)
    print('Accuracy: ', accuracy)


if __name__ == '__main__':
    main()
