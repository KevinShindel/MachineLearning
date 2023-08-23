import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def main():
    MAX_TEMP = 39
    x = range(MAX_TEMP)  # C
    y = [round(1.8 * F, 1) + 32 for F in x]  # F
    # plt.plot(x, y, '-*r')
    # plt.show()
    x0 = np.array(x).reshape(-1, 1)
    y0 = np.array(y).reshape(-1, 1)
    xTrain, xTest, yTrain, yTest = train_test_split(x0, y0, test_size=0.2)
    model = LinearRegression()
    model.fit(xTrain, yTrain)
    accuracy = model.score(xTest, yTest)
    accuracy = round(accuracy * 100, 2)
    print('Accuracy: ', accuracy)


if __name__ == '__main__':
    main()
