import numpy as np

mathpi = 3.141592653589793


class testfunction(object):
    def __init__(self):
        self.xmin = None
        self.xmax = None
        self.dx = None

    def __call__(self, x):
        raise ValueError("Function not defined")

    def get_range(self):
        return np.array([self.xmin, self.xmax])

    def check_input(self, x):
        x = x.reshape((-1))
        if not x.shape[0] == self.dx or any(x > self.xmax) or any(x < self.xmin):
            raise ValueError("x is wrong dim or out of bounds")
        return x

    def RandomCall(self):
        x = np.random.uniform(self.xmin, self.xmax)
        f = self.__call__(x)
        # f = self.testmode(x)
        print("\n\nFunction: {}".format(type(self).__name__))
        print("Input = {}".format(x))
        print("Output = {}".format(f))

    def testmode(self, x):
        return self.__call__(x, noise_std=0)


class Styblinski(testfunction):
    def __init__(self, dim=10):
        self.dx = dim
        self.xmin = -5.0 * np.ones(dim)
        self.xmax = 5.0 * np.ones(dim)

    def __call__(self, x, noise_std=1):
        x = self.check_input(x)
        output = 0.5 * (np.sum(x ** 4) - 16.0 * np.sum(x ** 2) + 5.0 * np.sum(x))
        return output + np.random.normal(size=output.shape) * noise_std


class Hartmann6(testfunction):
    def __init__(self):
        self.dx = 6
        self.xmin = np.zeros(6)
        self.xmax = np.ones(6)
        self.A = np.asarray(
            [
                [10.0, 3.00, 17.0, 3.50, 1.70, 8.00],
                [0.05, 10.0, 17.0, 0.10, 8.00, 14.0],
                [3.00, 3.50, 1.70, 10.0, 17.0, 8.00],
                [17.0, 8.00, 0.05, 10.0, 0.10, 14.0],
            ]
        )
        self.P = np.asarray(
            [
                [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
            ]
        )
        self.alpha = np.asarray([1.0, 1.2, 3.0, 3.2])

    def __call__(self, x, noise_std=1):
        x = self.check_input(x)
        x = np.tile(x, (4, 1))
        output = self.A * ((x - self.P) ** 2)
        output = np.exp(np.sum(output, axis=1))
        output = np.sum(self.alpha * output)
        return output + np.random.normal(size=output.shape) * noise_std


class CamelBack(testfunction):
    def __init__(self):
        self.dx = 2
        self.xmin = np.asarray([-3, -2])
        self.xmax = np.asarray([3, 2])

    def __call__(self, x, noise_std=1):
        x = self.check_input(x)
        term1 = (4 - 2.1 * x[0] ** 2 - (x[0] ** 4) / 3) * x[0] ** 2
        term2 = x[0] * x[1] + (x[1] ** 2) * (4 * x[1] ** 2 - 4)
        return term1 + term2 + np.random.normal(size=term1.shape) * noise_std


class Levy(testfunction):
    def __init__(self, dim=10):
        self.dx = dim
        self.xmin = -10 * np.ones(dim)
        self.xmax = 10 * np.ones(dim)

    def __call__(self, x, noise_std=1):
        x = self.check_input(x)
        w = 1 + 0.25 * (x - 1)
        wi = w[: (self.dx - 1)]
        wd = w[self.dx - 1]
        term1 = np.sin(mathpi * w[0]) ** 2
        term2 = ((wi - 1) ** 2) * (1 + 10 * (np.sin(mathpi * wi + 1) ** 2))
        term3 = ((wd - 1) ** 2) * (1 + np.sin(2 * mathpi * wd) ** 2)
        return term1 + np.sum(term2) + term3 + np.random.normal(size=term1.shape) * noise_std


def main():

    for funclass in [Styblinski, Hartmann6, CamelBack, Levy]:
        f = funclass()
        f.RandomCall()


if __name__ == "__main__":
    main()
