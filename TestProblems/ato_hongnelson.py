# -*- coding: utf-8 -*-
"""testfun3_ATO_HongNelson.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1z3Gpc0vMFJgjfJx2C-QWSm12Od4SVApA

Assemble-to-Order Simulator

code ported from

[SimOpt Library (matlab)](http://simopt.org/wiki/index.php?title=Assemble_to_order)

Requires:
 - numpy
"""

# @title Source code, make the function
import numpy as np


class ATO_HongNelson:
    def __init__(self, simreps=5):

        self.seed = 1

        self.NumComponentType = 8  # numebr of products
        self.NumCustomerType = 5  # number of customer types

        self.xmin = np.zeros([self.NumComponentType])
        self.xmax = 20 * np.ones([self.NumComponentType])

        self.simreps = simreps

        self.ProTimeMean = np.asarray([0.15, 0.4, 0.25, 0.15, 0.25, 0.08, 0.13, 0.4])

        self.ProTimeStd = 0.15 * self.ProTimeMean
        self.Profit = np.linspace(1, self.NumComponentType, self.NumComponentType)
        self.HoldingCost = 2 * np.ones([self.NumComponentType])

        self.ArrivalRate = 12.0  # assume Possion arrival
        self.iArrivalRate = 1.0 / self.ArrivalRate

        self.WarmUp = 20
        self.TotalTime = 70

        self.CustomerProb = np.asarray([0.3, 0.25, 0.2, 0.15, 0.1])  # prob. of each customer

        self.KeyComponent = np.asarray(
            [
                [1, 0, 0, 1, 0, 1, 0, 0],
                [1, 0, 0, 0, 1, 1, 0, 0],
                [0, 1, 0, 1, 0, 1, 0, 0],
                [0, 0, 1, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 1, 1, 0, 0],
            ]
        )

        self.NonKeyComponent = np.asarray(
            [
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0],
            ]
        )

    def get_range(self):
        return np.array([self.xmin, self.xmax])

    def Simulate(self, BaseStockLevel=np.ones([8]), seed=None):

        if not np.all(np.abs(np.round(BaseStockLevel) - BaseStockLevel) < 1e-5):
            raise ValueError("ATO inputs must be whole numbers!")

        self.seed += 1

        seed = seed if seed else self.seed

        fnSum = 0.0  # profit added up over simulation reps.
        np.random.seed(seed)

        SeriesLength = int(self.TotalTime * self.ArrivalRate * self.simreps * 4)

        Customer_series = np.random.choice(self.NumCustomerType, size=SeriesLength, p=self.CustomerProb)

        t = 1  # index for ticking through the customer series

        for k in range(self.simreps):  # do the simulation reps!

            EventTime = 1e5 * np.ones([1 + self.NumComponentType])
            EventTime[0] = -np.log(np.random.uniform()) * self.iArrivalRate
            TotalProfit = 0.0
            TotalCost = 0.0
            Inventory = np.ceil(BaseStockLevel)

            Clock = 0

            while Clock < self.TotalTime:

                OldInventory = Inventory
                OldClock = Clock
                Clock = np.min(EventTime)
                event = np.argmin(EventTime)

                if event == 0:
                    # a customer has come! Reset clock
                    EventTime[0] = Clock - np.log(np.random.uniform()) * self.iArrivalRate

                    # Get new customer
                    CustomerType = Customer_series[t]
                    t = t + 1

                    KeyOrder = self.KeyComponent[CustomerType,]
                    NonKeyOrder = self.NonKeyComponent[CustomerType,]

                    Sell = np.all(Inventory >= KeyOrder)

                    if Sell:
                        NonKeyOrder = Inventory * (Inventory <= NonKeyOrder) + NonKeyOrder * (Inventory > NonKeyOrder)

                        Inventory = Inventory - KeyOrder - NonKeyOrder

                        if Clock > self.WarmUp:
                            TotalProfit += np.sum(self.Profit * (KeyOrder + NonKeyOrder))

                        for i in range(self.NumComponentType):
                            if (Inventory[i] < BaseStockLevel[i]) & (EventTime[i + 1] > 1e4):
                                NewArrivalTime = self.ProTimeMean[i] + np.random.normal(1) * self.ProTimeStd[i]
                                EventTime[i + 1] = Clock + np.max([0, NewArrivalTime])

                else:
                    # stock has arrived!
                    ComponentType = event - 1
                    Inventory[ComponentType] = Inventory[ComponentType] + 1

                    if Inventory[ComponentType] >= BaseStockLevel[ComponentType]:
                        EventTime[event] = 1e5
                        if Clock > self.WarmUp:
                            TotalCost += (Clock - OldClock) * np.sum(OldInventory * self.HoldingCost)

            # end of this simulation rep, save profit
            fnSum += (TotalProfit - TotalCost) / (self.TotalTime - self.WarmUp)

        # end of all simuation reps
        fnAvg = fnSum / self.simreps

        return fnAvg

    def __call__(self, x, seed=None):
        return self.Simulate(x, seed)

    def testmode(self, x, num_seeds=100):
        testseeds = np.linspace(1000000, 1000000 + num_seeds, num_seeds).astype(int)
        test_f = np.array([self.__call__(x, s) for s in testseeds])

        return test_f.mean()


# @title Instantiate and Call the function with random input
def main():
    f = ATO_HongNelson()
    x_range = f.get_range()

    x = np.round(np.random.uniform(f.xmin, f.xmax))
    print("Input = {}".format(x))
    print("Output = {}".format(f.testmode(x)))


if __name__ == "__main__":
    main()
