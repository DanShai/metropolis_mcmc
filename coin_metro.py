import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from metropolis import Metropolis


class Coin(Metropolis):
    def __init__(self, itr=500, cur=.1):
        super().__init__(itr=itr, cur=cur)
        self._sigma = 0.3
        self._n = 100
        self._a = 10
        self._b = 8
        self._heads = 65

    def proposal(self, tetha):
        return tetha + stats.norm(0, self._sigma).rvs()

    def target(self, tetha):
        if tetha < 0 or tetha > 1:
            return 0
        else:
            return self.likelihood(tetha).pmf(self._heads)*self.prior().pdf(tetha)

    def likelihood(self, tetha):
        return stats.binom(self._n, tetha)

    def prior(self):
        return stats.beta(self._a, self._b)

    def simulate(self):
        thetas = np.linspace(0, 1, 200)
        samples = self.sample()
        print(samples[:10])
        print("Efficiency = ", self._accepted)
        post = stats.beta(self._heads + self._a, self._n-self._heads + self._b)
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Stats')

        ax1.hist(samples, 40, histtype='step', density=True,
                 linewidth=1, label='Calculated')
        ax1.hist(self.prior().rvs(len(samples)), 40, histtype='step',
                 density=True, linewidth=1, label='Prior')
        ax1.plot(thetas, post.pdf(thetas), c='green',
                 linestyle='--', alpha=0.5, label='True posterior')
        ax1.set_xlim(0, 1)
        ax1.legend(loc='upper left')

        ax2.plot(samples, '-o', label='samples')
        ax2.set_title('samples convergence')
        ax2.set_xlim(0, self._iter)
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper left')

        plt.show()


if __name__ == "__main__":
    coin = Coin()
    coin.simulate()
