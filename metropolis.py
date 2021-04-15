
from abc import ABC, abstractmethod

import numpy as np



class Metropolis(ABC):
    def __init__(self, itr=1000, cur=1):
        self._post = []
        self._iter = itr
        self._burn = int(itr/5)
        self._current = cur
        self._accepted = 0

    def sample(self):
        self._post = [self._current]
        for i in range(self._iter):
            proposed = self.proposal(self._current)
            #print(self._current, proposed)

            p = min(self.target(proposed)/self.target(self._current), 1)
            if np.random.random() < p:
                self._current = proposed
                if i >= self._burn:
                    self._accepted += 1
            self._post.append(self._current)

        return self._post[self._burn:]

    @abstractmethod
    def target(self, *args):
        raise NotImplementedError('Implement me bro!')

    @abstractmethod
    def proposal(self, *args):
        raise NotImplementedError('Implement me bro!')
