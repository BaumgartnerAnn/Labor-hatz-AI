from typing import Optional
from evotorch import Problem, Solution
from evotorch.algorithms.searchalgorithm import (
    SearchAlgorithm,
    SinglePopulationAlgorithmMixin,
)
from Net import Net
from evotorch.neuroevolution import NEProblem
import torch
import itertools
from copy import deepcopy

class SimpleGA(SearchAlgorithm, SinglePopulationAlgorithmMixin):
    def __init__(
        self,
        problem: Problem,
        *,
        popsize: int,  # Total population size n
        num_elites: int,  # Number of elites that survive each generation e
        num_parents: int,  # Number of parents from which to generate children
        mutation_power: float,  # Scale of gaussian noise used to generate children
        nn_values: list # list of neural networks for the population
    ):  # sourcery skip: remove-unnecessary-cast

        # Call the __init__(...) method of the superclass
        SearchAlgorithm.__init__(
            self,
            # Problem to work on:
            problem,
            # The remaining keyword arguments are for registering
            # the status getter methods.
            # The result of these status getter methods will
            # automatically be shown in the status dictionary.
            pop_best=self._get_pop_best,
            pop_best_eval=self._get_pop_best_eval,
        )
        SinglePopulationAlgorithmMixin.__init__(
            self,
        )

        # Store the hyperparameters
        self._popsize = int(popsize)
        self._num_elites = int(num_elites)
        self._num_parents = int(num_parents)
        self._mutation_power = float(mutation_power)

        # Generate the initial population -- note that this uses the problem's initial bounds as a uniform hyper-cube.
        self._population = self._problem.generate_batch(self._popsize)
        self._population.set_values(nn_values)

        # The following variable stores a copy of the current population's
        # best solution
        self._pop_best: Optional[Solution] = None
    def _get_pop_best(self):
        return self._pop_best

    def _get_pop_best_eval(self):
        return self._pop_best.get_evals()

    def _step(self):
        # the population is already evaluated, the evalutation are set via the problem.set_evals() method
        # Sort the population
        self._population = self._population[self._population.argsort()]

        # Select the parents.
        parents = self._population[: self._num_parents]

        # Pick a random parent for each child
        num_children = self._popsize - self._num_elites
        parent_indices = self.problem.make_randint(num_children, n=self._num_parents)
        parent_values = parents.values[parent_indices]

        # Add gaussian noise
        child_values = (
            parent_values
            + self._mutation_power
            * self.problem.make_gaussian(num_children, self.problem.solution_length)
        )

        # Overwrite all the non-elite solutions with the new generation
        self._population.access_values()[self._num_elites :] = child_values

    @property
    def population(self):
        return self._population