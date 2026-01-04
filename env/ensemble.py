import time

import gym
from gym import spaces
import numpy as np
from .optimizer import *
from .utils import *
from .Population import *
import warnings
import copy
import collections


class Ensemble(gym.Env):
    def __init__(self, optimizers, problem, periods, MaxFEs, sample_times, sample_size, seed=0, qr_len=5, record_period=-1, sample_FEs_type=2, terminal_error=1e-8):
        self.step_idx = 0
        self.dim = problem.dim
        self.MaxFEs = MaxFEs
        self.periods = periods
        self.optimizers = []
        self.max_step = len(periods)
        for optimizer in optimizers:
            if optimizer == 'random_optimizer':
                # Special initialization for random_optimizer
                self.optimizers.append(eval(optimizer)(self.dim, self.periods))
            else:
                # Standard initialization for others
                self.optimizers.append(eval(optimizer)(self.dim))
        self.sample_size = sample_size
        self.sample_times = sample_times
        self.best_history = [[] for _ in range(len(optimizers))]
        self.worst_history = [[] for _ in range(len(optimizers))]
        self.q_r_history = [np.zeros(len(self.optimizers) + 1)] * qr_len
        self.qr_len = qr_len
        self.optimzer_used = np.zeros(len(self.optimizers))

        self.problem = problem
        self.n_dim_obs = 6
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_dim_obs,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(self.optimizers))
        self.sample_FEs_type = sample_FEs_type
        self.baseline = False
        if record_period > 0:
            self.baseline = True
        self.record_period = record_period if record_period > 0 else MaxFEs // 100
        self.Fevs = np.array([])
        self.seed = seed
        self.final_obs = None
        self.terminal_error = terminal_error

    def local_sample(self):
        samples = []
        costs = []
        min_len = 1e9
        sample_size = self.sample_size if self.sample_size > 0 else self.population.NP
        for i in range(self.sample_times):
            sample, _, _ = self.optimizers[np.random.randint(len(self.optimizers))].step(copy.deepcopy(self.population),
                                                                                         self.problem,
                                                                                         self.FEs,
                                                                                         self.FEs + sample_size,
                                                                                         self.MaxFEs)
            samples.append(sample)
            cost = sample.cost
            costs.append(cost)
            min_len = min(min_len, cost.shape[0])
        if self.sample_FEs_type > 0:
            if self.FEs % self.record_period + sample_size * self.sample_times >= self.record_period and not self.done:
                # print(self.Fevs.shape[0])
                self.Fevs = np.append(self.Fevs, self.population.gbest)
                # print(self.Fevs.shape[0])
            self.FEs += sample_size * self.sample_times
            if self.FEs >= self.MaxFEs:
                self.done = True
        for i in range(self.sample_times):
            costs[i] = costs[i][:min_len]
        return np.array(samples), np.array(costs)

    # observed env state
    def observe(self):
        samples, sample_costs = self.local_sample()
        feature = self.population.get_feature(self.problem,
                                              sample_costs,
                                              self.cost_scale_factor,
                                              self.FEs / self.MaxFEs)

        best_move = np.zeros((len(self.optimizers), self.dim)).tolist()
        worst_move = np.zeros((len(self.optimizers), self.dim)).tolist()
        move = np.zeros((len(self.optimizers) * 2, self.dim)).tolist()
        for i in range(len(self.optimizers)):
            if len(self.best_history[i]) > 0:
                move[i*2] = np.mean(self.best_history[i], 0).tolist()
                move[i * 2 + 1] = np.mean(self.worst_history[i], 0).tolist()
                best_move[i] = np.mean(self.best_history[i], 0).tolist()
                worst_move[i] = np.mean(self.worst_history[i], 0).tolist()
        # move = list(np.concatenate((best_move, worst_move), 0))
        move.insert(0, feature)
        # move.append(np.array(self.q_r_history).reshape(-1))
        # move.append(np.array((self.optimzer_used / np.sum(self.optimzer_used)) if np.sum(self.optimzer_used) > 0 else np.zeros(len(self.optimizers))).reshape(-1))
        return move

    def seed(self, seed=None):
        np.random.seed(seed)

    # initialize env
    def reset(self):
        np.random.seed(self.seed)
        self.population = Population(self.dim)
        self.population.initialize_costs(self.problem)
        self.cost_scale_factor = self.population.gbest
        self.FEs = self.population.NP
        self.Fevs = np.array([])
        self.done = (self.population.gbest <= self.terminal_error or self.FEs >= self.MaxFEs)
        if not self.baseline:
            return self.observe()

    def step(self, action):
        warnings.filterwarnings("ignore")
        if not self.done:
            act = action['action']
            qvalue = action['qvalue']

            # 1. CHECK SCHEDULE COMPLETION
            if self.step_idx >= len(self.periods):
                self.done = True
                # Ensure we return a consistent shape even if called extra times
                info = self._get_info()
                return self.final_obs, 0, self.done, {'info': info}

            # 2. DETERMINE PERIOD LENGTH
            safe_idx = min(self.step_idx, len(self.periods) - 1)
            current_period_len = self.periods[safe_idx]

            # 3. SET TARGET FEs
            target_FEs = min(self.FEs + current_period_len, self.MaxFEs)

            optimizer = self.optimizers[act]

            # 4. RUN OPTIMIZER
            # We pass record_period, but we don't trust the returned list length strictly
            self.population, step_fevs, self.FEs = optimizer.step(
                self.population,
                self.problem,
                self.FEs,
                target_FEs,
                self.MaxFEs,
                record_period=current_period_len
            )

            # 5. STRICT HISTORY UPDATE
            # Regardless of whether step_fevs has 0, 1, or 5 values,
            # we append exactly ONE value: the current global best.
            current_best = self.population.gbest
            self.Fevs = np.append(self.Fevs, current_best)

            # RL Updates
            self.optimzer_used[act] += 1
            last_cost = self.population.gbest  # Note: logic uses cost before update usually, assuming previous step
            # reward = max((start_of_step_cost - current_best) / scale, 0)

            # Re-calculating reward based on improvement during this step
            # last_cost is defined at start of function, so we use it here.
            reward = max((last_cost - self.population.gbest) / self.cost_scale_factor, 0)

            self.q_r_history.append(np.concatenate((qvalue, [reward])))
            self.q_r_history = self.q_r_history[-self.qr_len:]

            # 6. ADVANCE
            self.step_idx += 1

            # 7. CHECK TERMINATION
            self.done = (self.step_idx >= len(self.periods)) or (self.FEs >= self.MaxFEs)

            # 8. OBSERVATION
            if self.baseline:
                observe = None
            else:
                observe = self.observe()
                self.final_obs = observe

            # 9. RETURN INFO
            return observe, reward, self.done, {'info': self._get_info()}
        else:
            return self.final_obs, -1, self.done, {'info': self._get_info()}

    def _get_info(self):
        """Helper to ensure info arrays are always the correct length (len(periods))."""
        # If we finished early (e.g. MaxFEs hit), pad the array to match n_periods
        # so broadcasting in Testing.py works.
        target_len = len(self.periods)
        current_fevs = self.Fevs.copy()

        if len(current_fevs) < target_len:
            # Pad with the last known value
            last_val = current_fevs[-1] if len(current_fevs) > 0 else 0.0
            padding = np.full(target_len - len(current_fevs), last_val)
            current_fevs = np.concatenate((current_fevs, padding))
        elif len(current_fevs) > target_len:
            # Trim (shouldn't happen with strict update, but for safety)
            current_fevs = current_fevs[:target_len]

        return Info(
            descent_seq=1 - current_fevs / self.cost_scale_factor,
            done=self.done,
            FEs=self.FEs,
            descent=1 - self.population.gbest / self.cost_scale_factor,
            best_cost=self.population.gbest,
        )


class random_optimizer:
    def __init__(self, dim, periods):
        self.dim = dim
        self.periods = periods
        # Initialize the underlying optimizers once
        self.jde = JDE21(self.dim)
        self.shade = NL_SHADE_RSP(self.dim)

    # an uniform interface for testing
    def test_run(self,
                 problem,  # the problem instance to be optimize
                 seed,  # the random seed for running to ensure fairness
                 MaxFEs  # the max number of evaluations
                 ):
        np.random.seed(seed)
        # initialize population and optimizers
        population = Population(self.dim)
        population.initialize_costs(problem)
        factor = population.gbest
        k = 0
        Fevs = np.array([])
        while population.NP >= int(np.power(self.dim, k / 5 - 3) * MaxFEs):
            Fevs = np.append(Fevs, np.min(population.cost[:int(np.power(self.dim, k / 5 - 3) * MaxFEs)]))
            k += 1
        jde = JDE21(self.dim)
        shade = NL_SHADE_RSP(self.dim)
        current_FEs = population.NP
        prob = 0.5
        for period_len in self.periods:
            # Stop if we have exceeded budget
            if current_FEs >= MaxFEs:
                break

            # Calculate the target end point for this specific step
            target_FEs = min(current_FEs + period_len, MaxFEs)

            # randomly select an optimizer
            if np.random.random() < prob:
                # We ignore the middle return value (step_fevs) for the baseline runner
                # and focus on updating the population and FEs count
                population, current_FEs = jde.step(population, problem, current_FEs, target_FEs, MaxFEs)
            else:
                population, current_FEs = shade.step(population, problem, current_FEs, target_FEs, MaxFEs)

            # Append the current global best to the history
            # This ensures we get exactly 1 point per period
            Fevs = np.append(Fevs, population.gbest)

            if population.gbest < 1e-8:
                break
        while Fevs.shape[0] < 16:
            Fevs = np.append(Fevs, 0.0)

        return 1 - Fevs[-1] / factor, current_FEs

    def step(self, population, problem, FEs, FEs_end, MaxFEs, record_period=None):
        """
        Executes one step (period) using a randomly selected optimizer.
        """
        # Randomly select between NL_SHADE_RSP (0) and JDE21 (1)
        if np.random.random() < 0.5:
            # We ignore the returned step_fevs (middle arg) to rely on the global logic
            population, _, FEs = self.jde.step(population, problem, FEs, FEs_end, MaxFEs)
        else:
            population, _, FEs = self.shade.step(population, problem, FEs, FEs_end, MaxFEs)

        # Return the exact format expected by Ensemble.step
        # Note: We return [gbest] as the 'step_fevs' to satisfy the strict recording logic
        return population, [population.gbest], FEs

