import numpy as np

from leap_ec.real_rep.problems import ScalarProblem

##############################
# Class ExecutableProblem
##############################
class ExecutableProblem(ScalarProblem):
    """Defines a fitness function over :class:`~leap_ec.executable.phenotype.Executable` by 
    evaluating them within a given environment.
    
    :param int runs: The number of independent runs to aggregate data over.
    :param int steps: The number of steps to run the simulation for within each run.
    :param environment: A simulation environment corresponding to the OpenAI Gym environment interface.
    :param behavior_fitness: A function 
    """

    def __init__(self, runs: int, steps: int, environment, fitness_type: str,
                 gui: bool, stop_on_done=True, maximize=True):
        assert(runs > 0)
        assert(steps > 0)
        assert(environment is not None)
        assert(fitness_type is not None)
        super().__init__(maximize)
        self.runs = runs
        self.steps = steps
        self.environment = environment
        self.environment._max_episode_steps = steps
        self.stop_on_done = stop_on_done
        self.gui = gui
        if fitness_type == 'reward':
            self.fitness = ExecutableProblem._reward_fitness
        elif fitness_type == 'survival':
            self.fitness = ExecutableProblem._survival_fitness
        else:
            raise ValueError(f"Unrecognized fitness type: '{fitness_type}'")

    @classmethod
    def _reward_fitness(cls, observations, rewards):
        """Compute fitness by summing the rewards across all runs."""
        sums = [sum(run) for run in rewards]
        return np.mean(sums)

    @classmethod
    def _survival_fitness(cls, observations, rewards):
        """Compute fitness as the average length of the runs."""
        return np.mean([len(o) for o in observations])

    def evaluate(self, executable):
        """Run the environmental simulation using `executable` as a controller,
        and use the resulting observations & rewards to compute a fitness value."""
        observations = []
        rewards = []
        for r in range(self.runs):
            observation = self.environment.reset()
            run_observations = [observation]
            run_rewards = []
            for t in range(self.steps):
                if self.gui:
                    self.environment.render()
                action = executable.output(observation)
                observation, reward, done, info = self.environment.step(action)
                run_observations.append(observation)
                run_rewards.append(reward)
                if self.stop_on_done and done:
                    break
            observations.append(run_observations)
            rewards.append(run_rewards)
        return self.fitness(observations, rewards)