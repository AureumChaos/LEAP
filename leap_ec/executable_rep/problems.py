import numpy as np

from leap_ec.real_rep.problems import ScalarProblem

##############################
# Class EnvironmentProblem
##############################
class EnvironmentProblem(ScalarProblem):
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
        self.environment._max_episode_steps = steps  # This may not work with all environments.
        self.stop_on_done = stop_on_done
        self.gui = gui
        if fitness_type == 'reward':
            self.fitness = EnvironmentProblem._reward_fitness
        elif fitness_type == 'survival':
            self.fitness = EnvironmentProblem._survival_fitness
        else:
            raise ValueError(f"Unrecognized fitness type: '{fitness_type}'")

    @property
    def num_inputs(self):
        """Return the number of dimensions in the environment's input space."""
        self.space_dimensions(self.environment.observation_space)

    @property
    def num_outputs(self):
        """Return the number of dimensions in the environment's action space."""
        self.space_dimensions(self.environment.action_space)

    @classmethod
    def _reward_fitness(cls, observations, rewards):
        """Compute fitness by summing the rewards across all runs."""
        sums = [sum(run) for run in rewards]
        return np.mean(sums)

    @classmethod
    def _survival_fitness(cls, observations, rewards):
        """Compute fitness as the average length of the runs."""
        return np.mean([len(o) for o in observations])

    @staticmethod
    def space_dimensions(observation_space) -> int:
        """Helper to get the number of dimensions (variables) in an OpenAI Gym space.
        
        The point of this helper is that it works on simple spaces:

        >>> from gym import spaces
        >>> discrete = spaces.Discrete(8)
        >>> EnvironmentProblem.space_dimensions(discrete)
        1

        Box spaces:

        >>> box = spaces.Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        >>> EnvironmentProblem.space_dimensions(box)
        12

        And Tuple spaces:

        >>> tup = spaces.Tuple([discrete, box])
        >>> EnvironmentProblem.space_dimensions(tup)
        13
        """
        if hasattr(observation_space, 'spaces'):
            # If we're a Tuple space, count the inputs across each space in the Tuple
            return sum([ int(np.prod(s.shape)) for s in observation_space.spaces ])
        else:
            # Otherwise just look at the shape of the space directly
            return int(np.prod(observation_space.shape))

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
                action = executable(observation)
                observation, reward, done, info = self.environment.step(action)
                run_observations.append(observation)
                run_rewards.append(reward)
                if self.stop_on_done and done:
                    break
            observations.append(run_observations)
            rewards.append(run_rewards)
        return self.fitness(observations, rewards)


##############################
# Class TruthTableProblem
##############################
class TruthTableProblem(ScalarProblem):
    """Defines a fitness function over a :class:`~leap_ec.executable.phenotype.Executable` by 
    evaluating it against each row of a given Boolean function's truth table.

    Both the executable we receive and the `boolean_function` we compare against should return 
    a list of 1 or more outputs.
    """

    def __init__(self, boolean_function, num_inputs, num_outputs, pad_inputs=False, maximize=True):
        super().__init__(maximize)
        assert(boolean_function is not None)
        assert(callable(boolean_function))
        assert(num_inputs > 0)
        assert(num_outputs > 0)
        self.function = boolean_function
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.pad_inputs = pad_inputs

    def evaluate(self, executable):
        """
        Say our object function is $(x_0 \wedge x_1) \vee x_3$:

        >>> problem = TruthTableProblem(lambda x: [ (x[0] and x[1]) or x[2] ], num_inputs=3, num_outputs=1)

        The truth table for this Boolean function has eight entries:

        F F F=F
        F F T=T
        F T F=F
        F T T=T
        T F F=F
        T F T=T
        T T F=T
        T T T=T

        Now consider a different function, $(x_0 \wedge x_1) \oplus x_3$.
        
        >>> executable = lambda x: [ (x[0] and x[1]) ^ x[2] ]

        This function's truth table differs from the first one by exactly one
        entry (in the second one, TTT=F).  So we expect a fitness value of 
        $7/8 = 0.875$:
        
        >>> problem.evaluate(executable)
        0.875

        Note that we our lambda functions above return a list that contains a 
        computed value, rather than just the value directly.  This is because
        this framework allows us to work with functions of more than one output:

        >>> problem = TruthTableProblem(lambda x: [ x[0] and x[1], x[0] or x[1] ], num_inputs=3, num_outputs=2)
        >>> problem.evaluate(lambda x: [ x[0] and x[1], x[0] or x[1] ])
        1.0

        """
        assert(executable is not None)
        assert(callable(executable))
        input_samples = self._enumerate_tt(self.num_inputs)

        score = 0
        for input_ in input_samples:
            expected = self.function(input_)
            assert(hasattr(expected, '__len__')), "The function given to a TruthTableProblem must return a list of outputs with length 1 or greater."
            assert(len(expected) > 0), f"The function given to TruthTableProblem must return a list of outputs with length 1 or greater, but its length was {len(expected)}."
            if self.pad_inputs and (len(input_) < executable.num_inputs):
                    input_ += [ 0 for _ in range(executable.num_inputs - len(input_)) ]
            observed = executable(input_)
            if observed == expected:
                score += 1

        return score/len(input_samples)

    @staticmethod
    def _enumerate_tt(num_inputs):
        """Generate input permutations for a complex truth table."""
        # TODO Rewriting this as a generator function would save a lot of memory on big tables.
        assert(num_inputs > 0)
        if num_inputs == 1:
            return [[0], [1]]
        else:
            tt_minus_1 = TruthTableProblem._enumerate_tt(num_inputs - 1)
            zeros = [ [0] + row for row in tt_minus_1 ]
            ones = [ [1] + row for row in tt_minus_1 ]
            return zeros + ones

