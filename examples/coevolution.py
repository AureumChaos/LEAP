import numpy as np
from PIL import Image, ImageOps

from leap import core, ops
from leap.problem import ScalarProblem
from leap.algorithm import multi_population_ea

class ImageProblem(ScalarProblem):
    def __init__(self, path, maximize=True, size=(100, 100)):
        super().__init__(maximize)
        self.size = size
        self.img = ImageProblem._process_image(path, size)
        self.flat_img = np.ndarray.flatten(np.array(self.img))
        
    @staticmethod
    def _process_image(path, size):
        """Load an image and convert it to black-and-white."""
        x = Image.open(path)
        x = ImageOps.fit(x, size)
        return x.convert('1')
    
    def evaluate(self, phenome):
        assert(len(phenome) == len(self.flat_img)), f"Bad genome length: got {len(phenome)}, expected {len(self.flat_img)}"
        diff = np.logical_not(phenome ^ self.flat_img)
        return sum(diff)


if __name__ == '__main__':
    problem = ImageProblem('./centipede.jpg', size=(10, 10))
    pop_size=5

    @ops.listlist_op
    def const_evaluator(population):
        for ind in population:
            ind.fitness = -100
            
        return population

    ea = multi_population_ea(generations=100, pop_size=pop_size, num_populations=100,
                            evaluate=const_evaluator,
                            individual_cls=core.Individual,
                            decoder=core.IdentityDecoder(),
                            problem=problem,
                            initialize=core.create_binary_sequence(length=1),
                            shared_pipeline=[ops.tournament(k=10),
                                            ops.clone,
                                            ops.mutate_bitflip(expected=1),
                                            ops.coop_evaluate(context=core.context,
                                                              num_evaluators=1,
                                                              evaluator_selector=ops.random_selection),
                                            ops.pool(size=pop_size)])

    for g, x in ea:
        print(f"{g}, {[ind.fitness for ind in x]}")