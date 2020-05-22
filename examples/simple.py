from leap.simple import ea_solve

def function(values):
    return sum([x ** 2 for x in values])

ea_solve(function,
         bounds=[(-5.12, 5.12) for _ in range(5)],
         mutation_std=0.1)
