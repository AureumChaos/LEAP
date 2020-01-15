Roadmap
=======

The LEAP development roadmap is as follows:

1) pre-Minimally Viable Product -- released 1/14/2020 as ``0.1-pre``

- basic support for binary representations
    - operators
        - bit flip mutation
        - point-wise crossover
        - uniform crossover
- basic support for real-valued representations
    - operators
        - mutate gaussian
- selection operators
    - truncation selection
    - tournament selection
    - random selection
    - deterministic cyclic selection
    - insertion selection
- continuous integration via Travis
- common test functions
    - binary
        - MAXONES
    - real-valued, optionally translated, rotated, and scaled
        - Ackley
        - Cosine
        - Griewank
        - Langermann
        - Lunacek
        - Noisy Quartic
        - Rastrigin
        - Rosenbock
        - Schwefel
        - Shekel
        - Spheroid
        - Step
        - Weierstrass
- test harnesses
    - ``pytest`` supported
- simple usage examples
    - canonical EAs, such as genetic algorithms (GA), evolutionary programming (EP), and evolutionary strategies (ES)
    - simple island model
    - basic run-time visualizations
    - use with Jupyter notebooks
- documentation outline/stubs for ReadTheDocs

2) Minimally Viable Product -- tentative release in mid-February

- distributed / parallel fitness evaluations
    - distribute local cores vs. distributed cluster nodes
    - synchronous vs. asynchronous evaluations
- variable-length genomes
- parsimony pressure
- multi-objective optimization
- minimally complete documentation
    - fleshed out ReadTheDocs documentation
    - technical report

3) ???
