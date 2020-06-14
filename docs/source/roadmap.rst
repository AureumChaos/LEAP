# Roadmap

The LEAP development roadmap is as follows:

1. ~~pre-Minimally Viable Product~~ -- released 1/14/2020 as ``0.1-pre``
    - ~~basic support for binary representations~~
        - ~~bit flip mutation~~
        - ~~point-wise crossover~~
        - ~~uniform crossover~~
    - ~~basic support for real-valued representations~~
        - ~~mutate gaussian~~
    - ~~selection operators~~
        - ~~truncation selection~~
        - ~~tournament selection~~
        - ~~random selection~~
        - ~~deterministic cyclic selection~~
        - ~~insertion selection~~
    - ~~continuous integration via Travis~~
    - ~~common test functions~~
        - ~~binary~~
            - ~~MAXONES~~
        - ~~real-valued, optionally translated, rotated, and scaled~~
            - ~~Ackley~~
            - ~~Cosine~~
            - ~~Griewank~~
            - ~~Langermann~~
            - ~~Lunacek~~
            - ~~Noisy Quartic~~
            - ~~Rastrigin~~
            - ~~Rosenbock~~
            - ~~Schwefel~~
            - ~~Shekel~~
            - ~~Spheroid~~
            - ~~Step~~
            - ~~Weierstrass~~
    - ~~test harnesses~~
        - `pytest` ~~supported~~
    - ~~simple usage examples~~
        - ~~canonical EAs~~
            - ~~genetic algorithms (GA)~~
            - ~~evolutionary programming (EP)~~
            - ~~evolutionary strategies (ES)~~
        - ~~simple island model~~
        - ~~basic run-time visualizations~~
        - ~~use with Jupyter notebooks~~
    - ~~documentation outline/stubs for ReadTheDocs~~
1. Minimally Viable Product -- released 6/14/2020 as ``0.2.0``
    - ~~distributed / parallel fitness evaluations~~
        - ~~distribute local cores vs. distributed cluster nodes~~
        - ~~synchronous vs. asynchronous evaluations~~
    - ~~variable-length genomes~~
    - ~~Gray encoding~~
1. Future features, in no particular order of priority
    - parsimony pressure
    - multi-objective optimization
    - minimally complete documentation
        - fleshed out ReadTheDocs documentation
        - technical report
    - checkpoint / restart support
    - hall of fame
    - Rule systems
        - Mich Approach
        - Pitt Approach
    - Genetic Programming (GP)
    - Estimation of Distribution Algorithms (EDA)
        - Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
        - Population-based Incremental Learning (PBIL)
        - Bayesian Optimization Algorithm (BOA)
