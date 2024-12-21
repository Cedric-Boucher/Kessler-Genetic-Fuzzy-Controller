# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import config as config
from genetic_learner import fitness

import pygad
import os

if (os.path.exists(config.GA_MODEL_FILE+".pkl") and os.path.isfile(config.GA_MODEL_FILE+".pkl")):
    ga_instance: pygad.GA = pygad.load(config.GA_MODEL_FILE)
    best_solution = ga_instance.last_generation_elitism
    assert best_solution is not None
    print(best_solution)
    fitness(ga_instance, best_solution[0], 0, run_with_graphics=True)
