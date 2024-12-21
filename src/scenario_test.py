# -*- coding: utf-8 -*-
# Copyright © 2022 Thales. All Rights Reserved.
# NOTICE: This file is subject to the license agreement defined in file 'LICENSE', which is part of
# this source code package.

import time

from kesslergame import Scenario, KesslerGame, GraphicsType
from diamond_pickaxe_controller import DiamondPickaxeController

from typing import Any

# Define game scenario
my_test_scenario = Scenario(name='Test Scenario',
                            num_asteroids=10,
                            ship_states=[
                                {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
                                # {'position': (400, 600), 'angle': 90, 'lives': 3, 'team': 2, "mines_remaining": 3},
                            ],
                            map_size=(1000, 800),
                            time_limit=60,
                            ammo_limit_multiplier=0,
                            stop_if_no_ammo=False)

# Define Game Settings
game_settings: dict[str, Any] = {
    'perf_tracker': True,
    'graphics_type': GraphicsType.Tkinter,
    'realtime_multiplier': 1,
    'graphics_obj': None,
    'frequency': 30
}

game = KesslerGame(settings=game_settings)  # Use this to visualize the game scenario

# Evaluate the game
pre: float = time.perf_counter()
score, perf_data = game.run(scenario=my_test_scenario, controllers=[DiamondPickaxeController()])

# Print out some general info about the result
print('Scenario eval time: {:.2f} seconds'.format(time.perf_counter()-pre))
print(f'Reason for end of evaluation: {score.stop_reason}')
print(f'Asteroids hit: {score.teams[0].asteroids_hit}')
print(f'Deaths: {score.teams[0].deaths}')
print('Accuracy: {:.2f}%'.format(score.teams[0].accuracy*100))
print('Mean eval time: {:.1f} milliseconds'.format(score.teams[0].mean_eval_time*1000))
print('Mean eval time as percentage of lag-free maximum: {:.1f}%'.format(
    score.teams[0].mean_eval_time / (1/game_settings['frequency']) * 100
))
