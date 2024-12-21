from kesslergame.scenario import Scenario
import random

random.seed(42)

LOG_FILE_PATH: str = "./KesslerController.log"

USE_SIMULATION_CACHE: bool = True # default is True
FLUSH_SIMULATION_CACHE_AFTER_RUN: int = 1000 # default is 1000

MAP_SIZE: tuple[int, int] = (3400, 2000)
LIVES: int = 3
MINES: int = 3
TIME_LIMIT: int = 240

GA_POPULATION_SIZE: int = 42
GA_GENERATION_GOAL: int = 10000
GA_FITNESS_GOAL: float = 10000
GA_NUMBER_OF_PARENTS: int = 21
GA_NUMBER_OF_GENES_TO_MUTATE: int = 1

GA_NUMBER_OF_THREADS: int = 5 # number of threads/processes to use for GA

GA_CHROMOSOME_LENGTH: int = 65

GA_MODEL_FILE: str = "team_cam_controller"

GA_STOP_FLAG_FILE: str = "safely_stop_genetic_learner.txt"

SCENARIOS: list[Scenario] = [
    Scenario(
        name = "Easy Scenario",
        asteroid_states = [
            {
                "position": (random.randint(0, MAP_SIZE[0]), random.randint(0, MAP_SIZE[1])),
                "speed": random.randint(-100, 100)
            }
            for _
            in range(10)
        ],
        ship_states = [
            {
                "position": tuple([i/2 for i in MAP_SIZE]),
                "angle": 90,
                "lives": LIVES,
                "team": 1,
                "mines_remaining": MINES,
            }
        ],
        map_size = MAP_SIZE,
        time_limit = TIME_LIMIT,
        ammo_limit_multiplier = 0,
        stop_if_no_ammo = False,
        seed = 42
    ),
    Scenario(
        name = "Medium Scenario",
        asteroid_states = [
            {
                "position": (random.randint(0, MAP_SIZE[0]), random.randint(0, MAP_SIZE[1])),
                "speed": random.randint(-200, 200)
            }
            for _
            in range(20)
        ],
        ship_states = [
            {
                "position": tuple([i/2 for i in MAP_SIZE]),
                "angle": 90,
                "lives": LIVES,
                "team": 1,
                "mines_remaining": MINES,
            }
        ],
        map_size = MAP_SIZE,
        time_limit = TIME_LIMIT,
        ammo_limit_multiplier = 0,
        stop_if_no_ammo = False,
        seed = 69
    ),
    Scenario(
        name = "Hard Scenario",
        asteroid_states = [
            {
                "position": (random.randint(0, MAP_SIZE[0]), random.randint(0, MAP_SIZE[1])),
                "speed": random.randint(-300, 300)
            }
            for _
            in range(40)
        ],
        ship_states = [
            {
                "position": tuple([i/2 for i in MAP_SIZE]),
                "angle": 90,
                "lives": LIVES,
                "team": 1,
                "mines_remaining": MINES,
            }
        ],
        map_size = MAP_SIZE,
        time_limit = TIME_LIMIT,
        ammo_limit_multiplier = 0,
        stop_if_no_ammo = False,
        seed = 420
    ),
    Scenario(
        name = "Very Hard Scenario",
        asteroid_states = [
            {
                "position": (random.randint(0, MAP_SIZE[0]), random.randint(0, MAP_SIZE[1])),
                "speed": random.randint(-400, 400)
            }
            for _
            in range(60)
        ],
        ship_states = [
            {
                "position": tuple([i/2 for i in MAP_SIZE]),
                "angle": 90,
                "lives": LIVES,
                "team": 1,
                "mines_remaining": MINES,
            }
        ],
        map_size = MAP_SIZE,
        time_limit = TIME_LIMIT,
        ammo_limit_multiplier = 0,
        stop_if_no_ammo = False,
        seed = 666
    ),
    Scenario(
        name = "Extremely Hard Scenario",
        asteroid_states = [
            {
                "position": (random.randint(0, MAP_SIZE[0]), random.randint(0, MAP_SIZE[1])),
                "speed": random.randint(-500, 500)
            }
            for _
            in range(80)
        ],
        ship_states = [
            {
                "position": tuple([i/2 for i in MAP_SIZE]),
                "angle": 90,
                "lives": LIVES,
                "team": 1,
                "mines_remaining": MINES,
            }
        ],
        map_size = MAP_SIZE,
        time_limit = TIME_LIMIT,
        ammo_limit_multiplier = 0,
        stop_if_no_ammo = False,
        seed = 4420
    ),
    Scenario(
        name = "Extremely Hard Scenario",
        asteroid_states = [
            {
                "position": (random.randint(0, MAP_SIZE[0]), random.randint(0, MAP_SIZE[1])),
                "speed": random.randint(-100, 100)
            }
            for _
            in range(100)
        ],
        ship_states = [
            {
                "position": tuple([i/2 for i in MAP_SIZE]),
                "angle": 90,
                "lives": LIVES,
                "team": 1,
                "mines_remaining": MINES,
            }
        ],
        map_size = MAP_SIZE,
        time_limit = TIME_LIMIT,
        ammo_limit_multiplier = 0,
        stop_if_no_ammo = False,
        seed = 4420
    )
]
