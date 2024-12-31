from kesslergame.score import Score
from kesslergame.kessler_game import KesslerGame
from kesslergame.scenario import Scenario
from kesslergame.controller import KesslerController
from kesslergame.team import Team
from diamond_pickaxe_controller import DiamondPickaxeController
from kesslergame.graphics import GraphicsType
from kesslergame.kessler_game import TrainerEnvironment
import config

import os

from typing import Any

import pygad

from chromosome import Chromosome

def execute_fuzzy_inference(
    kessler_game: KesslerGame,
    scenario: Scenario,
    controller: KesslerController
    ) -> Score:
    """executes the fuzzy system and returns the results we care about

    Returns:
        tuple[int, float]: number of asteroids hit, accuracy
    """
    score: Score
    score, _ = kessler_game.run(scenario=scenario, controllers=[controller])

    return score

def fitness_score_function(score: Score, scenario: Scenario) -> float:
    """function to compute a fitness score to be maximized

    Args:
        score (Score): the Score object containing the score parameters from the game

    Returns:
        float: fitness score to be maximized
    """
    team_score: Team = score.teams[0] # get this team's score (assuming we're the only team)

    remaining_time: float = scenario.time_limit - score.sim_time
    average_asteroids_hit_per_second: float = team_score.asteroids_hit / score.sim_time

    fitness_score: float = team_score.asteroids_hit - 30 * team_score.deaths

    if (team_score.lives_remaining > 0):
        fitness_score += (remaining_time * average_asteroids_hit_per_second)

    return fitness_score

def fitness(ga_instance: pygad.GA, chromosome: Chromosome, solution_idx: int, run_with_graphics: bool = False) -> float:
    """runs the controller with the given chromosome
    and returns a fitness score to be maximized

    Args:
        ga_instance (pygad.GA): pygad.GA instance
        chromosome (Chromosome): chromosome to use for the controller fuzzy system
        solution_idx (int): idk lol

    Returns:
        float: fitness score to be maximized
    """
    final_fitness_score: float = 0

    for scenario in config.SCENARIOS:
        controller: DiamondPickaxeController = DiamondPickaxeController(chromosome)

        game_settings: dict[str, Any] = {
            "perf_tracker": True,
            "graphics_type": GraphicsType.Tkinter,
            "realtime_multiplier": 1,
            "graphics_obj": None,
            "frequency": config.FRAME_RATE
        }

        game: KesslerGame
        if run_with_graphics:
            game = KesslerGame(settings = game_settings)
        else:
            game = TrainerEnvironment(settings = game_settings)

        score: Score = execute_fuzzy_inference(game, scenario, controller)

        final_fitness_score += fitness_score_function(score, scenario)

    final_fitness_score /= len(config.SCENARIOS)

    print("iteration fitness: {:.2f}".format(final_fitness_score))

    return final_fitness_score

def fitness_for_pygad(ga_instance: pygad.GA, chromosome: Chromosome, solution_idx: int) -> float:
    return fitness(ga_instance, chromosome, solution_idx, run_with_graphics = False)

def on_generation(ga_instance: pygad.GA):
    ga_instance.save(config.GA_MODEL_FILE)
    print("Generation {:d} completed".format(ga_instance.generations_completed))
    print("Fitness of best solution: {:.2f}".format(ga_instance.best_solution(ga_instance.last_generation_fitness)[1]))
    if check_stop_flag():
        print("Detected change in stop flag file, ending")
        ga_instance.plot_fitness()
        exit(1)

def create_stop_flag_file():
    """creates the flag file or empties it if it exists
    this file can be used to safely stop the genetic learner once the current generation completes,
    simply by adding any text into the file
    """
    with open(config.GA_STOP_FLAG_FILE, "w"):
        pass
    assert (os.stat(config.GA_STOP_FLAG_FILE).st_size == 0)
    
    return

def check_stop_flag() -> bool:
    if (
        not os.path.exists(config.GA_STOP_FLAG_FILE)
        or os.stat(config.GA_STOP_FLAG_FILE).st_size > 0
    ):
        # if flag file was deleted or modified
        return True
    return False

def run_genetic_algorithm():
    create_stop_flag_file()
    if (
        os.path.exists(config.GA_MODEL_FILE+".pkl")
        and os.path.isfile(config.GA_MODEL_FILE+".pkl")
    ):
        print("Continuing training from saved state")
        ga_instance: pygad.GA = pygad.load(config.GA_MODEL_FILE)
        # reset functions to prevent pickling error
        ga_instance.fitness_func = fitness_for_pygad
        ga_instance.on_generation = on_generation
        print("Saved state loaded from file")
    else:
        print("Save file not found,\nRestarting training from scratch")
        ga_instance: pygad.GA = pygad.GA(
            num_generations=config.GA_GENERATION_GOAL,
            num_parents_mating=config.GA_NUMBER_OF_PARENTS,
            fitness_func=fitness_for_pygad,
            sol_per_pop=config.GA_POPULATION_SIZE,
            num_genes=config.GA_CHROMOSOME_LENGTH,
            on_generation=on_generation,
            mutation_num_genes=config.GA_NUMBER_OF_GENES_TO_MUTATE,
            gene_type=float,
            gene_space={"low": 0, "high": 1},
            parallel_processing=["process", config.GA_NUMBER_OF_THREADS]
        )

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution : {solution_idx}")
    ga_instance.save(config.GA_MODEL_FILE)
    ga_instance.plot_fitness()


if __name__ == "__main__":
    run_genetic_algorithm()
