# ECE 449 Intelligent Systems Engineering
# Fall 2023
# Dr. Scott Dick
from typing import Dict, Tuple, Any
from immutabledict import immutabledict
from math import radians, degrees, sqrt, atan2, sin, cos, pi

# Demonstration of a fuzzy tree-based controller for Kessler Game.
# Please see the Kessler Game Development Guide by Dr. Scott Dick for a
#   detailed discussion of this source code.

from kesslergame import KesslerController # In Eclipse, the name of the library is kesslergame, not src.kesslergame
from typing import Dict, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
import config as config

from gene import Gene
from chromosome import Chromosome
from converted_chromosome import ConvertedChromosome

class DiamondPickaxeController(KesslerController): 
    def __init__(self, chromosome: Chromosome):
        self.__current_frame = 0
        self.__name: str = "Diamond Pickaxe"

        self.__chromosome: Chromosome = chromosome
        self.__converted_chromosome: ConvertedChromosome | None = None

        self.__greatest_threat_asteroid_threat_time: ctrl.Antecedent | None = None
        self.__greatest_threat_asteroid_size: ctrl.Antecedent | None = None
        self.__ship_distance_from_nearest_edge: ctrl.Antecedent | None = None
        self.__target_ship_firing_heading_delta: ctrl.Antecedent | None = None
        self.__ship_speed: ctrl.Antecedent | None = None
        self.__ship_stopping_distance: ctrl.Antecedent | None = None
        self.__closest_mine_distance: ctrl.Antecedent | None = None
        self.__closest_mine_remaining_time: ctrl.Antecedent | None = None
        self.__best_next_frame_asteroid_distance: ctrl.Antecedent | None = None
        self.__best_next_frame_asteroid_size: ctrl.Antecedent | None = None
        self.__asteroid_selection: ctrl.Consequent | None = None
        self.__ship_fire: ctrl.Consequent | None = None
        self.__drop_mine: ctrl.Consequent | None = None
        self.__ship_thrust: ctrl.Consequent | None = None
        self.__ship_is_invincible: ctrl.Antecedent | None = None

        self.__asteroid_select_fuzzy_rules: list[ctrl.Rule] | None = None
        self.__ship_fire_fuzzy_rules: list[ctrl.Rule] | None = None
        self.__drop_mine_fuzzy_rules: list[ctrl.Rule] | None = None
        self.__ship_thrust_fuzzy_rules: list[ctrl.Rule] | None = None

        self.__greatest_threat_asteroid_threat_time_range: tuple[float, float] = (0, 100)
        self.__greatest_threat_asteroid_size_range: tuple[float, float] = (0, 4)
        self.__best_next_frame_asteroid_size_range: tuple[float, float] = (0, 4)
        self.__ship_distance_from_nearest_edge_range: tuple[float, float] = (0, 1) # gets set correctly on first iteration of game (once the map size is known)
        self.__target_ship_firing_heading_delta_range: tuple[float, float] = (-pi, pi) # Radians due to Python
        self.__ship_speed_range: tuple[float, float] = (-240, 240) # m/s
        self.__ship_stopping_distance_range: tuple[float, float] = (0, 60) # m
        self.__closest_mine_distance_range: tuple[float, float] = (0, 1000) # m
        self.__closest_mine_remaining_time_range: tuple[float, float] = (0, 100)
        self.__best_next_frame_asteroid_distance_range: tuple[float, float] = (0, 1000) # m
        self.__ship_turn_range: tuple[float, float] = (-180, 180) # Degrees due to Kessler
        self.__ship_fire_range: tuple[float, float] = (-1, 1)
        self.__ship_drop_mine_range: tuple[float, float] = (-1, 1)
        self.__ship_thrust_range: tuple[float, float] = (-480.0, 480.0) # m/s^2
        self.__asteroid_selection_range: tuple[float, float] = (-1, 1)
        self.__ship_is_invincible_range: tuple[float, float] = (-1, 1)

        self.__asteroid_select_simulation: ctrl.ControlSystemSimulation | None = None
        self.__ship_fire_simulation: ctrl.ControlSystemSimulation | None = None
        self.__drop_mine_simulation: ctrl.ControlSystemSimulation | None = None
        self.__ship_thrust_simulation: ctrl.ControlSystemSimulation | None = None

        self.__setup_simulations()

    def __convert_chromosome(self) -> None:
        """converts a list of floats into something usable by setup_fuzzy_sets

        Args:
            chromosome (Chromosome): a list of floats

        Returns:
            ConvertedChromosome: something in a format usable in trimf functions
        """
        chromosome_list: list[float] = self.__chromosome.tolist()

        start_gene_index: int = 0
        genes_needed: int = 3
        end_gene_index: int = start_gene_index + genes_needed
        values: list[float] = chromosome_list[start_gene_index:end_gene_index]
        values.extend([-0.01, 1.01])
        values = sorted(values)
        ship_distance_from_nearest_edge_gene: Gene = { # type: ignore
            "S": tuple(values[0:3]),
            "M": tuple(values[1:4]),
            "L": tuple(values[2:5])
        }
        ship_distance_from_nearest_edge_gene = self.__scale_gene(
            ship_distance_from_nearest_edge_gene,
            self.__ship_distance_from_nearest_edge_range[0],
            self.__ship_distance_from_nearest_edge_range[1]
        )

        start_gene_index = end_gene_index
        genes_needed = 7
        end_gene_index = start_gene_index + genes_needed
        values: list[float] = chromosome_list[start_gene_index:end_gene_index]
        values.extend([-0.01, 1.01])
        values = sorted(values)
        target_ship_firing_heading_delta_gene: Gene = { # type: ignore
            "NL": tuple(values[0:3]),
            "NM": tuple(values[1:4]),
            "NS": tuple(values[2:5]),
            "Z": tuple(values[3:6]),
            "PS": tuple(values[4:7]),
            "PM": tuple(values[5:8]),
            "PL": tuple(values[6:9])
        }
        target_ship_firing_heading_delta_gene = self.__scale_gene(
            target_ship_firing_heading_delta_gene,
            self.__target_ship_firing_heading_delta_range[0],
            self.__target_ship_firing_heading_delta_range[1]
        )

        start_gene_index = end_gene_index
        genes_needed = 2
        end_gene_index = start_gene_index + genes_needed
        values: list[float] = chromosome_list[start_gene_index:end_gene_index]
        values.extend([-0.01, 1.01])
        values = sorted(values)
        drop_mine_gene: Gene = { # type: ignore
            "N": tuple(values[0:3]),
            "Y": tuple(values[1:4])
        }
        drop_mine_gene = self.__scale_gene(
            drop_mine_gene,
            self.__ship_drop_mine_range[0],
            self.__ship_drop_mine_range[1]
        )

        start_gene_index = end_gene_index
        genes_needed = 7
        end_gene_index = start_gene_index + genes_needed
        values: list[float] = chromosome_list[start_gene_index:end_gene_index]
        values.extend([-0.01, 1.01])
        values = sorted(values)
        ship_thrust_gene: Gene = { # type: ignore
            "NL": tuple(values[0:3]),
            "NM": tuple(values[1:4]),
            "NS": tuple(values[2:5]),
            "Z": tuple(values[3:6]),
            "PS": tuple(values[4:7]),
            "PM": tuple(values[5:8]),
            "PL": tuple(values[6:9])
        }
        ship_thrust_gene = self.__scale_gene(
            ship_thrust_gene,
            self.__ship_thrust_range[0],
            self.__ship_thrust_range[1]
        )

        start_gene_index = end_gene_index
        genes_needed = 7
        end_gene_index = start_gene_index + genes_needed
        values: list[float] = chromosome_list[start_gene_index:end_gene_index]
        values.extend([-0.01, 1.01])
        values = sorted(values)
        ship_speed_gene: Gene = { # type: ignore
            "NL": tuple(values[0:3]),
            "NM": tuple(values[1:4]),
            "NS": tuple(values[2:5]),
            "Z": tuple(values[3:6]),
            "PS": tuple(values[4:7]),
            "PM": tuple(values[5:8]),
            "PL": tuple(values[6:9])
        }
        ship_speed_gene = self.__scale_gene(
            ship_speed_gene,
            self.__ship_speed_range[0],
            self.__ship_speed_range[1]
        )

        start_gene_index = end_gene_index
        genes_needed = 4
        end_gene_index = start_gene_index + genes_needed
        values: list[float] = chromosome_list[start_gene_index:end_gene_index]
        values.extend([-0.01, 1.01])
        values = sorted(values)
        ship_stopping_distance_gene: Gene = { # type: ignore
            "Z": tuple(values[0:3]),
            "PS": tuple(values[1:4]),
            "PM": tuple(values[2:5]),
            "PL": tuple(values[3:6])
        }
        ship_stopping_distance_gene = self.__scale_gene(
            ship_stopping_distance_gene,
            self.__ship_stopping_distance_range[0],
            self.__ship_stopping_distance_range[1]
        )

        start_gene_index = end_gene_index
        genes_needed = 2
        end_gene_index = start_gene_index + genes_needed
        values: list[float] = chromosome_list[start_gene_index:end_gene_index]
        values.extend([-0.01, 1.01])
        values = sorted(values)
        closest_mine_distance_gene: Gene = { # type: ignore
            "in_range": tuple(values[0:3]),
            "out_of_range": tuple(values[1:4])
        }
        closest_mine_distance_gene = self.__scale_gene(
            closest_mine_distance_gene,
            self.__closest_mine_distance_range[0],
            self.__closest_mine_distance_range[1]
        )

        start_gene_index = end_gene_index
        genes_needed = 4
        end_gene_index = start_gene_index + genes_needed
        values: list[float] = chromosome_list[start_gene_index:end_gene_index]
        values.extend([-0.01, 1.01])
        values = sorted(values)
        best_next_frame_asteroid_distance_gene: Gene = { # type: ignore
            "Z": tuple(values[0:3]),
            "PS": tuple(values[1:4]),
            "PM": tuple(values[2:5]),
            "PL": tuple(values[3:6])
        }
        best_next_frame_asteroid_distance_gene = self.__scale_gene(
            best_next_frame_asteroid_distance_gene,
            self.__best_next_frame_asteroid_distance_range[0],
            self.__best_next_frame_asteroid_distance_range[1]
        )

        start_gene_index = end_gene_index
        genes_needed = 5
        end_gene_index = start_gene_index + genes_needed
        values: list[float] = chromosome_list[start_gene_index:end_gene_index]
        values.extend([-0.01, 1.01])
        values = sorted(values)
        greatest_threat_asteroid_threat_time_gene: Gene = { # type: ignore
            "XS": tuple(values[0:3]),
            "S": tuple(values[1:4]),
            "M": tuple(values[2:5]),
            "L": tuple(values[3:6]),
            "XL": tuple(values[4:7])
        }
        greatest_threat_asteroid_threat_time_gene = self.__scale_gene(
            greatest_threat_asteroid_threat_time_gene,
            self.__greatest_threat_asteroid_threat_time_range[0],
            self.__greatest_threat_asteroid_threat_time_range[1]
        )

        start_gene_index = end_gene_index
        genes_needed = 4
        end_gene_index = start_gene_index + genes_needed
        values: list[float] = chromosome_list[start_gene_index:end_gene_index]
        values.extend([-0.01, 1.01])
        values = sorted(values)
        greatest_threat_asteroid_size_gene: Gene = { # type: ignore
            "S": tuple(values[0:3]),
            "M": tuple(values[1:4]),
            "L": tuple(values[2:5]),
            "XL": tuple(values[3:6])
        }
        greatest_threat_asteroid_size_gene = self.__scale_gene(
            greatest_threat_asteroid_size_gene,
            self.__greatest_threat_asteroid_size_range[0],
            self.__greatest_threat_asteroid_size_range[1]
        )

        start_gene_index = end_gene_index
        genes_needed = 4
        end_gene_index = start_gene_index + genes_needed
        values: list[float] = chromosome_list[start_gene_index:end_gene_index]
        values.extend([-0.01, 1.01])
        values = sorted(values)
        best_next_frame_asteroid_size_gene: Gene = { # type: ignore
            "S": tuple(values[0:3]),
            "M": tuple(values[1:4]),
            "L": tuple(values[2:5]),
            "XL": tuple(values[3:6])
        }
        best_next_frame_asteroid_size_gene = self.__scale_gene(
            best_next_frame_asteroid_size_gene,
            self.__best_next_frame_asteroid_size_range[0],
            self.__best_next_frame_asteroid_size_range[1]
        )

        start_gene_index = end_gene_index
        genes_needed = 3
        end_gene_index = start_gene_index + genes_needed
        values: list[float] = chromosome_list[start_gene_index:end_gene_index]
        values.extend([-0.01, 1.01])
        values = sorted(values)
        closest_mine_remaining_time_gene: Gene = { # type: ignore
            "S": tuple(values[0:3]),
            "M": tuple(values[1:4]),
            "L": tuple(values[2:5])
        }
        closest_mine_remaining_time_gene = self.__scale_gene(
            closest_mine_remaining_time_gene,
            self.__closest_mine_remaining_time_range[0],
            self.__closest_mine_remaining_time_range[1]
        )

        start_gene_index = end_gene_index
        genes_needed = 2
        end_gene_index = start_gene_index + genes_needed
        values: list[float] = chromosome_list[start_gene_index:end_gene_index]
        values.extend([-0.01, 1.01])
        values = sorted(values)
        asteroid_selection_gene: Gene = { # type: ignore
            "closest": tuple(values[0:3]),
            "greatest_threat": tuple(values[1:4]),
        }
        asteroid_selection_gene = self.__scale_gene(
            asteroid_selection_gene,
            self.__asteroid_selection_range[0],
            self.__asteroid_selection_range[1]
        )

        start_gene_index = end_gene_index
        genes_needed = 2
        end_gene_index = start_gene_index + genes_needed
        values: list[float] = chromosome_list[start_gene_index:end_gene_index]
        values.extend([-0.01, 1.01])
        values = sorted(values)
        ship_is_invincible_gene: Gene = { # type: ignore
            "N": tuple(values[0:3]),
            "Y": tuple(values[1:4])
        }
        ship_is_invincible_gene = self.__scale_gene(
            ship_is_invincible_gene,
            self.__ship_is_invincible_range[0],
            self.__ship_is_invincible_range[1]
        )

        start_gene_index = end_gene_index
        genes_needed = 2
        end_gene_index = start_gene_index + genes_needed
        values: list[float] = chromosome_list[start_gene_index:end_gene_index]
        values.extend([-0.01, 1.01])
        values = sorted(values)
        ship_fire_gene: Gene = { # type: ignore
            "N": tuple(values[0:3]),
            "Y": tuple(values[1:4])
        }
        ship_fire_gene = self.__scale_gene(
            ship_fire_gene,
            self.__ship_fire_range[0],
            self.__ship_fire_range[1]
        )

        self.__converted_chromosome = {
            "ship_distance_from_nearest_edge": ship_distance_from_nearest_edge_gene,
            "target_ship_firing_heading_delta": target_ship_firing_heading_delta_gene,
            "ship_speed": ship_speed_gene,
            "ship_stopping_distance": ship_stopping_distance_gene,
            "closest_mine_distance": closest_mine_distance_gene,
            "best_next_frame_asteroid_distance": best_next_frame_asteroid_distance_gene,
            "greatest_threat_asteroid_threat_time": greatest_threat_asteroid_threat_time_gene,
            "greatest_threat_asteroid_size": greatest_threat_asteroid_size_gene,
            "best_next_frame_asteroid_size": best_next_frame_asteroid_size_gene,
            "closest_mine_remaining_time": closest_mine_remaining_time_gene,
            "asteroid_selection": asteroid_selection_gene,
            "ship_fire": ship_fire_gene,
            "drop_mine": drop_mine_gene,
            "ship_thrust": ship_thrust_gene,
            "ship_is_invincible": ship_is_invincible_gene
        }

    @staticmethod
    def __scale_gene(gene: Gene, minimum: float, maximum: float) -> Gene:
        scaled_gene: Gene = dict()
        for key in gene.keys():
            scaled_gene[key] = tuple([(gene[key][i] * (maximum - minimum)) + minimum for i in range(3)]) # type: ignore

        return scaled_gene

    def __setup_antecedents_and_consequents(self) -> None:
        self.__ship_distance_from_nearest_edge = ctrl.Antecedent(np.arange(self.__ship_distance_from_nearest_edge_range[0], self.__ship_distance_from_nearest_edge_range[1], 1), 'ship_distance_from_nearest_edge')
        self.__target_ship_firing_heading_delta = ctrl.Antecedent(np.arange(self.__target_ship_firing_heading_delta_range[0], self.__target_ship_firing_heading_delta_range[1], 0.01), 'target_ship_firing_heading_delta')
        self.__ship_speed = ctrl.Antecedent(np.arange(self.__ship_speed_range[0], self.__ship_speed_range[1], 5), 'ship_speed')
        self.__ship_stopping_distance = ctrl.Antecedent(np.arange(self.__ship_stopping_distance_range[0], self.__ship_stopping_distance_range[1], 1), 'ship_stopping_distance')
        self.__closest_mine_distance = ctrl.Antecedent(np.arange(self.__closest_mine_distance_range[0], self.__closest_mine_distance_range[1], 1), 'closest_mine_distance')
        self.__closest_mine_remaining_time = ctrl.Antecedent(np.arange(self.__closest_mine_remaining_time_range[0], self.__closest_mine_remaining_time_range[1], 0.1), 'closest_mine_remaining_time')
        self.__best_next_frame_asteroid_distance = ctrl.Antecedent(np.arange(self.__best_next_frame_asteroid_distance_range[0], self.__best_next_frame_asteroid_distance_range[1], 1), 'best_next_frame_asteroid_distance')
        self.__greatest_threat_asteroid_threat_time = ctrl.Antecedent(np.arange(self.__greatest_threat_asteroid_threat_time_range[0], self.__greatest_threat_asteroid_threat_time_range[1], 0.01), 'greatest_threat_asteroid_threat_time')
        self.__greatest_threat_asteroid_size = ctrl.Antecedent(np.arange(self.__greatest_threat_asteroid_size_range[0], self.__greatest_threat_asteroid_size_range[1], 0.1), 'greatest_threat_asteroid_size')
        self.__best_next_frame_asteroid_size = ctrl.Antecedent(np.arange(self.__best_next_frame_asteroid_size_range[0], self.__best_next_frame_asteroid_size_range[1], 0.1), 'best_next_frame_asteroid_size')
        self.__ship_is_invincible = ctrl.Antecedent(np.arange(self.__ship_is_invincible_range[0], self.__ship_is_invincible_range[1], 0.1), 'ship_is_invincible')

        self.__asteroid_selection = ctrl.Consequent(np.arange(self.__asteroid_selection_range[0], self.__asteroid_selection_range[1], 0.1), 'asteroid_selection')
        self.__ship_fire = ctrl.Consequent(np.arange(self.__ship_fire_range[0], self.__ship_fire_range[1], 0.1), 'ship_fire')
        self.__drop_mine = ctrl.Consequent(np.arange(self.__ship_drop_mine_range[0], self.__ship_drop_mine_range[1], 0.1), 'drop_mine')
        self.__ship_thrust = ctrl.Consequent(np.arange(self.__ship_thrust_range[0], self.__ship_thrust_range[1], 5), 'ship_thrust')

    def __setup_fuzzy_sets(self) -> None:
        self.__convert_chromosome()
        assert (self.__converted_chromosome is not None)

        self.__setup_antecedents_and_consequents()
        assert (self.__greatest_threat_asteroid_threat_time is not None)
        assert (self.__greatest_threat_asteroid_size is not None)
        assert (self.__ship_distance_from_nearest_edge is not None)
        assert (self.__target_ship_firing_heading_delta is not None)
        assert (self.__ship_speed is not None)
        assert (self.__ship_stopping_distance is not None)
        assert (self.__closest_mine_distance is not None)
        assert (self.__closest_mine_remaining_time is not None)
        assert (self.__best_next_frame_asteroid_distance is not None)
        assert (self.__best_next_frame_asteroid_size is not None)
        assert (self.__asteroid_selection is not None)
        assert (self.__ship_fire is not None)
        assert (self.__drop_mine is not None)
        assert (self.__ship_thrust is not None)
        assert (self.__ship_is_invincible is not None)

        greatest_threat_asteroid_threat_time_gene: Gene = self.__converted_chromosome["greatest_threat_asteroid_threat_time"]
        self.__greatest_threat_asteroid_threat_time['XS'] = fuzz.trimf(self.__greatest_threat_asteroid_threat_time.universe, greatest_threat_asteroid_threat_time_gene["XS"])
        self.__greatest_threat_asteroid_threat_time['S'] = fuzz.trimf(self.__greatest_threat_asteroid_threat_time.universe, greatest_threat_asteroid_threat_time_gene["S"])
        self.__greatest_threat_asteroid_threat_time['M'] = fuzz.trimf(self.__greatest_threat_asteroid_threat_time.universe, greatest_threat_asteroid_threat_time_gene["M"])
        self.__greatest_threat_asteroid_threat_time['L'] = fuzz.trimf(self.__greatest_threat_asteroid_threat_time.universe, greatest_threat_asteroid_threat_time_gene["L"])
        self.__greatest_threat_asteroid_threat_time['XL'] = fuzz.trimf(self.__greatest_threat_asteroid_threat_time.universe, greatest_threat_asteroid_threat_time_gene["XL"])

        # there are 4 possible asteroid sizes in the game
        greatest_threat_asteroid_size_gene: Gene = self.__converted_chromosome["greatest_threat_asteroid_size"]
        self.__greatest_threat_asteroid_size['S'] = fuzz.trimf(self.__greatest_threat_asteroid_size.universe, greatest_threat_asteroid_size_gene["S"])
        self.__greatest_threat_asteroid_size['M'] = fuzz.trimf(self.__greatest_threat_asteroid_size.universe, greatest_threat_asteroid_size_gene["M"])
        self.__greatest_threat_asteroid_size['L'] = fuzz.trimf(self.__greatest_threat_asteroid_size.universe, greatest_threat_asteroid_size_gene["L"])
        self.__greatest_threat_asteroid_size['XL'] = fuzz.trimf(self.__greatest_threat_asteroid_size.universe, greatest_threat_asteroid_size_gene["XL"])

        best_next_frame_asteroid_size_gene: Gene = self.__converted_chromosome["best_next_frame_asteroid_size"]
        self.__best_next_frame_asteroid_size['S'] = fuzz.trimf(self.__best_next_frame_asteroid_size.universe, best_next_frame_asteroid_size_gene["S"])
        self.__best_next_frame_asteroid_size['M'] = fuzz.trimf(self.__best_next_frame_asteroid_size.universe, best_next_frame_asteroid_size_gene["M"])
        self.__best_next_frame_asteroid_size['L'] = fuzz.trimf(self.__best_next_frame_asteroid_size.universe, best_next_frame_asteroid_size_gene["L"])
        self.__best_next_frame_asteroid_size['XL'] = fuzz.trimf(self.__best_next_frame_asteroid_size.universe, best_next_frame_asteroid_size_gene["XL"])

        ship_is_invincible_gene: Gene = self.__converted_chromosome["ship_is_invincible"]
        self.__ship_is_invincible['Y'] = fuzz.trimf(self.__ship_is_invincible.universe, ship_is_invincible_gene["Y"])
        self.__ship_is_invincible['N'] = fuzz.trimf(self.__ship_is_invincible.universe, ship_is_invincible_gene["N"])

        #Declare fuzzy sets for ship_distance_from_nearest_edge (how long it takes for the bullet to reach the intercept point)
        ship_distance_from_nearest_edge_gene: Gene = self.__converted_chromosome["ship_distance_from_nearest_edge"]
        self.__ship_distance_from_nearest_edge['S'] = fuzz.trimf(self.__ship_distance_from_nearest_edge.universe, ship_distance_from_nearest_edge_gene["S"])
        self.__ship_distance_from_nearest_edge['M'] = fuzz.trimf(self.__ship_distance_from_nearest_edge.universe, ship_distance_from_nearest_edge_gene["M"])
        self.__ship_distance_from_nearest_edge['L'] = fuzz.trimf(self.__ship_distance_from_nearest_edge.universe, ship_distance_from_nearest_edge_gene["L"])

        # Declare fuzzy sets for target_ship_firing_heading_delta (degrees of turn needed to reach the calculated firing angle)
        # Hard-coded for a game step of 1/30 seconds
        target_ship_firing_heading_delta_gene: Gene = self.__converted_chromosome["target_ship_firing_heading_delta"]
        self.__target_ship_firing_heading_delta['NL'] = fuzz.trimf(self.__target_ship_firing_heading_delta.universe, target_ship_firing_heading_delta_gene["NL"])
        self.__target_ship_firing_heading_delta['NM'] = fuzz.trimf(self.__target_ship_firing_heading_delta.universe, target_ship_firing_heading_delta_gene["NM"])
        self.__target_ship_firing_heading_delta['NS'] = fuzz.trimf(self.__target_ship_firing_heading_delta.universe, target_ship_firing_heading_delta_gene["NS"])
        self.__target_ship_firing_heading_delta['Z']  = fuzz.trimf(self.__target_ship_firing_heading_delta.universe, target_ship_firing_heading_delta_gene["Z"])
        self.__target_ship_firing_heading_delta['PS'] = fuzz.trimf(self.__target_ship_firing_heading_delta.universe, target_ship_firing_heading_delta_gene["PS"])
        self.__target_ship_firing_heading_delta['PM'] = fuzz.trimf(self.__target_ship_firing_heading_delta.universe, target_ship_firing_heading_delta_gene["PM"])
        self.__target_ship_firing_heading_delta['PL'] = fuzz.trimf(self.__target_ship_firing_heading_delta.universe, target_ship_firing_heading_delta_gene["PL"])

        ship_speed_gene: Gene = self.__converted_chromosome["ship_speed"]
        self.__ship_speed['NL'] = fuzz.trimf(self.__ship_speed.universe, ship_speed_gene["NL"])
        self.__ship_speed['NM'] = fuzz.trimf(self.__ship_speed.universe, ship_speed_gene["NM"])
        self.__ship_speed['NS'] = fuzz.trimf(self.__ship_speed.universe, ship_speed_gene["NS"])
        self.__ship_speed['Z']  = fuzz.trimf(self.__ship_speed.universe, ship_speed_gene["Z"])
        self.__ship_speed['PS'] = fuzz.trimf(self.__ship_speed.universe, ship_speed_gene["PS"])
        self.__ship_speed['PM'] = fuzz.trimf(self.__ship_speed.universe, ship_speed_gene["PM"])
        self.__ship_speed['PL'] = fuzz.trimf(self.__ship_speed.universe, ship_speed_gene["PL"])

        ship_stopping_distance_gene: Gene = self.__converted_chromosome["ship_stopping_distance"]
        self.__ship_stopping_distance['Z']  = fuzz.trimf(self.__ship_stopping_distance.universe, ship_stopping_distance_gene["Z"])
        self.__ship_stopping_distance['PS'] = fuzz.trimf(self.__ship_stopping_distance.universe, ship_stopping_distance_gene["PS"])
        self.__ship_stopping_distance['PM'] = fuzz.trimf(self.__ship_stopping_distance.universe, ship_stopping_distance_gene["PM"])
        self.__ship_stopping_distance['PL'] = fuzz.trimf(self.__ship_stopping_distance.universe, ship_stopping_distance_gene["PL"])

        closest_mine_distance_gene: Gene = self.__converted_chromosome["closest_mine_distance"]
        self.__closest_mine_distance['in_range']  = fuzz.trimf(self.__closest_mine_distance.universe, closest_mine_distance_gene["in_range"])
        self.__closest_mine_distance['out_of_range'] = fuzz.trimf(self.__closest_mine_distance.universe, closest_mine_distance_gene["out_of_range"])

        closest_mine_remaining_time_gene: Gene = self.__converted_chromosome["closest_mine_remaining_time"]
        self.__closest_mine_remaining_time['S'] = fuzz.trimf(self.__closest_mine_remaining_time.universe, closest_mine_remaining_time_gene["S"])
        self.__closest_mine_remaining_time['M'] = fuzz.trimf(self.__closest_mine_remaining_time.universe, closest_mine_remaining_time_gene["M"])
        self.__closest_mine_remaining_time['L'] = fuzz.trimf(self.__closest_mine_remaining_time.universe, closest_mine_remaining_time_gene["L"])

        best_next_frame_asteroid_distance_gene: Gene = self.__converted_chromosome["best_next_frame_asteroid_distance"]
        self.__best_next_frame_asteroid_distance['Z']  = fuzz.trimf(self.__best_next_frame_asteroid_distance.universe, best_next_frame_asteroid_distance_gene["Z"])
        self.__best_next_frame_asteroid_distance['PS'] = fuzz.trimf(self.__best_next_frame_asteroid_distance.universe, best_next_frame_asteroid_distance_gene["PS"])
        self.__best_next_frame_asteroid_distance['PM'] = fuzz.trimf(self.__best_next_frame_asteroid_distance.universe, best_next_frame_asteroid_distance_gene["PM"])
        self.__best_next_frame_asteroid_distance['PL'] = fuzz.trimf(self.__best_next_frame_asteroid_distance.universe, best_next_frame_asteroid_distance_gene["PL"])

        asteroid_selection_gene: Gene = self.__converted_chromosome["asteroid_selection"]
        self.__asteroid_selection['closest'] = fuzz.trimf(self.__asteroid_selection.universe, asteroid_selection_gene["closest"])
        self.__asteroid_selection['greatest_threat'] = fuzz.trimf(self.__asteroid_selection.universe, asteroid_selection_gene["greatest_threat"])

        ship_fire_gene: Gene = self.__converted_chromosome["ship_fire"]
        self.__ship_fire['N'] = fuzz.trimf(self.__ship_fire.universe, ship_fire_gene["N"])
        self.__ship_fire['Y'] = fuzz.trimf(self.__ship_fire.universe, ship_fire_gene["Y"])

        drop_mine_gene: Gene = self.__converted_chromosome["drop_mine"]
        self.__drop_mine['N'] = fuzz.trimf(self.__drop_mine.universe, drop_mine_gene["N"])
        self.__drop_mine['Y'] = fuzz.trimf(self.__drop_mine.universe, drop_mine_gene["Y"])

        ship_thrust_gene: Gene = self.__converted_chromosome["ship_thrust"]
        self.__ship_thrust['NL'] = fuzz.trimf(self.__ship_thrust.universe, ship_thrust_gene["NL"])
        self.__ship_thrust['NM'] = fuzz.trimf(self.__ship_thrust.universe, ship_thrust_gene["NM"])
        self.__ship_thrust['NS'] = fuzz.trimf(self.__ship_thrust.universe, ship_thrust_gene["NS"])
        self.__ship_thrust['Z']  = fuzz.trimf(self.__ship_thrust.universe, ship_thrust_gene["Z"])
        self.__ship_thrust['PS'] = fuzz.trimf(self.__ship_thrust.universe, ship_thrust_gene["PS"])
        self.__ship_thrust['PM'] = fuzz.trimf(self.__ship_thrust.universe, ship_thrust_gene["PM"])
        self.__ship_thrust['PL'] = fuzz.trimf(self.__ship_thrust.universe, ship_thrust_gene["PL"])

    def __setup_fuzzy_rules(self) -> None:
        self.__setup_fuzzy_sets()
        assert (self.__greatest_threat_asteroid_threat_time is not None)
        assert (self.__greatest_threat_asteroid_size is not None)
        assert (self.__ship_distance_from_nearest_edge is not None)
        assert (self.__target_ship_firing_heading_delta is not None)
        assert (self.__ship_speed is not None)
        assert (self.__ship_stopping_distance is not None)
        assert (self.__closest_mine_distance is not None)
        assert (self.__closest_mine_remaining_time is not None)
        assert (self.__best_next_frame_asteroid_distance is not None)
        assert (self.__best_next_frame_asteroid_size is not None)
        assert (self.__asteroid_selection is not None)
        assert (self.__ship_fire is not None)
        assert (self.__drop_mine is not None)
        assert (self.__ship_thrust is not None)
        assert (self.__ship_is_invincible is not None)

        self.__asteroid_select_fuzzy_rules = [
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['XS'] & self.__greatest_threat_asteroid_size['S'],
                self.__asteroid_selection['greatest_threat']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['XS'] & self.__greatest_threat_asteroid_size['M'],
                self.__asteroid_selection['greatest_threat']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['XS'] & self.__greatest_threat_asteroid_size['L'],
                self.__asteroid_selection['greatest_threat']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['XS'] & self.__greatest_threat_asteroid_size['XL'],
                self.__asteroid_selection['greatest_threat']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['S'] & self.__greatest_threat_asteroid_size['S'] & self.__best_next_frame_asteroid_size['S'],
                self.__asteroid_selection['closest']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['S'] & self.__greatest_threat_asteroid_size['S'] & self.__best_next_frame_asteroid_size['M'],
                self.__asteroid_selection['greatest_threat']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['S'] & self.__greatest_threat_asteroid_size['S'] & self.__best_next_frame_asteroid_size['L'],
                self.__asteroid_selection['greatest_threat']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['S'] & self.__greatest_threat_asteroid_size['S'] & self.__best_next_frame_asteroid_size['XL'],
                self.__asteroid_selection['greatest_threat']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['S'] & self.__greatest_threat_asteroid_size['M'],
                self.__asteroid_selection['greatest_threat']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['S'] & self.__greatest_threat_asteroid_size['L'],
                self.__asteroid_selection['greatest_threat']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['S'] & self.__greatest_threat_asteroid_size['XL'],
                self.__asteroid_selection['greatest_threat']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['M'] & self.__greatest_threat_asteroid_size['S'] & self.__best_next_frame_asteroid_size['S'],
                self.__asteroid_selection['closest']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['M'] & self.__greatest_threat_asteroid_size['S'] & self.__best_next_frame_asteroid_size['M'],
                self.__asteroid_selection['greatest_threat']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['M'] & self.__greatest_threat_asteroid_size['S'] & self.__best_next_frame_asteroid_size['L'],
                self.__asteroid_selection['greatest_threat']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['M'] & self.__greatest_threat_asteroid_size['S'] & self.__best_next_frame_asteroid_size['XL'],
                self.__asteroid_selection['greatest_threat']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['M'] & self.__greatest_threat_asteroid_size['M'] & self.__best_next_frame_asteroid_size['S'],
                self.__asteroid_selection['closest']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['M'] & self.__greatest_threat_asteroid_size['M'] & self.__best_next_frame_asteroid_size['M'],
                self.__asteroid_selection['closest']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['M'] & self.__greatest_threat_asteroid_size['M'] & self.__best_next_frame_asteroid_size['L'],
                self.__asteroid_selection['greatest_threat']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['M'] & self.__greatest_threat_asteroid_size['M'] & self.__best_next_frame_asteroid_size['XL'],
                self.__asteroid_selection['greatest_threat']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['M'] & self.__greatest_threat_asteroid_size['L'],
                self.__asteroid_selection['greatest_threat']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['M'] & self.__greatest_threat_asteroid_size['XL'],
                self.__asteroid_selection['greatest_threat']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['L'] & self.__greatest_threat_asteroid_size['S'] & self.__best_next_frame_asteroid_size['S'],
                self.__asteroid_selection['closest']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['L'] & self.__greatest_threat_asteroid_size['S'] & self.__best_next_frame_asteroid_size['M'],
                self.__asteroid_selection['closest']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['L'] & self.__greatest_threat_asteroid_size['S'] & self.__best_next_frame_asteroid_size['L'],
                self.__asteroid_selection['greatest_threat']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['L'] & self.__greatest_threat_asteroid_size['S'] & self.__best_next_frame_asteroid_size['XL'],
                self.__asteroid_selection['greatest_threat']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['L'] & self.__greatest_threat_asteroid_size['M'] & self.__best_next_frame_asteroid_size['S'],
                self.__asteroid_selection['closest']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['L'] & self.__greatest_threat_asteroid_size['M'] & self.__best_next_frame_asteroid_size['M'],
                self.__asteroid_selection['closest']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['L'] & self.__greatest_threat_asteroid_size['M'] & self.__best_next_frame_asteroid_size['L'],
                self.__asteroid_selection['closest']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['L'] & self.__greatest_threat_asteroid_size['M'] & self.__best_next_frame_asteroid_size['XL'],
                self.__asteroid_selection['greatest_threat']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['L'] & self.__greatest_threat_asteroid_size['L'] & self.__best_next_frame_asteroid_size['S'],
                self.__asteroid_selection['closest']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['L'] & self.__greatest_threat_asteroid_size['L'] & self.__best_next_frame_asteroid_size['M'],
                self.__asteroid_selection['closest']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['L'] & self.__greatest_threat_asteroid_size['L'] & self.__best_next_frame_asteroid_size['L'],
                self.__asteroid_selection['closest']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['L'] & self.__greatest_threat_asteroid_size['L'] & self.__best_next_frame_asteroid_size['XL'],
                self.__asteroid_selection['greatest_threat']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['L'] & self.__greatest_threat_asteroid_size['XL'],
                self.__asteroid_selection['greatest_threat']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['XL'] & self.__greatest_threat_asteroid_size['S'],
                self.__asteroid_selection['closest']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['XL'] & self.__greatest_threat_asteroid_size['M'],
                self.__asteroid_selection['closest']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['XL'] & self.__greatest_threat_asteroid_size['L'],
                self.__asteroid_selection['closest']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['XL'] & self.__greatest_threat_asteroid_size['XL'],
                self.__asteroid_selection['closest']
            )
        ]

        self.__ship_fire_fuzzy_rules = [
            ctrl.Rule(
                self.__ship_is_invincible['N'],
                self.__ship_fire['Y']
            ),
            ctrl.Rule(
                self.__ship_is_invincible['Y'] & (self.__greatest_threat_asteroid_threat_time['XS'] | self.__greatest_threat_asteroid_threat_time['S'] | self.__greatest_threat_asteroid_threat_time['M']),
                self.__ship_fire['N']
            ),
            ctrl.Rule(
                self.__ship_is_invincible['Y'] & (self.__greatest_threat_asteroid_threat_time['L'] | self.__greatest_threat_asteroid_threat_time['XL']),
                self.__ship_fire['Y']
            )
        ]

        self.__drop_mine_fuzzy_rules = [
            ctrl.Rule(
                self.__ship_is_invincible['Y'],
                self.__drop_mine['N']
            ),
            ctrl.Rule(
                self.__greatest_threat_asteroid_threat_time['XS'] & self.__ship_is_invincible['N'],
                self.__drop_mine['Y']
            ),
            ctrl.Rule(
                (self.__greatest_threat_asteroid_threat_time['S'] | self.__greatest_threat_asteroid_threat_time['M'] | self.__greatest_threat_asteroid_threat_time['L'] | self.__greatest_threat_asteroid_threat_time['XL']) & self.__ship_is_invincible['N'],
                self.__drop_mine['N']
            )
        ]

        self.__ship_thrust_fuzzy_rules = [
            ctrl.Rule(
                (self.__closest_mine_remaining_time['L'] | self.__closest_mine_distance['out_of_range']) & (self.__greatest_threat_asteroid_threat_time['XS'] | self.__greatest_threat_asteroid_threat_time['S']) & (self.__target_ship_firing_heading_delta['NL'] | self.__target_ship_firing_heading_delta['PL']),
                self.__ship_thrust['PL']
            ),
            ctrl.Rule(
                (self.__closest_mine_remaining_time['L'] | self.__closest_mine_distance['out_of_range']) & (self.__greatest_threat_asteroid_threat_time['XS'] | self.__greatest_threat_asteroid_threat_time['S']) & (self.__target_ship_firing_heading_delta['NS'] | self.__target_ship_firing_heading_delta['Z'] | self.__target_ship_firing_heading_delta['PS']),
                self.__ship_thrust['NL']
            ),
            ctrl.Rule(
                (self.__closest_mine_remaining_time['L'] | self.__closest_mine_distance['out_of_range']) & (self.__greatest_threat_asteroid_threat_time['XS'] | self.__greatest_threat_asteroid_threat_time['S']) & (self.__target_ship_firing_heading_delta['NM'] | self.__target_ship_firing_heading_delta['PM']) & (self.__ship_speed['Z'] | self.__ship_speed['PS'] | self.__ship_speed['PM'] | self.__ship_speed['PL']),
                self.__ship_thrust['NL']
            ),
            ctrl.Rule(
                (self.__closest_mine_remaining_time['L'] | self.__closest_mine_distance['out_of_range']) & (self.__greatest_threat_asteroid_threat_time['XS'] | self.__greatest_threat_asteroid_threat_time['S']) & (self.__target_ship_firing_heading_delta['NM'] | self.__target_ship_firing_heading_delta['PM']) & (self.__ship_speed['NS'] | self.__ship_speed['NM'] | self.__ship_speed['NL']),
                self.__ship_thrust['PL']
            ),
            ctrl.Rule(
                (self.__closest_mine_remaining_time['L'] | self.__closest_mine_distance['out_of_range']) & self.__greatest_threat_asteroid_threat_time['M'] & (self.__target_ship_firing_heading_delta['NL'] | self.__target_ship_firing_heading_delta['PL']),
                self.__ship_thrust['PS']
            ),
            ctrl.Rule(
                (self.__closest_mine_remaining_time['L'] | self.__closest_mine_distance['out_of_range']) & self.__greatest_threat_asteroid_threat_time['M'] & (self.__target_ship_firing_heading_delta['NS'] | self.__target_ship_firing_heading_delta['Z'] | self.__target_ship_firing_heading_delta['PS']),
                self.__ship_thrust['NS']
            ),
            ctrl.Rule(
                (self.__closest_mine_remaining_time['L'] | self.__closest_mine_distance['out_of_range']) & self.__greatest_threat_asteroid_threat_time['M'] & (self.__target_ship_firing_heading_delta['NM'] | self.__target_ship_firing_heading_delta['PM']) & (self.__ship_speed['NS'] | self.__ship_speed['Z'] | self.__ship_speed['PS'] | self.__ship_speed['PM'] | self.__ship_speed['PL']),
                self.__ship_thrust['NS']
            ),
            ctrl.Rule(
                (self.__closest_mine_remaining_time['L'] | self.__closest_mine_distance['out_of_range']) & self.__greatest_threat_asteroid_threat_time['M'] & (self.__target_ship_firing_heading_delta['NM'] | self.__target_ship_firing_heading_delta['PM']) & (self.__ship_speed['NM'] | self.__ship_speed['NL']),
                self.__ship_thrust['PS']
            ),
            ctrl.Rule(
                (self.__closest_mine_remaining_time['L'] | self.__closest_mine_distance['out_of_range']) & self.__greatest_threat_asteroid_threat_time['L'],
                self.__ship_thrust['Z']
            ),
            ctrl.Rule(
                self.__closest_mine_distance['in_range'] & self.__closest_mine_remaining_time['M'],
                self.__ship_thrust['NM']
            ),
            ctrl.Rule(
                self.__closest_mine_distance['in_range'] & self.__closest_mine_remaining_time['S'],
                self.__ship_thrust['NL']
            )
        ]

    def __setup_simulations(self) -> None:
        self.__setup_fuzzy_rules()
        assert (self.__asteroid_select_fuzzy_rules is not None)
        assert (self.__ship_fire_fuzzy_rules is not None)
        assert (self.__drop_mine_fuzzy_rules is not None)
        assert (self.__ship_thrust_fuzzy_rules is not None)

        asteroid_select = ctrl.ControlSystem(self.__asteroid_select_fuzzy_rules)
        self.__asteroid_select_simulation = ctrl.ControlSystemSimulation(
            asteroid_select,
            cache=config.USE_SIMULATION_CACHE,
            flush_after_run=config.FLUSH_SIMULATION_CACHE_AFTER_RUN
        )

        ship_fire = ctrl.ControlSystem(self.__ship_fire_fuzzy_rules)
        self.__ship_fire_simulation = ctrl.ControlSystemSimulation(
            ship_fire,
            cache=config.USE_SIMULATION_CACHE,
            flush_after_run=config.FLUSH_SIMULATION_CACHE_AFTER_RUN
        )

        drop_mine = ctrl.ControlSystem(self.__drop_mine_fuzzy_rules)
        self.__drop_mine_simulation = ctrl.ControlSystemSimulation(
            drop_mine,
            cache=config.USE_SIMULATION_CACHE,
            flush_after_run=config.FLUSH_SIMULATION_CACHE_AFTER_RUN
        )

        ship_thrust = ctrl.ControlSystem(self.__ship_thrust_fuzzy_rules)
        self.__ship_thrust_simulation = ctrl.ControlSystemSimulation(
            ship_thrust,
            cache=config.USE_SIMULATION_CACHE,
            flush_after_run=config.FLUSH_SIMULATION_CACHE_AFTER_RUN
        )

    def actions(self, ship_state: Dict[str, Any], game_state: immutabledict[Any, Any]) -> Tuple[float, float, bool, bool]:
        """
        Method processed each time step by this controller.
        """
        assert (self.__asteroid_select_simulation is not None)
        assert (self.__ship_fire_simulation is not None)
        assert (self.__drop_mine_simulation is not None)
        assert (self.__ship_thrust_simulation is not None)

        game_map_size: tuple[int, int] = game_state["map_size"]
        self.__ship_distance_from_nearest_edge_range = (0, min(game_map_size)/2)
        max_map_distance: float = sqrt(game_map_size[0]**2 + game_map_size[1]**2)

        # TODO fix reason why ship doesn't drop mines before getting hit
        bullets: list[dict[str, Any]] = game_state["bullets"]
        asteroids: list[dict[str, Any]] = game_state["asteroids"]
        mines: list[dict[str, Any]] = game_state["mines"]

        asteroids = DiamondPickaxeController.__pop_doomed_asteroids(asteroids, bullets)

        if len(asteroids) == 0:
            # there are no more asteroids to target,
            # the last remaining asteroid(s) have bullets heading towards them
            thrust: float = 0
            turn_rate: float = 0
            fire: bool = False
            drop_mine: bool = False

            return thrust, turn_rate, fire, drop_mine

        # there is at least one asteroid to target:
        ship_is_respawning: bool = ship_state["is_respawning"]
        ship_lives_remaining: int = ship_state["lives_remaining"]
        ship_position: tuple[float, float] = ship_state["position"]
        ship_heading: float = radians(ship_state["heading"])
        ship_radius: int = ship_state["radius"]
        safety_margin_ship_radius: float = ship_radius * 1.5
        ship_velocity: tuple[float, float] = ship_state["velocity"]
        ship_speed: float = ship_state["speed"]
        stopping_time: float = abs(ship_speed / (self.__ship_thrust_range[0]-80)) # -80 is for the drag
        stopping_distance: float = (abs(ship_speed) * stopping_time) + (self.__ship_thrust_range[0] * (stopping_time**2) / 2)
        ship_distance_from_nearest_edge: float = self.__calculate_distance_to_closest_edge(ship_position, game_map_size)

        assert (stopping_distance >= 0)

        bullet_speed: float = 800

        closest_mine_index: None | int = self.__find_closest_mine(ship_position, mines)
        closest_mine_distance: float
        closest_mine_remaining_time: float

        if closest_mine_index is None:
            # there were no mines on the field
            closest_mine_distance = max_map_distance
            closest_mine_remaining_time = 100
        else:
            mine_position: tuple[float, float] = mines[closest_mine_index]["position"]
            closest_mine_distance = sqrt((ship_position[0] - mine_position[0])**2 + (ship_position[1] - mine_position[1])**2)
            closest_mine_remaining_time = mines[closest_mine_index]["remaining_time"]

        max_turn_angle_per_frame_radians: float = radians(self.__ship_turn_range[1]) / config.FRAME_RATE # (degrees / second) /  (frames / second) = degrees / frame
        best_next_frame_asteroid_index: int | None = self.__select_best_asteroid_in_angle_range(ship_position, bullet_speed, ship_heading, max_turn_angle_per_frame_radians, asteroids, game_map_size)
        if (best_next_frame_asteroid_index is None):
            best_next_frame_asteroid_index = self.__find_closest_asteroid(ship_position, asteroids)
        assert (best_next_frame_asteroid_index is not None) # the game should have ended if there are no more asteroids
        best_next_frame_asteroid: dict[str, Any] = asteroids[best_next_frame_asteroid_index]
        best_next_frame_asteroid_distance: float = sqrt((ship_position[0] - best_next_frame_asteroid["position"][0])**2 + (ship_position[1] - best_next_frame_asteroid["position"][1])**2)
        best_next_frame_asteroid_radius: float = best_next_frame_asteroid["radius"]
        best_next_frame_asteroid_size: int = best_next_frame_asteroid["size"]
        best_next_frame_asteroid_position: tuple[float, float] = best_next_frame_asteroid["position"]
        best_next_frame_asteroid_velocity: tuple[float, float] = best_next_frame_asteroid["velocity"]
        best_next_frame_asteroid_heading: float = atan2(best_next_frame_asteroid_velocity[1], best_next_frame_asteroid_velocity[0])

        greatest_threat_asteroid_index: None | int = self.__find_greatest_threat_asteroid(ship_position, ship_velocity, safety_margin_ship_radius, asteroids)
        greatest_threat_asteroid: dict[str, Any] | None = None
        greatest_threat_asteroid_threat_time: float = 100
        greatest_threat_asteroid_radius: float = 0
        greatest_threat_asteroid_size: int = 0
        greatest_threat_asteroid_position: tuple[float, float] = (0, 0)
        greatest_threat_asteroid_velocity: tuple[float, float] = (0, 0)
        greatest_threat_asteroid_heading: float = 0
        if (greatest_threat_asteroid_index is not None):
            greatest_threat_asteroid = asteroids[greatest_threat_asteroid_index]
            greatest_threat_asteroid_radius: float = greatest_threat_asteroid["radius"]
            greatest_threat_asteroid_size: int = greatest_threat_asteroid["size"]
            greatest_threat_asteroid_position: tuple[float, float] = greatest_threat_asteroid["position"]
            greatest_threat_asteroid_velocity: tuple[float, float] = greatest_threat_asteroid["velocity"]
            greatest_threat_asteroid_heading: float = atan2(best_next_frame_asteroid_velocity[1], best_next_frame_asteroid_velocity[0])
            asteroid_intercept = self.__calculate_intercept(
                ship_position,
                ship_velocity,
                safety_margin_ship_radius,
                greatest_threat_asteroid_position,
                greatest_threat_asteroid_velocity,
                greatest_threat_asteroid_radius
            )
            assert (asteroid_intercept is not None) # if it is None, something is wrong with self.__find_greatest_threat_asteroid()
            greatest_threat_asteroid_threat_time = asteroid_intercept[2]

        self.__asteroid_select_simulation.input['greatest_threat_asteroid_threat_time'] = min(greatest_threat_asteroid_threat_time, 100)
        self.__asteroid_select_simulation.input['greatest_threat_asteroid_size'] = greatest_threat_asteroid_size
        self.__asteroid_select_simulation.input['best_next_frame_asteroid_size'] = best_next_frame_asteroid_size
        self.__asteroid_select_simulation.compute()

        selected_asteroid_position: tuple[float, float]
        selected_asteroid_velocity: tuple[float, float]
        try:
            if (greatest_threat_asteroid is None or self.__asteroid_select_simulation.output['asteroid_selection'] < 0):
                selected_asteroid_position = best_next_frame_asteroid_position
                selected_asteroid_velocity = best_next_frame_asteroid_velocity
            else:
                selected_asteroid_position = greatest_threat_asteroid_position
                selected_asteroid_velocity = greatest_threat_asteroid_velocity
        except KeyError:
            #print("error in asteroid selection")
            selected_asteroid_position = best_next_frame_asteroid_position
            selected_asteroid_velocity = best_next_frame_asteroid_velocity

        target_ship_firing_heading: float | None = self.__calculate_bullet_intercept(ship_position, bullet_speed, selected_asteroid_position, selected_asteroid_velocity, game_map_size)

        # Lastly, find the difference betwwen firing angle and the ship's current orientation.
        target_ship_firing_heading_delta: float
        if target_ship_firing_heading is None:
            # this should never normally happen, as the selected asteroid should not be an impossible target,
            # but it is possible that there are no possible targets that can be hit within the game screen,
            # and in that case this may happen, and the direction and amount we want to turn the ship is unknown
            target_ship_firing_heading_delta = 0
        else:
            target_ship_firing_heading_delta = target_ship_firing_heading - ship_heading

        # Wrap all angles to (-pi, pi)
        target_ship_firing_heading_delta = (target_ship_firing_heading_delta + pi) % (2 * pi) - pi

        # Pass the inputs to the rulebase and fire it

        self.__drop_mine_simulation.input['greatest_threat_asteroid_threat_time'] = greatest_threat_asteroid_threat_time
        self.__drop_mine_simulation.input['ship_is_invincible'] = int(ship_is_respawning)
        self.__ship_thrust_simulation.input['greatest_threat_asteroid_threat_time'] = greatest_threat_asteroid_threat_time
        self.__ship_thrust_simulation.input['ship_speed'] = ship_speed
        self.__ship_thrust_simulation.input['closest_mine_distance'] = closest_mine_distance
        self.__ship_thrust_simulation.input['closest_mine_remaining_time'] = closest_mine_remaining_time
        self.__ship_thrust_simulation.input['target_ship_firing_heading_delta'] = target_ship_firing_heading_delta
        self.__ship_fire_simulation.input['ship_is_invincible'] = 1 if ship_is_respawning else -1
        self.__ship_fire_simulation.input['greatest_threat_asteroid_threat_time'] = min(greatest_threat_asteroid_threat_time, 100)

        ideal_turn_rate: float = degrees(target_ship_firing_heading_delta) * config.FRAME_RATE
        turn_rate: float = min(max(self.__ship_turn_range[0], ideal_turn_rate), self.__ship_turn_range[1])

        fire: bool
        if config.CONSERVE_BULLETS:
            if int(ideal_turn_rate*1000000) == int(turn_rate*1000000):
                # if they are close enough to a few decimal places, then we have turned as much as we wanted to,
                # and are aiming exactly where we need to to hit an asteroid
                fire = True
            else:
                # we are highly unlikely to be able to hit an asteroid if we shoot now, so we don't shoot
                fire = False
        else:
            # always fire, no need to conserve bullets
            fire = True

        if fire: # if we aren't planning to fire, no need to waste resources computing fuzzy simulation
            try:
                self.__ship_fire_simulation.compute()
                if self.__ship_fire_simulation.output['ship_fire'] < 0:
                    fire = False
            except (ValueError, KeyError):
                # default to not firing if an error occurred
                fire = False

        drop_mine: bool
        try:
            self.__drop_mine_simulation.compute()
            if self.__drop_mine_simulation.output['drop_mine'] >= 0:
                drop_mine = True
            else:
                drop_mine = False
        except (ValueError, KeyError):
            drop_mine = False

        thrust: float
        try:
            self.__ship_thrust_simulation.compute()
            thrust = self.__ship_thrust_simulation.output['ship_thrust']
        except (ValueError, KeyError):
            thrust = 0

        self.__current_frame +=1

        return thrust, turn_rate, fire, drop_mine

    @staticmethod
    def __pop_doomed_asteroids(
        asteroids: list[dict[str, Any]],
        bullets: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        returns the list of asteroids in the same format as provided,
        except that all asteroids that will be hit by bullets at some point in the future
        are popped from that list
        """
        for bullet in bullets:
            bullet_position: tuple[float, float] = bullet["position"]
            bullet_velocity: tuple[float, float] = bullet["velocity"]
            bullet_radius: float = 0
            intercept_asteroid_index: int | None = None
            minimum_intercept_time: float | None = None

            for asteroid_index, asteroid in enumerate(asteroids):
                asteroid_position: tuple[float, float] = asteroid["position"]
                asteroid_velocity: tuple[float, float] = asteroid["velocity"]
                asteroid_radius: float = asteroid["radius"]
                asteroid_size: int = asteroid["size"]

                output = DiamondPickaxeController.__calculate_intercept(
                    bullet_position,
                    bullet_velocity,
                    bullet_radius,
                    asteroid_position,
                    asteroid_velocity,
                    asteroid_radius
                )
                if output is None:
                    continue # no intercept

                if asteroid_size != 1:
                    # asteroid will split into multiple when hit, so we cannot just pop it from the list
                    asteroids[asteroid_index]["size"] -= 1 # subtract 1 from the size in case another bullet also hits this asteroid
                    continue

                intercept_position: tuple[float, float] = output[0]
                intercept_time: float = output[2]
                if (minimum_intercept_time is None or intercept_time < minimum_intercept_time):
                    intercept_asteroid_index = asteroid_index
                    minimum_intercept_time = intercept_time

            if intercept_asteroid_index is None:
                continue # no asteroid will be hit by this bullet
            # the asteroid that will be hit by this bullet has been found
            asteroids.pop(intercept_asteroid_index)

        return asteroids

    @staticmethod
    def __calculate_bullet_intercept(
        ship_position: tuple[float, float],
        bullet_speed: float,
        asteroid_position: tuple[float, float],
        asteroid_velocity: tuple[float, float],
        map_size: tuple[int, int]
    ) -> float | None:
        """
        returns the target ship firing heading (radians),
        returns None if it is impossible for the ship to shoot the asteroid within
        the borders of the screen, or at all
        """
        position_delta: tuple[float, float] = (ship_position[0] - asteroid_position[0], ship_position[1] - asteroid_position[1])
        distance: float = sqrt(position_delta[0]**2 + position_delta[1]**2)
        angle_delta: float = atan2(position_delta[1], position_delta[0])
        heading_2: float = atan2(asteroid_velocity[1], asteroid_velocity[0])

        my_theta2: float = angle_delta - heading_2
        cos_my_theta2: float = cos(my_theta2)
        speed_1: float = bullet_speed
        speed_2: float = sqrt(asteroid_velocity[0]**2 + asteroid_velocity[1]**2)

        if speed_1 == speed_2:
            # this would result in a division by 0 error in the normal method of solving the equations
            # FIXME need a different way to find intercept when the speeds are exactly the same
            return None

        # Determinant of the quadratic formula b^2-4ac
        targ_det: float = (2 * distance * speed_2 * cos_my_theta2)**2 - (4*(speed_2**2 - speed_1**2) * distance**2)

        assert (targ_det >= 0)

        # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
        intrcpt1: float = ((2 * distance * speed_2 * cos_my_theta2) + sqrt(targ_det)) / (2 * (speed_2**2 - speed_1**2))
        intrcpt2: float = ((2 * distance * speed_2 * cos_my_theta2) - sqrt(targ_det)) / (2 * (speed_2**2 - speed_1**2))

        if (intrcpt1 < 0 and intrcpt2 < 0):
            # it is impossible to hit the asteroid, there are no intercepts with positive time
            return None

        # Take the smaller intercept time, as long as it is positive; if not, take the larger one.
        time: float
        if intrcpt1 > intrcpt2:
            if intrcpt2 >= 0:
                time = intrcpt2
            else:
                time = intrcpt1
        else:
            if intrcpt1 >= 0:
                time = intrcpt1
            else:
                time = intrcpt2

        intercept: tuple[float, float] = (
            asteroid_position[0] + asteroid_velocity[0] * (time+1/30),
            asteroid_position[1] + asteroid_velocity[1] * (time+1/30)
        )

        if (intercept[0] <= 0 or intercept[1] <= 0 or intercept[0] >= map_size[0] or intercept[1] >= map_size[1]):
            # bullet would intercept asteroid outside of the game map bounds, so it would not hit due to game mechanics
            return None

        target_ship_firing_heading: float = atan2((intercept[1] - ship_position[1]), (intercept[0] - ship_position[0]))

        return target_ship_firing_heading

    @staticmethod
    def __calculate_intercept(
        position_1: tuple[float, float],
        velocity_1: tuple[float, float],
        radius_1: float,
        position_2: tuple[float, float],
        velocity_2: tuple[float, float],
        radius_2: float
    ) -> tuple[tuple[float, float], tuple[float, float], float] | None:
        """
        calculates the intercept between two objects traveling at constant velocity,
        returns the time and positions of the intercept ((x1, y1), (x2, y2), t), or None if they do not collide.
        the positions are different due to the radius of the objects
        """
        position_delta: tuple[float, float] = (position_1[0] - position_2[0], position_1[1] - position_2[1])
        velocity_delta: tuple[float, float] = (velocity_1[0] - velocity_2[0], velocity_1[1] - velocity_2[1])

        distance: float = sqrt(position_delta[0] ** 2 + position_delta[1] ** 2)
        if (distance <= radius_1 + radius_2):
            # the two objects are currently inside of each other and currently colliding
            return (position_1, position_2, 0)

        # the distance between the centers of the objects as a function of time t is given by:
        # d(t) = sqrt( (delta_x + delta_v_x * t)**2 + (delta_y + delta_v_y * t)**2 )
        # collision if d(t) <= radius_1 + radius_2
        # square both sides of equation, rearrange terms to group by A*t^2 + B*t + C <= 0

        a: float = velocity_delta[0]**2 + velocity_delta[1]**2
        b: float = 2 * position_delta[0] * velocity_delta[0] + 2 * position_delta[1] * velocity_delta[1]
        c: float = position_delta[0]**2 + position_delta[1]**2 - (radius_1 + radius_2)**2

        # solve quadratic equation
        determinant: float = b**2 - 4*a*c

        if (determinant < 0):
            # the objects never collide!
            return None

        if (a == 0):
            # the object never collide! (this avoids division by 0)
            return None

        collision_time_candidate_1: float = (-b + sqrt(determinant)) / (2 * a)
        collision_time_candidate_2: float = (-b - sqrt(determinant)) / (2 * a)

        # the real collision time is the smallest positive one of the two
        collision_time: float = -1
        if collision_time_candidate_1 < 0:
            if collision_time_candidate_2 < 0:
                # both collision times are negative, so the objects don't collide in the future
                return None
            collision_time = collision_time_candidate_2
        elif collision_time_candidate_2 < 0:
            if collision_time_candidate_1 < 0:
                # both collision times are negative, so the objects don't collide in the future
                return None
            collision_time = collision_time_candidate_1
        else:
            collision_time = min(collision_time_candidate_1, collision_time_candidate_2)

        assert (collision_time >= 0)

        # find position
        collision_position_1: tuple[float, float] = (
            position_1[0] + velocity_1[0] * collision_time,
            position_1[1] + velocity_1[1] * collision_time
        )
        collision_position_2: tuple[float, float] = (
            position_2[0] + velocity_2[0] * collision_time,
            position_2[1] + velocity_2[1] * collision_time
        )

        return (collision_position_1, collision_position_2, collision_time)

    @staticmethod
    def __find_closest_mine(
        ship_position: tuple[float, float],
        mines: list[dict[str, Any]]
    ) -> int | None:
        """
        returns the index of the closest mine,
        returns None if there are no mines
        """
        closest_mine_distance: None | float = None
        closest_mine_index: None | int = None
        for index, mine in enumerate(mines):
            mine_position: tuple[float, float] = mine["position"]
            mine_distance: float = sqrt((ship_position[0] - mine_position[0])**2 + (ship_position[1] - mine_position[1])**2)

            if closest_mine_distance is None or closest_mine_distance > mine_distance:
                closest_mine_distance = mine_distance
                closest_mine_index = index

        assert ((closest_mine_distance is None) == (closest_mine_index is None))

        return closest_mine_index

    @staticmethod
    def __find_closest_asteroid(
        ship_position: tuple[float, float],
        asteroids: list[dict[str, Any]]
    ) -> int | None:
        """
        returns the index of the closest asteroid,
        returns None if there are no asteroids
        """
        closest_asteroid_distance: None | float = None
        closest_asteroid_index: None | int = None
        for index, asteroid in enumerate(asteroids):
            asteroid_position: tuple[float, float] = asteroid["position"]
            asteroid_distance: float = sqrt((ship_position[0] - asteroid_position[0])**2 + (ship_position[1] - asteroid_position[1])**2)

            if closest_asteroid_distance is None or closest_asteroid_distance > asteroid_distance:
                closest_asteroid_distance = asteroid_distance
                closest_asteroid_index = index

        assert ((closest_asteroid_distance is None) == (closest_asteroid_index is None))

        return closest_asteroid_index

    @staticmethod
    def __find_asteroids_in_angle_range(
        ship_position: tuple[float, float],
        bullet_speed: float,
        ship_heading_radians: float,
        max_heading_change_per_frame_radians: float,
        asteroids: list[dict[str, Any]],
        map_size: tuple[int, int]
    ) -> list[int]:
        """
        returns the list of asteroids that can possibly be targeted in the next frame,
        obviously the list will be empty if there is no such asteroid
        """
        # two lines are drawn from the ship's position, representing the extremeties of headings it can aim at
        # on the next frame.
        # asteroids must be between these two lines, and they must be in the direction that the ship is facing as well,
        # (not behind it)
        assert (ship_heading_radians >= 0)
        assert (ship_heading_radians <= 2*pi)

        asteroids_in_angle_range: list[int] = []
        for index, asteroid in enumerate(asteroids):
            asteroid_position: tuple[float, float] = asteroid["position"]
            asteroid_velocity: tuple[float, float] = asteroid["velocity"]
            ship_firing_angle: float | None = DiamondPickaxeController.__calculate_bullet_intercept(
                ship_position,
                bullet_speed,
                asteroid_position,
                asteroid_velocity,
                map_size
            )

            if ship_firing_angle is not None:
                heading_delta: float = ship_firing_angle - ship_heading_radians
                heading_delta = (heading_delta + pi) % (2 * pi) - pi # wrap angles to between -pi and pi
                if abs(heading_delta) < max_heading_change_per_frame_radians:
                    asteroids_in_angle_range.append(index)

        return asteroids_in_angle_range

    @staticmethod
    def __select_best_asteroid_in_angle_range(
        ship_position: tuple[float, float],
        bullet_speed: float,
        ship_heading_radians: float,
        max_heading_change_per_frame_radians: float,
        asteroids: list[dict[str, Any]],
        map_size: tuple[int, int]
    ) -> int | None:
        """
        returns the index of the best asteroid to target in the given angle range,
        returns None if there are no asteroids in the angle range.

        Selects the asteroid with the smallest size and least angle to turn the ship
        """
        asteroids_in_angle_range_indices: list[int] = DiamondPickaxeController.__find_asteroids_in_angle_range(
            ship_position,
            bullet_speed,
            ship_heading_radians,
            max_heading_change_per_frame_radians,
            asteroids,
            map_size
        )
        best_asteroid_index: int | None = None
        best_asteroid_size: int | None = None
        best_asteroid_angle_delta: float | None = None
        for index in asteroids_in_angle_range_indices:
            asteroid: dict[str, Any] = asteroids[index]
            asteroid_size: int = asteroid["size"]
            asteroid_position: tuple[float, float] = asteroid["position"]
            asteroid_velocity: tuple[float, float] = asteroid["velocity"]
            ship_firing_angle: float | None = DiamondPickaxeController.__calculate_bullet_intercept(
                ship_position,
                bullet_speed,
                asteroid_position,
                asteroid_velocity,
                map_size
            )

            if ship_firing_angle is not None:
                heading_delta: float = ship_firing_angle - ship_heading_radians
                heading_delta = (heading_delta + pi) % (2 * pi) - pi # wrap angles to between -pi and pi
                heading_delta = abs(heading_delta)

                if (best_asteroid_index is None):
                    best_asteroid_index = index
                    best_asteroid_size = asteroid_size
                    best_asteroid_angle_delta = heading_delta
                    continue

                assert (best_asteroid_index is not None)
                assert (best_asteroid_size is not None)
                assert (best_asteroid_angle_delta is not None)

                if asteroid_size > best_asteroid_size:
                    best_asteroid_index = index
                    best_asteroid_size = asteroid_size
                    best_asteroid_angle_delta = heading_delta
                if asteroid_size == best_asteroid_angle_delta and heading_delta < best_asteroid_angle_delta:
                    best_asteroid_index = index
                    best_asteroid_size = asteroid_size
                    best_asteroid_angle_delta = heading_delta

        return best_asteroid_index

    @staticmethod
    def __find_greatest_threat_asteroid(
        ship_position: tuple[float, float],
        ship_velocity: tuple[float, float],
        ship_radius: float,
        asteroids: list[dict[str, Any]]
    ) -> int | None:
        """
        returns the index of the greatest threat asteroid (one that would hit the ship soonest),
        returns None if there are no asteroids that pose any threat
        """
        greatest_threat_asteroid_threat_time: None | float = None
        greatest_threat_asteroid_index: None | int = None
        for index, asteroid in enumerate(asteroids):
            asteroid_radius: float = asteroid["radius"]
            asteroid_position: tuple[float, float] = asteroid["position"]
            asteroid_velocity: tuple[float, float] = asteroid["velocity"]

            asteroid_intercept = DiamondPickaxeController.__calculate_intercept(
                ship_position,
                ship_velocity,
                ship_radius,
                asteroid_position,
                asteroid_velocity,
                asteroid_radius
            )
            asteroid_will_intercept: bool = asteroid_intercept is not None
            if (asteroid_will_intercept):
                asteroid_threat_time: float = asteroid_intercept[2]
                if greatest_threat_asteroid_threat_time is None or greatest_threat_asteroid_threat_time > asteroid_threat_time:
                    greatest_threat_asteroid_threat_time = asteroid_threat_time
                    greatest_threat_asteroid_index = index

        assert ((greatest_threat_asteroid_threat_time is None) == (greatest_threat_asteroid_index is None))

        return greatest_threat_asteroid_index

    @staticmethod
    def __calculate_distance_to_closest_edge(
        ship_position: tuple[float, float],
        map_size: tuple[int, int]
    ) -> float:
        """
        returns the shortest distance between the ship and the nearest edge
        """
        shortest_distance: float = min(ship_position[0], ship_position[1], map_size[0] - ship_position[0], map_size[1] - ship_position[1])

        return shortest_distance

    @property
    def name(self) -> str:
        return self.__name
