from Custom_functions import get_instance_names
import os
from Observer import GA_TSP_info
import AMPL_tools, GA_study_tools
from configparser import ConfigParser

config = ConfigParser()
config.read("src/Configuracion.conf")
config_name = "CONFIGURACION"
GA_SOLUTIONS = config.get(config_name,"GA_solutions")
SUMMARIES = config.get(config_name,"summaries")


if __name__ == "__main__":
    info = GA_TSP_info(encoding="permutational",
                       mutation_operator="sim-dynamic",
                       crossover_operator="OX",
                       selection_operator="tournament")
    info.max_running_time = 3600
    info.max_saturation = 100

    info.alpha = 0.15
    info.population_size = 2000
    info.number_elites = 20
    info.crossover_probability = 1
    info.mutation_probability = 0.6
    info.selection_parameter = 25

    header= ["problem_name",
             "problem_dimension",
                "encoding",
                "selection_operator",
                "crossover_operator",
                "mutation_operator",
                "population_size",
                "number_elites",
                "crossover_probability",
                "mutation_probability",
                "alpha",
                "selection_parameter",
                "average_score",
                "best_score",
                "worst_score",
                "average_running_time",
                "average_generations_completed"]

    tsp_names = get_instance_names()
    GA_study_tools.perform_full_study(info=info,
                                instance_names=tsp_names,
                                header=header,
                                name_of_study="Estudio2_GA",
                                repeats=5,
                                save_summary=True,
                                save_solutions=True,
                                use_seed=False
                                )



    # tsp_names = ["fri26","gr24","bayg29"]
    # csv_path = os.path.join(SUMMARIES,"Estudio_AMPL.csv")
    # AMPL_tools.AMPL_study(problem_names=tsp_names,csv_path=csv_path)