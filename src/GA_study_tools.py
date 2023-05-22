import os, multiprocessing
from TSP_instance import TSP_instance
from Observer import Observer,GA_TSP_info
from configparser import ConfigParser
from Custom_functions import *

config = ConfigParser()
config.read("src/Configuracion.conf")
config_name = "CONFIGURACION"
# PLOTS = config.get(config_name,"plots")
GA_SOLUTIONS = config.get(config_name,"GA_solutions")
SUMMARIES = config.get(config_name,"summaries")


# FUNCIONES PARA HACER ESTUDIOS COMPUTACIONALES DE GAS

def run_and_save_information(nombre_TSP: str,
                             info: GA_TSP_info,
                             observador: Observer,
                             repeats: int,
                             name_of_study: str,
                             save_sumary:bool,
                             save_solutions:bool,
                             use_seed: bool,
                             header,
                             i):
    """Ejecuta un GA concreto sobre una instancia concreta del TSP y guarda los resultados en un .csv"""

   

    #El objeto informacion es utilizado por el observador para crear el GA
    # y ejecutarlo
    instancia = TSP_instance(problem_name=nombre_TSP)
    observador.run(info=info,
                   instance=instancia,
                   repeats=repeats,
                   use_seed=use_seed)
    #Guardo la información en un csv
    if save_sumary:
        file_path = os.path.join(SUMMARIES,f"{name_of_study}_{i}.csv")
        observador.save_summary_to_csv(file_path=file_path,fields=header)
    if save_solutions:
        directory = os.path.join(GA_SOLUTIONS,name_of_study)
        observador.save_solutions_to_csv(directory=directory)
    observador.reset_data()



def perform_full_study(info: GA_TSP_info,
                    instance_names: list,
                    name_of_study: str,
                    header: list,
                    repeats: int = 5,
                    use_seed: bool = True,
                    save_summary: bool = True,
                    save_solutions: bool = False,
                    num_processes: int = 4):
    """
    Ejecuta el GA dado por los parámetros del objeto información sobre todas las instancias dadas.
    Para ello se realizan varios procesos al mismo tiempo.
    """

    #Ejecuto un solo GA sobre todas las instancias tomándolas de 4 en 4, creando 4 procesos que funcionan de forma simultánea.
    #Cada proceso tendrá un observador distinto
    observadores = []
    for _ in range(num_processes):
        observadores.append(Observer())
    #Mientras queden instancias, se siguen ejecutando GAs
    while len(instance_names) > 1:
        processes = []
        for i in range(num_processes):
            if len(instance_names) != 0:
                #Tomo la siguiente instancia y ejecuto un GA creando un proceso
                problem_name = instance_names.pop(0)
                single_process = multiprocessing.Process(
                                                        target=run_and_save_information,
                                                        args=(problem_name,info,observadores[i],repeats,name_of_study,save_summary,save_solutions,use_seed,header,i)
                                                        )
                single_process.start()
                processes.append(single_process)
        #Hago que el programa espere a que finalicen todos los procesos creados.
        for process in processes:
            process.join()

    study_csv_path = os.path.join(SUMMARIES,f"{name_of_study}.csv")
    join_information(study_csv_path)