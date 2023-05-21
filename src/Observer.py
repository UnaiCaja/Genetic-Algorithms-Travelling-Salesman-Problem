#Contiene una clase para almacenar informacion de las ejecuciones y una clase Observador
#encargada de generar instancias de GA_for_TSP y ejecutarlas sobre instancias del TSP

import  time, numpy as np, os, csv, copy
from GA_for_TSP import GA_for_TSP
from TSP_instance import TSP_instance
from dataclasses import dataclass, field, fields
from functools import total_ordering



@dataclass
class GA_TSP_info:
    """
    Clase encargada de guardar información de un GA. Esto incluye tanto sus parámetros como los resultados al ejecutarlo sobre instancias concretas.

    problem_name: Nombre de la instancia del TSP sobre la que se ejecuta el GA.
    problem_dimension: Dimensión de la instancia (número de ciudades)
    problem_optimal_value: Valor óptimo de la instancia del TSP si se conoce
    problem_optimal_solution: Solución óptima de la instancia del TSP si se conoce
    encoding: Codificación empleada para el GA. Puede ser "permutational" o "ordinal".
    alpha_adaptive: Parámetro empleado en la mutación adaptativa (si es que se usa ese mecanismo).
    selection_parameter: En caso de usarse selección por torneo sería el número de individuos por grupo.
    max_running_time: Tiempo máximo que se permite correr al GA
    max_saturation: Número máximo de generaciones sin mejorar la mejor solución.
    scores: Si se conoce la solución óptima de la instancia, será el incremento porcentual de la obtenida por el GA respecto de la exacta. De no conocerse la óptima, será sencillamente el mejor valor objetivo encontrado.
    average_score: Cuando el GA se ejecuta varias veces, se guarda el score medio.
    best_score: Cuando el GA se ejecuta varias veces, se guarda el mejor score conseguido.
    worst_score: Cuando el GA se ejecuta varias veces, se guarda el peor score conseguido.
    average_running_time: Cuando el GA se ejecuta varias veces, guarda el tiempo medio que requirió el GA para terminar su ejecución.
    average_generations_completed: Cuando el GA se ejecuta varias veces, guarda el número medio de generaciones completadas.
    best_solution_found: Cuando el GA se ejecuta varias veces, guarda la mejor solución encontrada.
    solutions: Guarda todas las soluciones que se han obtenido en las ejecuciones de un GA sobre una misma instancia.
    """
    #Informacion sobre la instancia del problema que resuelve el GA
    problem_name: str = field(init=False,default=None)
    problem_dimension: int = field(init=False,repr=False,compare=False,default=None)
    problem_optimal_value: float = field(init=False,repr=False,default=None,compare=False)
    problem_optimal_solution: list = field(init=False,repr=False,default=None,compare=False)

    #Codificacion y operadores del GA
    encoding: str = field(default="permutational")
    selection_operator: str = field(default="tournament")
    crossover_operator: str = field(default="OX")
    mutation_operator: str = field(default="swap")
    initial_population_method: str = field(default="randomized_nn")

    #Parametros del GA
    population_size: int = field(init=False,repr=False,default=None)
    number_elites: int = field(init=False,repr=False,default=None)
    mutation_probability: float = field(init=False,repr=False,default=None)
    alpha: float = field(init=False,repr=False,default=None)
    crossover_probability: float = field(init=False,repr=False,default=None)
    selection_parameter: int = field(init=False,repr=False,default=None)
    max_running_time: float = field(init=False,repr=False,default=3600.0)
    max_saturation: float = field(init=False,repr=False,default=30)

    #Resultados de aplicar el GA a la instancia
    scores: list = field(init=False,repr=False,compare=False,default=None)
    average_score: float = field(init=False,repr=False,compare=False,default=None)
    best_score: float = field(init=False,compare=False,default=None)
    worst_score: float = field(init=False,repr=False,compare=False,default=None)
    average_running_time: float = field(init=False,repr=False,compare=False,default=None)
    average_generations_completed: float = field(init=False,repr=False,compare=False,default=None)
    best_solution_found: list = field(init=False,repr=False,compare=False,default=None)
    solutions: list = field(init=False,repr=False,compare=False,default=None)


    #Las dos siguientes funciones permiten emplear la clase como si fuera un diccionario
    def __getitem__(self, item):
        return getattr(self, item)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)

    @total_ordering
    def __lt__(self, other):
        return self.average_score < other.average_score
    
    #Permite iterar sobre la clase, retornando los nombres de los atributos.
    def __iter__(self):
        return (getattr(self, field.name) for field in fields(self))
    
    
    
    
    
        



class Observer:
    """
    Clase encargada de organizar la ejecución de GAs sobre instancias distintas y de guardar la información obtenida.
    """

    def __init__(self) -> None:
        self.data = []
    

    @staticmethod
    def info_to_GA(info: GA_TSP_info) -> GA_for_TSP:
        """Transformar en un GA_for_TSP con los parámetros especificados."""

        if info.selection_operator == "tournament":
            GA = GA_for_TSP(
                population_size=info.population_size,
                num_elites=info.number_elites,
                encoding=info.encoding,
                selection_method=info.selection_operator,
                tau_tournament=info.selection_parameter,
                crossover_operator=info.crossover_operator,
                mutation_operator=info.mutation_operator,
                mutation_probability=info.mutation_probability,
                max_time=info.max_running_time,
                crossover_probability=info.crossover_probability,
                max_saturation=info.max_saturation,
                alpha=info.alpha,
                initial_population_method=info.initial_population_method
                        )
        elif info.selection_operator == "rank":
            GA = GA_for_TSP(
                population_size=info.population_size,
                num_elites=info.number_elites,
                encoding=info.encoding,
                selection_method=info.selection_operator,
                tau_tournament=None,
                crossover_operator=info.crossover_operator,
                mutation_operator=info.mutation_operator,
                mutation_probability=info.mutation_probability,
                max_time=info.max_running_time,
                crossover_probability=info.crossover_probability,
                max_saturation=info.max_saturation,
                alpha=info.alpha,
                initial_population_method=info.initial_population_method
                        )
        return GA
    


    def run(self, info: GA_TSP_info, instance: TSP_instance, repeats: int = 5,use_seed = False):
        """Ejecuta un GA sobre una instancia del TSP el número especificado de veces y guarda la información que falta en el objeto info."""

        obj_values = np.zeros(repeats)
        solutions = [None]*repeats
        times = np.zeros(repeats)
        generations_completed = np.zeros(repeats)

        GA = Observer.info_to_GA(info=info)

        #Se ejecuta el algoritmo tantas veces como se pide y se guarda la informacion
        for i in range(repeats):
            if use_seed == True:
                GA.set_seed(i)

            start = time.time()
            GA.run_GA(instance)
            end = time.time()

            times[i] = end-start
            solutions[i] = GA.best_solution_found
            obj_values[i] = GA.best_obj_val_found#Es la distancia no su inverso
            generations_completed[i] = GA.generations_completed
        
        #Se calcula el mejor/peor valor objetivo: la minima/maxima distancia
        ind_max = np.argmax(obj_values)
        max_value = obj_values[ind_max]
        ind_min = np.argmin(obj_values)
        min_value = obj_values[ind_min]

        #Ahora guardamos la información que falta en info
        info.problem_name=instance.name
        info.problem_dimension=instance.dimension
        info.average_running_time = np.round(np.average(times),decimals=2)
        info.best_solution_found = solutions[ind_min]
        info.solutions = solutions
        info.average_generations_completed = np.average(generations_completed)

        if instance.has_optimal_solution:
            info.problem_optimal_solution = np.round(instance.optimal_solution,decimals=2)
            info.problem_optimal_value = np.round(instance.optimal_value,decimals=2)
            info.scores = np.round((obj_values-instance.optimal_value)/instance.optimal_value*100,decimals=2)
            info.best_score = np.round((min_value-instance.optimal_value)/instance.optimal_value*100,decimals=2)
            info.worst_score = np.round((max_value-instance.optimal_value)/instance.optimal_value*100,decimals=2)
            info.average_score = np.round((np.average(obj_values)-instance.optimal_value)/instance.optimal_value*100,decimals=2)
        else:
            info.scores = np.round(obj_values,decimals=2)
            info.best_score = np.round(min_value,decimals=2)
            info.worst_score = np.round(max_value,decimals=2)
            info.average_score = np.round(np.average(obj_values),decimals=2)

        self.data.append(info)

    
    def run_parameter_set(self,
                        info: GA_TSP_info,
                        population_sizes:list,
                        numbers_elites:list,
                        mutation_probabilities:list,
                        crossover_probabilities: list,
                        selection_parameters:list,
                        instance:TSP_instance,
                        repeats:int = 5
                        ):
        """
        Llama iterativamente al la función run hasta ejecutar el GA sobre el conjunto completo de parámetros especificado. Los resultados se guardan en un array.
        """

        for pop_size in population_sizes:
            for num_elites in numbers_elites:
                for mut_prob in mutation_probabilities:
                    for selec_param in selection_parameters:
                        for cros_prob in crossover_probabilities:
                            new_info = copy.copy(info)
                            new_info.population_size = pop_size
                            new_info.number_elites = num_elites
                            new_info.mutation_probability = mut_prob
                            new_info.selection_parameter = selec_param
                            new_info.crossover_probability = cros_prob
                            self.run(info=new_info,instance=instance,repeats=repeats)
    

                

                
    def save_summary_to_csv(self,file_path: str,fields: list[str] = None):
        """
        Guarda toda la información especificada en un .csv

        file_name: Nombre que se quiere que tenga el fichero sin .csv
        fields: Atributos de los objetos información que se quieren guardar
        """

        #Por defecto se asigna a la información que es más común guardar
        if fields is None:
            fields = ["problem_name",
                      "problem_dimension",
                        "encoding",
                        "selection_operator",
                        "crossover_operator",
                        "mutation_operator",
                        "population_size",
                        "number_elites",
                        "crossover_probability",
                        "mutation_probability",
                        "selection_parameter",
                        "average_score",
                        "best_score",
                        "worst_score",
                        "average_running_time",
                        "average_generations_completed"]

        file_exists = os.path.isfile(file_path)
        with open(file_path,"a") as csv_file:
            writer = csv.writer(csv_file,delimiter=";")
            if not file_exists:
                writer.writerow(fields)#Cabecera
            #Para cada objeto información creamos una nueva fila para añadir
            for data_item in self.data:
                row = []
                for key in fields:
                    row.append(data_item[key])
                writer.writerow(row)


    def save_solutions_to_csv(self, directory: str):
        """Guarda las soluciones encontradas por un algoritmo genético en la carpeta especificada."""

        #Si no existe el directorio, lo creamos
        if not os.path.isdir(directory):
            os.mkdir(directory)
        #Se guardan las soluciones de cada objeto informacion en un fichero con el nombre correspondiente
        for info in self.data:
            out_path = os.path.join(directory,f"{info.problem_name}_sol_GA.txt")
            np.savetxt(fname=out_path,X=info.solutions, delimiter=";", fmt = "%i")

    def reset_data(self):
        del self.data
        self.data = []
            
                    
                
                


            
    










