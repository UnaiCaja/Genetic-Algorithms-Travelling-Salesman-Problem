#Clase que representa un GA para resolver el TSP
import random, pygad , numpy as np, time
from Custom_functions import *
from TSP_instance import TSP_instance


class GA_for_TSP:
    """
    Representa un algoritmo genético que se aplicará al TSP. \n
    Adapta la clase pygad.ga al contexto concreto del TSP especificando los operadores de mutación o cruce a utilizar.\n
    Además, añade nuevas funcionalidades como la de detener el GA después de cierto tiempo de ejecución.
    """

    #Numero de generaciones tras las que se informa de la longitud de la solucion encontrada
    generations_to_inform = 100
        

    def get_offspring_fitness(self,ga_algorithm: pygad.GA) :
        """Obtener aptitudes de los cromosomas justo después de realizar el cruce. """

        #Uso el atributo de número de generación para controlar si las aptitudes que se tienen
        #guardadas están actualizadas.
        if self.current_generation == ga_algorithm.generations_completed:
            return self.offspring_fitness
        else:
            #Si se ha cambiado de generación, debemos calcular nuevamente las aptitudes.
            self.current_generation == ga_algorithm.generations_completed
            offspring = ga_algorithm.last_generation_offspring_crossover
            num_offspring = len(offspring)
            fitness = np.zeros(num_offspring)
            for i,chromosome in enumerate(offspring):
                fitness[i] = ga_algorithm.fitness_func(chromosome,i)
            
            self.offspring_fitness = fitness
            return self.offspring_fitness
    
    def get_average_offspring_fitness(self,ga_algorithm: pygad.GA) :
        """Obtener la aptitud media de los cromosomas justo después de realizar el cruce. """

        if self.current_generation == ga_algorithm.generations_completed:
            return self.average_offspring_fitness
        else:
            offspring_fitness = self.get_offspring_fitness(ga_algorithm)
            self.average_offspring_fitness = np.average(offspring_fitness)
            return self.average_offspring_fitness



    def __init__(self,
                mutation_probability: float,
                crossover_probability: float,
                population_size: int,
                num_elites: int,
                encoding: str,
                mutation_operator: str,
                crossover_operator: str,
                selection_method: str = "tournament",
                initial_population_method: str = "random",
                alpha: float = 1.0,
                max_saturation: int = 30,
                max_time = 3600.0,
                num_generations: int = 10**9,
                tau_tournament: int = 3,
                seed: int = None,
                log_info: bool = False,
                save_solutions: bool = False
                ) -> None:
        """
        Constructor de la clase. Acepta todos los parámetros necesarios para ejecutar un GA.\n

        mutation_probability: Probabilidad de que se realice al menos una mutación.
        crossover_probability: Dados dos cromosomas a cruzar. Probabilidad de realizar el cruce respecto de dejarlos como están.
        population_size: Tamaño de la población.
        num_elites: Número de élites.
        encoding: Codificación de los tours. Puede ser "permutational" o "ordinal".
        mutation_operator: Operador de mutación utilizado. Para representación permutacional puede ser "swap" o "sim"
        crossover_operator: Operador de cruce. Para representación permutacional puede ser "OX"
        selection_method: Método de selección (por defecto "tournament"). En pygad hay otros implementados.
        alpha: Parámetro alpha que se utiliza en la mutación adaptativa y en la mutación dinámica.
        max_saturation: Número de generaciones que permitimos que pasen hasta detener el algoritmo (por defecto 30).
        max_time: Tiempo máximo en segundos que se le da al GA para ejecutar (por defecto 3600s = 1h).
        num_generations: Número máximo de generaciones que se permite realizar al GA (por defecto 10000 para que no se alcance).
        tau_tournament: Tamaño de los grupos que se toman para la selección de torneo.
        seed: Semilla para la generación de números aleatorios. De este modo pueden reproducirse los resultados.
        log_info: Indica si queremos que se guarde la información en un log o no (por defecto es False).
        save_solutions: Indica si quiere que se guarden todos los cromosomas generados en un array
        """

        self.encoding = encoding
        self.log_info = log_info
        self.max_saturation = max_saturation
        self.max_time = max_time
        self.population_size = population_size
        self.num_generations = num_generations
        self.num_elites = num_elites
        self.set_seed(seed)
        self.mutation_operator = mutation_operator
        self.mutation_probability = mutation_probability
        self.alpha = alpha
        self.crossover_probability = crossover_probability
        self.initial_population_method = initial_population_method
        self.selection_method = selection_method
        self.tau_tournament = tau_tournament
        self.crossover_operator = crossover_operator
        self.save_solutions = save_solutions

        #Guardo los operadores en un diccionario para poder representar bien el GA
        self.operators = {"selection": selection_method,
                          "mutation": mutation_operator,
                          "crossover": crossover_operator}

        #Defino algunos atributos que se utilizarán más adelante.
        self.ga_algorithm = None
        self.best_solution_found = None
        self.best_obj_val_found = None
        self.offspring_fitness = None
        self.average_offspring_fitness = None
        self.current_generation = None




        #Se definen las funciones especificas que hay que pasar como argumento al constructor de pygad.GA

        #Operador de cruce
        if crossover_operator == "OX":
            def my_crossover(parents, offspring_size, ga_instance):
                dim = len(parents[0])
                l = offspring_size[0]
                p = ga_instance.crossover_probability
                offspring = np.zeros(shape = (l,dim))

                for i in range(l//2):
                    two_parents = [parents[2*i],parents[2*i+1]]
                    x = random.uniform(0,1)
                    if x < p:
                        [o1,o2] = order_crossover(parents=two_parents)
                    else:
                        [o1,o2] = two_parents
                    offspring[2*i] = o1
                    offspring[2*i+1] = o2
                return np.array(offspring)
            self.crossover_operator = my_crossover

        #Operador de mutación para un solo cromosoma
        if mutation_operator == "ordinal_random":
            self.single_chromosome_mutation = ordinal_random_mutation
        elif "swap" in mutation_operator:
            self.single_chromosome_mutation = swap_mutation
        elif "sim" in mutation_operator:
            self.single_chromosome_mutation = simple_inversion_mutation

        #Ahora, para pasarlo como argumento a pygad hay que definir un operador de mutación que opere sobre la población completa
        if "adaptive" in mutation_operator:
            def my_mutation(offspring, ga_instance):
                    dimension = len(offspring[0])
                    num_offspring = len(offspring)
                    mutated_offspring = np.zeros(shape=(num_offspring,dimension))
                    S = lambda x : x/(1+x)
                    fitness = self.get_offspring_fitness(ga_instance)
                    av_fitness = self.get_average_offspring_fitness(ga_instance)
                    for i in range(num_offspring):
                        #La probabilidad de mutacion se modula en funcion de la aptitud del individuo
                        p = S(self.alpha*av_fitness/fitness[i])*self.mutation_probability#AÑADIR PARÁMETRO?

                        #generamos aleatoriamente el numero de mutaciones a realizar
                        x = random.uniform(0,1)#Generado de manera uniforme en [0,1]
                        num_mutations = np.log(1-x)/np.log(p)
                        num_mutations = int(np.ceil(num_mutations))-1
                        mutated_offspring[i] = offspring[i]
                        for _ in range(num_mutations):
                            mutated_offspring[i] = self.single_chromosome_mutation(mutated_offspring[i])
                    return mutated_offspring
        elif "dynamic" in mutation_operator:
            def my_mutation(offspring, ga_instance: pygad.GA):
                    num_offspring = len(offspring)
                    mutated_offspring = np.copy(offspring)#np.zeros(shape=(num_offspring,dimension))

                    #En la mutación dinámica se hace que la probabilidad de mutación descienda con las generaciones completadas
                    t = ga_instance.generations_completed+1
                    denominator = np.log(self.mutation_probability/(t**self.alpha))
                    # mut_por_gen = 0
                    for i in range(num_offspring):
                        #Primero generamos aleatoriamente el numero de mutaciones a realizar
                        x = random.uniform(0,1)#Generado de manera uniforme en [0,1]
                        num_mutations = np.log(1-x)/denominator
                        num_mutations = int(np.ceil(num_mutations))-1
                        for _ in range(num_mutations):
                            mutated_offspring[i] = self.single_chromosome_mutation(mutated_offspring[i])
                    return mutated_offspring
        else:
            def my_mutation(offspring, ga_instance):
                    dimension = len(offspring[0])
                    num_offspring = len(offspring)
                    mutated_offspring = np.zeros(shape=(num_offspring,dimension))
                    for i in range(num_offspring):
                        #Primero generamos aleatoriamente el numero de mutaciones a realizar
                        x = random.uniform(0,1)#Generado de manera uniforme en [0,1]
                        num_mutations = np.log(1-x)/np.log(self.mutation_probability)
                        num_mutations = int(np.ceil(num_mutations))-1
                        mutated_offspring[i] = offspring[i]
                        for _ in range(num_mutations):
                            mutated_offspring[i] = self.single_chromosome_mutation(mutated_offspring[i])
                    return mutated_offspring
            #Fin else
        self.mutation_operator = my_mutation
        
        
    def set_seed(self,seed):
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)



    def __repr__(self) -> str:
        """
        Crea un breve resumen de todos los parámetros que tiene el GA.
        """

        selection = self.operators["selection"]
        crossover = self.operators["crossover"]
        mutation = self.operators["mutation"]
        if selection == "tournament":
            selection_param = self.tau_tournament
        elif selection == "rank":
            selection_param = ""


        summary = f"GA de {self.population_size} individuos, {self.num_elites} élites y {self.num_generations} generaciones.\n"
        summary += f"Representación: {self.encoding}, selección: {selection}-{selection_param}, "
        summary += f"cruce: {self.crossover_probability}-{crossover}, mutación: {mutation}-{self.mutation_probability}"
        if "adaptive" in self.operators["mutation"] or "dynamic" in self.operators["mutation"]:
            summary += f" alpha={self.alpha}"
        return summary
    

    def run_GA(self,instance: TSP_instance):
        """Ejecutar el algoritmo con sus parámetros sobre la instancia del TSP dada"""

        #Guardo el tiempo de inicio de ejecución
        self.start_run_time = time.time()
        
        print(f"Ejecutando {str(self)}\nEl problema es la instancia del TSP {str(instance)}\n")
        #Alfabeto en el que están los genes
        gene_space = range(instance.dimension)

        #Se genera la población inicial de forma aleatoria
        if self.initial_population_method == "random":
            initial_pop = generate_new_pop(population_size=self.population_size,
                                       dimension=instance.dimension,
                                       encoding=self.encoding) 
        elif self.initial_population_method == "randomized_nn":
            initial_pop = instance.generate_new_pop_with_nn(population_size=self.population_size)
        else:
            pass

        #Se define la función de aptitud
        def fitness_function(ga_algorithm, permutation: list[int], idx: int) -> float:
            return 1/(instance.get_obj_val(permutation,encoding=self.encoding)+1)
        
        #Funcion que se ejecuta al final de cada generacion.
        #Controla el tiempo de ejecucion e imprime cierta informacion por pantalla
        def on_generation(ga_algorithm):
            """
            Se ejecutará después de cada iteración del GA y controla si se ha excedido el tiempo máximo de ejecución.
            """

            t = ga_algorithm.generations_completed
            if t % GA_for_TSP.generations_to_inform == 0:
                _,best_solution_fitness,_ = ga_algorithm.best_solution()
                distance = np.round(1/best_solution_fitness - 1,decimals=2)
                print(f"Distancia en la generación {t}: {distance}")
                

            if (time.time()-self.start_run_time) > self.max_time :
                return "stop"#El GA se detiene cuando se genera
            else: 
                return None
        

        #Creo una instancia de pygad.ga con todos los parámetros y funciones definidas
        ga_instance = pygad.GA(num_generations=self.num_generations,
                               sol_per_pop=self.population_size,
                               num_genes=instance.dimension,
                               gene_space=gene_space,
                               initial_population=initial_pop,
                               fitness_func=fitness_function,
                               num_parents_mating=self.population_size-self.num_elites,
                               parent_selection_type=self.selection_method,
                               crossover_type=self.crossover_operator,
                               mutation_type=self.mutation_operator,
                               random_seed=self.seed,
                               keep_parents=0,
                               keep_elitism=self.num_elites,
                               stop_criteria=f"saturate_{self.max_saturation}",
                               K_tournament=self.tau_tournament,
                               save_solutions=self.save_solutions,
                               on_generation=on_generation,
                               crossover_probability=self.crossover_probability
                               )
        #Inicio el algoritmo
        ga_instance.run()
        #Guardo el objeto algoritmo y la mejor solución encontrada
        self.ga_algorithm = ga_instance
        self.best_solution_found,_,_ = ga_instance.best_solution()
        self.best_obj_val_found = instance.get_obj_val(chromosome=self.best_solution_found,
                                                       encoding=self.encoding)
        #Guardo el número de generaciones que han sido necesarias.
        self.generations_completed = ga_instance.generations_completed


