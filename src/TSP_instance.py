#Contiene una clase que representa una instancia del TSP

import numpy as np, networkx as nx
import os, tsplib95, copy
from configparser import ConfigParser
from amplpy import AMPL,add_to_path
from Custom_functions import ordinal_to_permutation
import matplotlib.pyplot as plt

#Se cargan las rutas a cada una de las carpetas que se necesitan
config = ConfigParser()
config.read("Configuracion.conf")
config_name = "CONFIGURACION"
INSTANCES = config.get(config_name,"instances_directory")
SOLUTIONS = config.get(config_name,"solutions_directory")
DAT_MATRIX_FOLDER = config.get(config_name,"dat_matrix_folder")
AMPL_PATH = config.get(config_name,"ampl")
MOD = config.get(config_name,"mod")

class TSP_instance:
    """
    Representa una instancia del problema del viajante de comercio (TSP).\n 
    La clase está diseñada principalmente para obtener las instancias mediante la librería tsplib95, guardar información relevante y crear plots de posibles soluciones.
    """

    def random_tour_with_nearest_neighbour(self, exponent = 4):
        tour = np.zeros(self.dimension)
        L = list(range(self.dimension))
        np.random.shuffle(L)
        tour[0] = L.pop()

        remaining_cities = self.dimension-1
        for i in range(1,self.dimension-1):
            inv_distances = np.zeros(remaining_cities)
            for j in range(remaining_cities):
                d = max(self.get_distance(tour[i-1],L[j]),1)
                inv_distances[j] = 1/(d**exponent)
            S = np.sum(inv_distances)
            probabilities = inv_distances/S
            selected_idx = np.random.choice(range(remaining_cities),p=probabilities)
            tour[i] = L.pop(selected_idx)
            remaining_cities -= 1
        tour[self.dimension-1] = L[0]
        return tour
    
    def generate_new_pop_with_nn(self,population_size: int) -> list:
        pop = [None]*population_size
        for i in range(population_size):
            pop[i] = self.random_tour_with_nearest_neighbour()
        return pop
    

    def __init__(self,problem_name = None , weight_matrix = None) -> None:
        if problem_name is not None:
            #Primero se carga el problema y se guardan sus datos
            instance_path = os.path.join(INSTANCES,f"{problem_name}.txt")
            problem = tsplib95.load(instance_path)
            dimension = problem.dimension

            matrix = np.zeros(shape=(dimension,dimension))
            e=0
            try:
                problem.get_weight(0,1)
            except (IndexError, KeyError):
                e=1

            for i in range(dimension):
                for j in range(dimension):
                    matrix[i][j] = problem.get_weight(i+e,j+e)
            
            self.weight_matrix = matrix
            self.dimension = dimension
            self.name = problem.name
            self.is_euclidean = (problem.edge_weight_type == "EUC_2D")
            self.path = instance_path

            self.has_optimal_solution = False
            self.optimal_solution = None
            self.optimal_value = None
            #Ahora cargamos la solucion optima que esta en otra carpeta
            solution_path = os.path.join(SOLUTIONS,self.name + "_sol.txt") 
            if os.path.isfile(solution_path):
                self.has_optimal_solution = True
                solution = tsplib95.load(solution_path)
                self.optimal_solution = np.array(solution.tours[0])-1
                self.optimal_value = self.get_obj_val(self.optimal_solution)

        elif weight_matrix is not None:
            self.dimension = len(weight_matrix)
            self.weight_matrix = weight_matrix
            self.name = f"TSP{self.dimension}"
    

    #------------------------------------------------------
    # La instancia del problema se representará con su nombre
    #------------------------------------------------------
    def __repr__(self) -> str:
        return self.name

    def get_distance(self, u: int, v: int) -> float:
        """Obtener distancia entre dos nodos."""

        u = int(u); v = int(v)
        return self.weight_matrix[u][v]
    

    def get_obj_val(self, chromosome: list[int], encoding : str = "permutational") -> float:
        """
        Calcular longitud total de un ciclo hamiltoniano.

        chromosome: Cromosoma que representa un tour
        encoding: Representación del tour. Puede ser "ordinal" o "permutational".
        """

        #Obtengo el tour en forma permutacional
        if encoding == "ordinal":
            cycle = ordinal_to_permutation(chromosome)
        elif encoding == "permutational":
            cycle = chromosome
        #Suma la longitud de todas las aristas
        suma = 0.0
        dimension = len(cycle)
        for i in range(dimension-1):
            suma += self.get_distance(cycle[i],cycle[i+1])
        suma += self.get_distance(cycle[dimension-1],cycle[0])
        return suma
    
    def get_score(self, chromosome: list[int], encoding : str = "permutational") -> float:
        if not self.has_optimal_solution:
            raise RuntimeError(f"La instancia {str(self)} no tiene disponible una solución óptima")
        distance = self.get_obj_val(chromosome=chromosome,encoding=encoding)
        return (distance - self.optimal_value)/self.optimal_value * 100

    

    def create_dat_file(self):
        """Crear un fichero .dat en el que se guarda la dimensión del problema y la matriz de distancias para su uso en AMPL."""

        d = self.dimension
        info = f"param d := {d};\n\n"
        info += "param w := [*,*] :\n"

        #Se añade una linea que lista las filas
        for i in range(1,d+1):
            info += f"{i} "
        info += ":=\n"

        #Ahora se añaden las filas de la matriz
        for fila in range(d):
            info += f"{fila+1} "
            for col in range(d):
                info += f"{self.get_distance(fila,col)} "
            info += "\n"
        
        dat_path = os.path.join(DAT_MATRIX_FOLDER,f"{self.name}.dat")
        with open(dat_path,"w") as dat_file:
            dat_file.write(info)



    def run_in_AMPL(self,solver: str) -> AMPL:
        """Ejecutar la instancia en AMPL mediante el modelo secuencial."""

        add_to_path(AMPL_PATH)
        ampl = AMPL()
        #Se asigna el solver deseado
        ampl.set_option("solver",solver)
        
        #Creo el path al fichero .dat y el .mod
        dat_path = os.path.join(DAT_MATRIX_FOLDER,f"{self.name}.dat")
        mod_path = MOD

        #Los leo con el objeto ampl
        ampl.read(mod_path)
        ampl.read_data(dat_path)

        #Resuelvo
        ampl.solve()

        return ampl
    

    def get_graph_without_edges(self) -> nx.Graph:
        """Obtener objeto de tipo Graph sin aristas en el que los nodos tienen guardadas sus coordenadas en el plano."""

        try:
            #La primera vez que se llama este método el grafo se crea, pero las siguientes ya se habrá guardado como atributo.
            return self.G
        except AttributeError:
            lib_problem = tsplib95.load(self.path)
            if lib_problem.edge_weight_type != "EUC_2D":
                raise ValueError("Para obtener el grafo de una instancia los nodos deben venir dados por coordenadas en el plano")
            
            node_coords = lib_problem.node_coords
            d = self.dimension

            G = nx.Graph()
            for i in range(d):
                x = node_coords[i+1][0]
                y = node_coords[i+1][1]
                G.add_node(node_for_adding=i,pos=(x,y))
            self.G = G
            return G
        
    
    def get_graph_with_cycle(self, chromosome:list, encoding:str = "permutational")-> nx.Graph:
        """Obtener objeto de tipo Graph en el que los nodos tienen posiciones en el plano. Además se añaden las aristas de un ciclo hamiltoniano."""

        if encoding == "permutational":
            cycle = chromosome
        elif encoding == "ordinal":
            cycle = ordinal_to_permutation(chromosome)
        G = copy.deepcopy(self.get_graph_without_edges())
        d = self.dimension
        for i in range(d-1):
            u = cycle[i]
            v = cycle[i+1]
            G.add_edge(u_of_edge=u,v_of_edge=v)
        G.add_edge(cycle[d-1],cycle[0])
        return G


    def get_graph_with_opt_cycle(self) -> nx.Graph:
        """Como get_graph_with_cycle pero con el ciclo óptimo que aporta la librería tsplib95"""
        if not self.has_optimal_solution:
            raise ValueError("El grafo no dispone de ciclo óptimo")
        return self.get_graph_with_cycle(self.optimal_solution)
    
    def draw_graph_with_cycle(self, chromosome: list = None, encoding: str = "permutational", node_size = 100, font_size = 5):
        """Crea una ventana con un plot del ciclo indicado."""

        if chromosome is None:
            G = self.get_graph_with_opt_cycle()
        else:
            G = self.get_graph_with_cycle(chromosome=chromosome,encoding=encoding)
        
        nx.draw(G, nx.get_node_attributes(G, 'pos'),with_labels=True,node_size=node_size,font_size=font_size)
        plt.show()

    def save_plot_with_cycle(self, path, chromosome: list = None, encoding: str = "permutational", node_size = 100, font_size = 5):
        """Guarda un plot del ciclo especificado en el directorio dado"""

        if chromosome is None:
            G = self.get_graph_with_opt_cycle()
        else:
            G = self.get_graph_with_cycle(chromosome=chromosome,encoding=encoding)

        nx.draw(G, nx.get_node_attributes(G, 'pos'),with_labels=True,node_size=node_size,font_size=font_size)
        plt.savefig(path)
        plt.clf()