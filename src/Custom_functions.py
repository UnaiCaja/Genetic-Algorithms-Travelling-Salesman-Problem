import os, csv, random, numpy as np
from configparser import ConfigParser

config = ConfigParser()
config.read("Configuracion.conf")
config_name = "CONFIGURACION"
INSTANCES_INFO = config.get(config_name,"instances_information")
GA_SOLUTIONS = config.get(config_name,"GA_solutions")

# FUNCIONES PARA AYUDAR A LA EJECUCION DE GAS

def generate_permutation(dimension: int) -> list[int]:
    """
    Generar permutacion aleatoria de [0,...,dimension-1]
    """

    result = [i for i in range(dimension)]
    random.shuffle(result)
    return np.array(result)



def permutation_to_ordinal(permutation: list[int],ref_list = None):
    """
    Transformar tour representado como sucesión de ciudades a representación ordinal.
    """

    #Si no se da una lista de referencia, se toma [0,...,n-1]
    if ref_list is None:
        n = len(permutation)
        ref_list = list(range(n))

    ordinal = np.zeros(n)
    for i, gen in enumerate(permutation):
        ordinal[i] = ref_list.index(gen)
        ref_list.pop(int(ordinal[i]))
    return ordinal


def ordinal_to_permutation(chromosome: list[int],ref_list = None):
    """
    Transformar tour con representación ordinal a representación permutacional.
    """

    #Si no se da una lista de referencia, se toma [0,...,n-1]
    if ref_list is None:
        n = len(chromosome)
        ref_list = list(range(n))

    permutation = np.zeros(n)
    for i, gen in enumerate(chromosome):
        permutation[i] = ref_list[int(gen)]
        ref_list.pop(int(gen))
    return permutation


def generate_ordinal_chromosome(dimension: int) -> list[int]:
    """
    Genera un tour aleatorio dado en representación ordinal.
    """

    chromosome = np.zeros(dimension)
    for i in range(dimension):
        gen = random.randint(0,dimension-i-1)
        chromosome[i] = gen
    return chromosome
        


def generate_new_pop(population_size: int , dimension: int, encoding : str = "permutational") -> list:
    """
    Generar población de tours aleatoriamente.

    population_size: Tamaño de la población.
    dimension: Longitud de cada tour.
    encoding: Representación de los tours. Puede ser "ordinal" o "permutational".
    """

    pop = [None]*population_size
    if encoding == "permutational":
        for i in range(population_size):
            pop[i] = generate_permutation(dimension)
        return pop
    elif encoding == "ordinal":
        for i in range(population_size):
            pop[i] = generate_ordinal_chromosome(dimension)
        return pop




def order_crossover(parents: list) -> list:
    """
    Cruzar dos tours mediante el algoritmo de cruce OX.

    parents: Lista con dos entradas. Cada una de ellas un tour con representación permutacional.
    """

    dimension = len(parents[0])
    #Primero se generan los puntos de cruce
    j1 = random.randrange(dimension)
    j2 = random.randrange(dimension)
    while j2 == j1:
        j2 = random.randrange(dimension)
    if j1 > j2:
        [j1,j2] = [j2,j1]
    
    #Ahora creamos los hijos
    child0 = np.copy(parents[1])
    child1 = np.copy(parents[0])
    #La parte central (j1+1):(j2+1) se copia de uno de los padres

    #Ciudades para rellenar en el primer hijo
    for_child0 = np.concatenate([
            np.setdiff1d(ar1=parents[0][(j2+1):dimension],
                         ar2=child0[(j1+1):(j2+1)],
                         assume_unique=True),
            np.setdiff1d(ar1=parents[0][0:(j2+1)],
                         ar2=child0[(j1+1):(j2+1)],
                         assume_unique=True)
        ])
    #Ciudades para rellenar en el segundo hijo
    for_child1 = np.concatenate([
            np.setdiff1d(ar1=parents[1][(j2+1):dimension],
                         ar2=child1[(j1+1):(j2+1)],
                         assume_unique=True),
            np.setdiff1d(ar1=parents[1][0:(j2+1)],
                         ar2=child1[(j1+1):(j2+1)],
                         assume_unique=True)
        ])
    
    child0[(j2+1):dimension] = for_child0[0:(dimension-j2-1)]
    child0[0:(j1+1)] = for_child0[(dimension-j2-1):(dimension-j2+j1+1)]
    child1[(j2+1):dimension] = for_child1[0:(dimension-j2-1)]
    child1[0:(j1+1)] = for_child1[(dimension-j2-1):(dimension-j2+j1+1)]
    return [child0,child1]

    


def ordinal_random_mutation(ordinal: list[int]) -> list[int]:
    """Cambiar una de las coordenadas de un tour con representación ordinal aleatoriamente."""

    dimension = len(ordinal)
    result = ordinal.copy()
    i = random.randint(0,dimension-1)
    #Solo hay que tener en cuenta que 0 <= ci <= n-i-1
    gen = random.randint(0,dimension-i-1)
    result[i] = gen
    return result


def swap_mutation(permutation: list[int]) -> list[int]:
    """Intercambiar la posición de dos ciudades elegidas aleatoriamente."""

    dimension = len(permutation)
    i = random.randrange(0,dimension)
    j = random.randrange(0,dimension)
    
    mutated = permutation.copy()
    aux = mutated[i]
    mutated[i] = mutated[j]
    mutated[j] = aux
    return mutated


def simple_inversion_mutation(permutation: list[int]) -> list[int]:
    """Elige un subtour dentro de una permutación de forma aleatoria e invierte su orden"""

    dimension = len(permutation)
    #Genero los dos puntos que definen el subtour y los ordeno
    j1 = random.randrange(0,dimension)
    j2 = j1
    while j2 == j1:
        j2 = random.randrange(0,dimension)
    if j1 > j2:
        [j1,j2] = [j2,j1]
    #Si la permutación es x1-...-xd entonces el subtour a invertir es x(j1+1)-...-xj2
    mutated = permutation.copy()
    mutated[(j1+1):(j2+1)] = np.flip(permutation[(j1+1):(j2+1)])
    return mutated

# FUNCIONES PARA MANEJAR ARCHIVOS

def get_instance_names():
    """Accede a un .csv donde están guardada la información sobre las instancias a las que aplico GA"""

    with open(INSTANCES_INFO,"r") as csv_file:
        reader = csv.DictReader(csv_file)
        names = []
        for dictionary in reader:
            names.append(dictionary["Nombre"])
    return names


def has_header(csv_path: str) -> bool:
    """Checks if a csv file already has content in it"""

    with open(csv_path, 'r') as csvfile:
        text=csvfile.readline()
        return (len(text) > 5)



def dump_csv(init_path,final_path):
    """Dadas dos rutas a ficheros csv, toma el contenido del primero, lo vuelca en el segundo y borra el primero."""

    if os.path.isfile(init_path):
        with open(final_path,"a") as final_csv:
            with open(init_path,"r") as init_csv:
                reader = csv.reader(init_csv)
                writer = csv.writer(final_csv)
                header = has_header(csv_path=final_path)
                writing_condition = not header
                for row in reader:
                    if writing_condition == True:
                        writer.writerow(row)
                    else:
                        writing_condition = True
        os.remove(init_path)


def join_information(final_path: str):
    """
    Juntar la información de ficheros de nombre file_name_1.csv,file_name_2.csv,... en un solo fichero file_name.csv
    """

    i=0
    (root,ext) = os.path.splitext(final_path)
    # init_path = os.path.join(root,f"_{i}{ext}")
    init_path = root + f"_{i}{ext}"
    while os.path.isfile(init_path):
        dump_csv(init_path=init_path,final_path=final_path)
        i += 1
        init_path = root + f"_{i}{ext}"