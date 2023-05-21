import numpy as np, time, os, csv
from amplpy import AMPL,add_to_path
from configparser import ConfigParser
from TSP_instance import TSP_instance

config = ConfigParser()
config.read("Configuracion.conf")
config_name = "CONFIGURACION"
DAT_MATRIX_FOLDER = config.get(config_name,"dat_matrix_folder")
AMPL_PATH = config.get(config_name,"ampl")
MOD = config.get(config_name,"mod")

def get_optimal_solution(ampl,dim) -> list:
    """Dado el objeto ampl, extrae un tour optimo a partir de los valores de las variables"""

    x_var = ampl.get_variable("x")
    x_list = x_var.get_values().to_list()
    
    x = np.zeros(shape=(dim,dim))
    for item in x_list:
        i = int(item[0]-1)
        j = int(item[1]-1)
        x[i][j]=item[2]
    
    tour = np.zeros(dim)

    for tour_idx in range(1,dim):
        city1 = int(tour[tour_idx-1])
        for city2 in range(dim):
            if x[city1,city2] == 1:
                tour[tour_idx] = city2
                break
    return tour

def get_score(tour:list,instance: TSP_instance) -> float:
    tour_dist = instance.get_obj_val(tour,encoding="permutational")
    opt_dist = instance.optimal_value
    return (tour_dist-opt_dist)/opt_dist*100

def write_row_in_csv(path: str, row):
    with open(path,"a") as csv_file:
            writer = csv.writer(csv_file,delimiter=";")
            writer.writerow(row)


def AMPL_study(problem_names: list[str],csv_path: str, solver = "gurobi"):
    """
    Ejecutar AMPL sobre todas las instancias para las que se indica su nombre, dando un tiempo maximo especificado. Los resultados se guardan en un csv.
    """

    #Preparo AMPL para la ejecucion
    add_to_path(AMPL_PATH)
    ampl = AMPL()
    #Cabecera para escribir los resultados en el csv
    header = ["Nombre","Tiempo de computo","Score"]
    write_row_in_csv(path=csv_path,row=header)

    #Para cada problema se ejecuta AMPL
    for problem_name in problem_names:
        #Se resetea AMPL
        ampl.reset()

        #Se cargan las rutas a los ficheros correspondientes
        dat_path = os.path.join(DAT_MATRIX_FOLDER,f"{problem_name}.dat")
        mod_path = MOD#Fichero con el modelo
        ampl.set_option("solver",solver)
        ampl.read(mod_path)
        ampl.read_data(dat_path)

        #Se ejecuta AMPL
        start = time.time()
        ampl.solve()
        end = time.time()

        #Guardamos los resultados 
        info = [0]*len(header)
        info[0] = problem_name
        info[1] = end-start


        #Para obtener el score se necesita cargar la instancia
        instance = TSP_instance(problem_name=problem_name)
        AMPL_tour = get_optimal_solution(ampl=ampl,dim=instance.dimension)
        info[2] = get_score(AMPL_tour,instance=instance)

        #Lo escribimos todo en un csv
        write_row_in_csv(path=csv_path,row=info)


def AMPL_study_on_limited_time(problem_names: list[str],max_times: list[float],csv_path: str, solver = "gurobi"):
    """
    Ejecutar AMPL sobre todas las instancias para las que se indica su nombre, dando un tiempo maximo especificado. Los resultados se guardan en un csv.
    """

    #Se comprueba que se dan tantos tiempos como problemas
    if len(problem_names) != len(max_times):
        raise ValueError(f"Los argumentos problem_names y max_times deben tener la misma longitud. Longitudes recibidas {len(problem_names)} y {len(max_times)}")

    #Preparo AMPL para la ejecucion
    add_to_path(AMPL_PATH)
    ampl = AMPL()
    #Cabecera para escribir los resultados en el csv
    header = ["Nombre","Tiempo maximo","Tiempo de computo","Score"]
    write_row_in_csv(path=csv_path,row=header)

    #Para cada problema se ejecuta AMPL
    for i,problem_name in enumerate(problem_names):
        #Se resetea AMPL
        ampl.reset()

        #Se cargan las rutas a los ficheros correspondientes
        dat_path = os.path.join(DAT_MATRIX_FOLDER,f"{problem_name}.dat")
        mod_path = MOD#Fichero con el modelo
        ampl.set_option("solver",solver)
        ampl.read(mod_path)
        ampl.read_data(dat_path)
        #Tiempo maximo de computo
        ampl.set_option(f"{solver}_options",f"timelimit={max_times[i]}")

        #Se ejecuta AMPL
        start = time.time()
        ampl.solve()
        end = time.time()

        #Guardamos los resultados 
        info = [0]*len(header)
        info[0] = problem_name
        info[1] = max_times[i]
        info[2] = end-start


        #Para obtener el score se necesita cargar la instancia
        instance = TSP_instance(problem_name=problem_name)
        AMPL_tour = get_optimal_solution(ampl=ampl,dim=instance.dimension)
        info[3] = get_score(AMPL_tour,instance=instance)

        #Lo escribimos todo en un csv
        write_row_in_csv(path=csv_path,row=info)