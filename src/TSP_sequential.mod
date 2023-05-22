# Fomurlación clásica del problema del viajante de comercio
param d;#Dimension o numero de ciudades
set nodes = 1..d;#Nodos
param w{nodes,nodes} >= 0;#Pesos o distancias

#variables
var x{i in nodes, j in nodes: i <> j} binary;
var u{i in nodes};

#Funcion objetivo
minimize distance: sum{i in nodes, j in nodes: i <> j} w[i,j]*x[i,j];

#Restricciones
subject to edgeIn{j in 1..d}: sum{i in nodes: i <> j} x[i,j] = 1;
subject to edgeOut{i in 1..d}: sum{j in nodes: j <> i} x[i,j] = 1;
subject to beginning:u[1] = 0;
subject to order{i in nodes, j in nodes: i <> j and j <> 1}: u[i]-u[j] + d*x[i,j] <= d-1;

