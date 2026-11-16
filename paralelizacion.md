# Procedimiento Clase

Te dejo el orden que seguiría, con los comandos concretos.
1. Identificar los ordenadores
   
   En cada ordenador:
   hostname
   hostname -I
   nproc
   
   
   Anota las IP y los procesadores lógicos disponibles.
2. Elegir el ordenador principal
   
   Será el ordenador desde el que ejecutarás todos los comandos y lanzarás los experimentos.
3. Clonar el repositorio en el principal
   git clone https://github.com/Jordi-Perles/knn_paralelizado.git
   cd knn2_Paralelizado
   chmod +x script.sh
4. Descargar MNIST
   ./script.sh --descargar
   
   
   Comprueba que los cuatro archivos del dataset se han descargado correctamente.
5. Crear el hostfile
   hostname
   hostname -I
   nproc
   Por ejemplo, para tres equipos con ocho slots declarados cada uno:
   localhost slots=8
   192.168.1.102 slots=8
   192.168.1.103 slots=8
6. Comprobar SSH
   ssh 192.168.1.102 hostname
   ssh 192.168.1.103 hostname
   
   Ambas conexiones deben funcionar sin solicitar contraseña.
7. Comprobar el clúster con una prueba pequeña
   
   Antes de lanzar los experimentos largos, prueba una ejecución MPI sencilla entre los equipos. También puedes comprobar dónde se ejecutan los procesos con:
   mpirun --hostfile hostfile --map-by node -np 3 hostname
   
   Deberías ver los nombres de los equipos donde se han iniciado los tres procesos.
8. Lanzar los experimentos dinámicos
   ./script.sh --descargar --cluster --dinamica
   
   
   Si ya has descargado MNIST, puedes omitir --descargar:
   ./script.sh --cluster --dinamica