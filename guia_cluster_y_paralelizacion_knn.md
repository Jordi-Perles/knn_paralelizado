# Pruebas en clúster y paralelización de kNN con MPI

## 1. Preparación del dataset

El dataset MNIST se descarga una sola vez en el ordenador principal, después de clonar el repositorio.

```bash
git clone URL_DEL_REPOSITORIO
cd knn2_Paralelizado
chmod +x script.sh
./script.sh --descargar
```

El script ejecuta `get_dataset.sh` y prepara estos cuatro archivos:

```text
build/
├── train/
│   ├── train-images-idx3-ubyte
│   └── train-labels-idx1-ubyte
└── test/
    ├── t10k-images-idx3-ubyte
    └── t10k-labels-idx1-ubyte
```

Una vez descargados, no es necesario repetir la descarga mientras los archivos sigan presentes. El `script.sh` actual sincroniza el proyecto mediante `rsync`; como no excluye `build/`, también copia el dataset a los nodos remotos.

Si se clona el repositorio en otro ordenador sin esos datos, habrá que descargarlos o copiarlos allí.

## 2. Configuración del clúster

### 2.1. Identificar los ordenadores

En cada ordenador Linux, ejecutar:

```bash
hostname
hostname -I
nproc
```

- `hostname`: nombre del equipo.
- `hostname -I`: direcciones IP asignadas.
- `nproc`: número de procesadores lógicos disponibles.

Se puede utilizar el nombre del equipo o su dirección IP en el `hostfile`, siempre que el nodo principal pueda resolver el nombre y conectarse por SSH. En una red de aula, si no se sabe si los nombres se resuelven correctamente, es práctico utilizar las IP de la red del aula.

### 2.2. Crear el hostfile

El archivo `hostfile` se coloca en el directorio raíz del proyecto, junto a `script.sh`.

Ejemplo para tres ordenadores:

```text
localhost slots=8
192.168.1.102 slots=8
192.168.1.103 slots=8
```

Los nombres y las IP anteriores son ilustrativos: deben sustituirse por los datos reales del aula.

- `localhost` representa el ordenador desde el que se lanza MPI.
- Las demás entradas representan los nodos remotos.
- `slots` indica la capacidad declarada a Open MPI para alojar procesos en ese equipo. No es necesariamente un límite físico estricto de CPU.

El `script.sh` actual utiliza automáticamente `hostfile` en el directorio del proyecto si no se especifica otro con `--hostfile`.

### 2.3. Comprobar SSH

Desde el ordenador principal, probar:

```bash
ssh 192.168.1.102 hostname
ssh 192.168.1.103 hostname
```

Las conexiones deben funcionar sin solicitar interactivamente una contraseña, porque el script utiliza `ssh -o BatchMode=yes`.

También se pueden comprobar las herramientas de los nodos:

```bash
ssh 192.168.1.102 "which mpirun && which mpic++ && which rsync"
```

Todos los nodos deben tener un entorno MPI compatible y las herramientas necesarias.

## 3. Comprobar que el clúster funciona

Conviene verificar por separado la conectividad SSH, el lanzamiento remoto de procesos MPI y la ejecución de un programa MPI.

### 3.1. Comprobar que MPI lanza procesos en diferentes nodos

Desde el ordenador principal:

```bash
mpirun --hostfile hostfile --map-by node -np 3 hostname
```

- `--hostfile hostfile`: utiliza la lista de nodos.
- `--map-by node`: intenta distribuir los procesos de forma rotatoria entre los nodos.
- `-np 3`: solicita tres procesos MPI.
- `hostname`: cada proceso muestra el equipo en el que se ejecuta.

Si se utilizan tres nodos, se espera ver sus nombres. Esta prueba comprueba el lanzamiento remoto. La opción `--map-by node` se usa explícitamente en esta prueba; el script de experimentos actual no la incorpora y Open MPI aplica su política predeterminada.

### 3.2. Comprobar la ejecución de un programa MPI

Crear temporalmente `prueba_mpi.cpp`:

```cpp
#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int length;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Get_processor_name(hostname, &length);

    std::cout << "Proceso " << rank
              << " de " << size
              << " ejecutándose en " << hostname
              << std::endl;

    MPI_Finalize();
    return 0;
}
```

Compilar y ejecutar:

```bash
mpic++ prueba_mpi.cpp -o prueba_mpi
mpirun --hostfile hostfile --map-by node -np 3 ./prueba_mpi
```

Si aparecen los tres procesos con sus respectivos equipos, se ha verificado que el clúster puede lanzar un programa MPI real. Esta prueba no mide el rendimiento ni valida todavía el algoritmo kNN.

## 4. Ejecutar los experimentos

### 4.1. Configuración

En `script.sh`:

```bash
PROCESS_COUNTS=(3 6 9 12 15 18 21 24 27)
REPETITIONS=1
PROBLEM_SIZES=(5000 10000 20000)
```

Esto representa 9 configuraciones de procesos por 3 tamaños de problema: 27 configuraciones en total, con una repetición por configuración.

Una repetición permite una primera exploración, pero no es suficiente para obtener resultados estadísticamente sólidos. Conviene repetir las configuraciones más interesantes varias veces.

### 4.2. Capacidad y número de procesos

Los `slots` del hostfile deben reflejar la capacidad que se declara para cada máquina. El número total de procesos que se solicita se controla mediante `PROCESS_COUNTS` y la opción `-np` que utiliza el script.

Por ejemplo, tres nodos con ocho slots cada uno declaran 24 slots en total. Solicitar 27 procesos superaría esa capacidad declarada y Open MPI podría rechazar la ejecución. No se deben aumentar artificialmente los slots: hay que comprobar la capacidad real y las políticas del laboratorio.

### 4.3. Orden de ejecución en clase

1. Identificar los ordenadores, sus IP y sus procesadores lógicos.
2. Elegir el ordenador principal.
3. Clonar el repositorio en el principal.
4. Descargar MNIST una vez.
5. Crear el `hostfile` con los nodos reales.
6. Comprobar SSH sin contraseña.
7. Probar el lanzamiento MPI con `hostname`.
8. Ejecutar el pequeño programa MPI de prueba.
9. Verificar que el número de procesos no supera los slots declarados, salvo que el laboratorio permita explícitamente la sobreasignación.
10. Ejecutar primero una prueba pequeña de kNN.
11. Lanzar la batería completa.

Comando para los experimentos dinámicos en clúster:

```bash
./script.sh --cluster --dinamica
```

Si el archivo no se llama `hostfile` o no está en el directorio raíz del proyecto:

```bash
./script.sh --cluster --hostfile RUTA_AL_HOSTFILE --dinamica
```

No es necesario volver a utilizar `--descargar` mientras el dataset siga presente.

### 4.4. Resultados

El script guarda los resultados en un directorio similar a:

```text
results/
└── dinamica/
    └── dinamica_experimentos_FECHA.txt
```

El registro incluye los tiempos, información de CPU y memoria y estadísticas de las repeticiones válidas.

---

# 5. Paralelización del algoritmo kNN
«Nuestro proyecto consiste en implementar el algoritmo kNN para clasificar imágenes del conjunto de datos MNIST. Hemos desarrollado tres versiones: una secuencial, una paralela con MPI estática y otra paralela con MPI dinámica.

En primer lugar, el programa tiene que encontrar cuál es el mejor valor de K, es decir, cuántos vecinos debemos tener en cuenta para clasificar una imagen.

Para encontrarlo, probamos diferentes valores de K y utilizamos validación cruzada con cinco particiones, o folds. Cada fold permite evaluar el rendimiento del algoritmo utilizando diferentes divisiones de los datos.

Aquí es donde hemos aplicado el paralelismo: las evaluaciones de los folds son independientes, por lo que podemos repartirlas entre diferentes procesos MPI. Así, en lugar de que un único proceso realice todas las evaluaciones, varios procesos trabajan simultáneamente.

Después, recogemos los resultados de las evaluaciones, determinamos qué valor de K ha obtenido mejores resultados y lo utilizamos para la clasificación final.

En esta segunda fase también hemos aplicado paralelismo. Repartimos las imágenes del conjunto de test entre los diferentes procesos, de manera que cada uno clasifica una parte de las imágenes de forma independiente. Finalmente, reunimos los resultados para calcular la precisión global.

Para distribuir el trabajo hemos implementado dos estrategias diferentes.
- En la versión estática, las tareas se reparten previamente entre los procesos. Cada proceso sabe desde el principio qué evaluaciones tiene que realizar.
- En la versión dinámica, utilizamos un modelo maestro-trabajadores. Un proceso maestro gestiona las tareas pendientes y las asigna a los procesos trabajadores conforme van quedando disponibles. Cuando un trabajador termina una tarea, devuelve el resultado y puede recibir otra.
La diferencia principal es que, en la versión estática, el reparto se establece de antemano, mientras que en la dinámica las tareas se van asignando durante la ejecución.

Por último, comparamos los tiempos de ejecución de las tres versiones para diferentes tamaños del conjunto de entrenamiento y distintos números de procesos, con el objetivo de estudiar la mejora de rendimiento que conseguimos mediante la paralelización».