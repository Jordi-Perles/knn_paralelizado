#include "sample.h"
#include "knn.h"
#include "mnist.h"
#include "crossValidation.h"

#include "timer.h"

#include <cassert>
#include <cmath>
#include <string>
#include <iostream>
#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <mpi.h>
#include <numeric>
#include <stdexcept>
#include <cstdlib>

using namespace std;
using namespace knn;
using namespace mnist;
using namespace crossValidation;

/// <summary>
/// Finds the best K parameter of the KNN by performing an N-Fold Cross Validation for the specified range of K.
/// </summary>
/// <param name="minK">The minimum K parameter to be tested.</param>
/// <param name="maxK">The maximum K parameter to be tested.</param>
/// <param name="dataset">The dataset to be used during the cross validation.</param>
/// <param name="folds">The number of folds.</param>
/// <returns>
/// The value of K that better performs in the cross validation.
/// </returns>

int fineTuning(unsigned int minK, unsigned int maxK, const std::vector<TSample> &dataset, unsigned int folds, int rank, int size){

    assert(minK >= 1 && maxK >= minK);

    CrossValidation crossValidation =
        CrossValidation::NFold(dataset, folds);

    const int totalTasks = (maxK - minK + 1) * folds;

    // Cada proceso almacena los resultados de las tareas
    // que tiene asignadas.
    std::vector<double> localAccuracies(totalTasks, 0.0);

    if (rank == 0)
    {
        std::cout << "Starting parallel search for best K in range ["
                  << minK << ", " << maxK << "] using "
                  << folds << " folds and "
                  << dataset.size() << " samples."
                  << std::endl;
    }

    double start = MPI_Wtime();

    // Reparto estático cíclico de las tareas.
    for (int taskId = rank; taskId < totalTasks; taskId += size)
    {
        unsigned int k = minK + taskId / folds;
        unsigned int fold = taskId % folds;

        double accuracy = crossValidation.validateFold(k, fold);

        localAccuracies[taskId] = accuracy;

        std::cout << "Process " << rank
                  << " completed K=" << k
                  << ", fold=" << fold
                  << ", accuracy=" << accuracy * 100 << "%"
                  << std::endl;
    }

    // Reunimos los resultados de todos los procesos.
    std::vector<double> globalAccuracies(totalTasks, 0.0);

    MPI_Reduce(
        localAccuracies.data(),
        globalAccuracies.data(),
        totalTasks,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    int bestK = minK;

    // Solo el proceso 0 calcula las medias y selecciona K.
    if (rank == 0)
    {
        double bestAccuracy = -1.0;

        for (unsigned int k = minK; k <= maxK; ++k)
        {
            double accuracy = 0.0;

            for (unsigned int fold = 0; fold < folds; ++fold)
            {
                int taskId = (k - minK) * folds + fold;
                accuracy += globalAccuracies[taskId];
            }

            accuracy /= folds;

            std::cout << "K: " << k
                      << " Accuracy: " << accuracy * 100
                      << "%" << std::endl;

            if (accuracy > bestAccuracy)
            {
                bestAccuracy = accuracy;
                bestK = k;
            }
        }

        std::cout << "Best k: " << bestK << std::endl;
        std::cout << "Parallel fine-tuning completed in "
                  << MPI_Wtime() - start << " seconds"
                  << std::endl;
    }

    // Todos los procesos reciben el K seleccionado.
    MPI_Bcast(&bestK, 1, MPI_INT, 0, MPI_COMM_WORLD);

    return bestK;
}

const int TAG_TASK = 1;
const int TAG_RESULT = 2;


int fineTuningDynamic(
    unsigned int minK,
    unsigned int maxK,
    const vector<TSample>& dataset,
    unsigned int folds,
    int rank,
    int size)
{
    assert(minK >= 1 && maxK >= minK);
    assert(folds >= 1);

    const int totalTasks = (maxK - minK + 1) * folds;

    int bestK = minK;
    double start = MPI_Wtime();

    if (rank == 0)
    {
        std::cout << "Starting dynamic search for best K in range ["
                  << minK << ", " << maxK << "] using "
                  << folds << " folds and "
                  << dataset.size() << " samples."
                  << std::endl;
    }

    // Caso especial: ejecución con un único proceso.
    if (size == 1)
    {
        CrossValidation cv =
            CrossValidation::NFold(dataset, folds);

        double bestAccuracy = -1.0;

        for (unsigned int k = minK; k <= maxK; ++k)
        {
            double accuracy = cv.validate(k);

            std::cout << "K: " << k
                      << " Accuracy: " << accuracy * 100
                      << "%" << std::endl;

            if (accuracy > bestAccuracy)
            {
                bestAccuracy = accuracy;
                bestK = k;
            }
        }

        std::cout << "Best k (dynamic): " << bestK << std::endl;
    }
    else
    {
        CrossValidation cv =
            CrossValidation::NFold(dataset, folds);

        if (rank == 0)
        {
            int nextTask = 0;
            int completedTasks = 0;

            std::vector<double> globalAccuracies(
                totalTasks, 0.0
            );

            // 1. Asignamos una tarea inicial a cada trabajador.
            for (int worker = 1; worker < size; ++worker)
            {
                if (nextTask < totalTasks)
                {
                    MPI_Send(
                        &nextTask, 1, MPI_INT,
                        worker, TAG_TASK, MPI_COMM_WORLD
                    );

                    ++nextTask;
                }
                else
                {
                    int stop = -1;

                    MPI_Send(
                        &stop, 1, MPI_INT,
                        worker, TAG_TASK, MPI_COMM_WORLD
                    );
                }
            }

            // 2. El maestro también calcula y gestiona resultados.
            while (completedTasks < totalTasks)
            {
                int flag = 0;
                MPI_Status status;

                // Comprobamos si algún trabajador ha terminado.
                MPI_Iprobe(
                    MPI_ANY_SOURCE,
                    TAG_RESULT,
                    MPI_COMM_WORLD,
                    &flag,
                    &status
                );

                if (flag)
                {
                    int receivedTask;
                    double receivedAccuracy;

                    // Recibimos el identificador de la tarea.
                    MPI_Recv(
                        &receivedTask,
                        1,
                        MPI_INT,
                        status.MPI_SOURCE,
                        TAG_RESULT,
                        MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE
                    );

                    // Recibimos la precisión obtenida.
                    MPI_Recv(
                        &receivedAccuracy,
                        1,
                        MPI_DOUBLE,
                        status.MPI_SOURCE,
                        TAG_RESULT,
                        MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE
                    );

                    globalAccuracies[receivedTask] =
                        receivedAccuracy;

                    ++completedTasks;

                    std::cout << "Received task "
                              << receivedTask
                              << " from process "
                              << status.MPI_SOURCE
                              << std::endl;

                    // Si quedan tareas, asignamos otra.
                    if (nextTask < totalTasks)
                    {
                        MPI_Send(
                            &nextTask, 1, MPI_INT,
                            status.MPI_SOURCE,
                            TAG_TASK, MPI_COMM_WORLD
                        );

                        ++nextTask;
                    }
                    else
                    {
                        // No quedan tareas: finalizamos el trabajador.
                        int stop = -1;

                        MPI_Send(
                            &stop, 1, MPI_INT,
                            status.MPI_SOURCE,
                            TAG_TASK, MPI_COMM_WORLD
                        );
                    }
                }
                else if (nextTask < totalTasks)
                {
                    // El maestro ejecuta una tarea propia.
                    int taskId = nextTask++;

                    unsigned int k =
                        minK + taskId / folds;

                    unsigned int fold =
                        taskId % folds;

                    double accuracy =
                        cv.validateFold(k, fold);

                    globalAccuracies[taskId] = accuracy;
                    ++completedTasks;

                    std::cout << "Process 0 completed K=" << k
                              << ", fold=" << fold
                              << ", accuracy=" << accuracy * 100
                              << "%" << std::endl;
                }
                else
                {
                    // No quedan tareas propias: esperamos un resultado.
                    MPI_Probe(
                        MPI_ANY_SOURCE,
                        TAG_RESULT,
                        MPI_COMM_WORLD,
                        &status
                    );

                    int receivedTask;
                    double receivedAccuracy;

                    MPI_Recv(
                        &receivedTask,
                        1,
                        MPI_INT,
                        status.MPI_SOURCE,
                        TAG_RESULT,
                        MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE
                    );

                    MPI_Recv(
                        &receivedAccuracy,
                        1,
                        MPI_DOUBLE,
                        status.MPI_SOURCE,
                        TAG_RESULT,
                        MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE
                    );

                    globalAccuracies[receivedTask] =
                        receivedAccuracy;

                    ++completedTasks;

                    // Ya no quedan tareas: enviamos la señal de parada.
                    int stop = -1;

                    MPI_Send(
                        &stop, 1, MPI_INT,
                        status.MPI_SOURCE,
                        TAG_TASK, MPI_COMM_WORLD
                    );
                }
            }

            // 3. Calculamos la precisión media de cada K.
            double bestAccuracy = -1.0;

            for (unsigned int k = minK; k <= maxK; ++k)
            {
                double accuracy = 0.0;

                for (unsigned int fold = 0; fold < folds; ++fold)
                {
                    int taskId =
                        (k - minK) * folds + fold;

                    accuracy += globalAccuracies[taskId];
                }

                accuracy /= folds;

                std::cout << "K: " << k
                          << " Accuracy: " << accuracy * 100
                          << "%" << std::endl;

                if (accuracy > bestAccuracy)
                {
                    bestAccuracy = accuracy;
                    bestK = k;
                }
            }

            std::cout << "Best k (dynamic): "
                      << bestK << std::endl;
        }
        else
        {
            // 4. Código de los trabajadores.
            while (true)
            {
                int taskId;

                // Esperamos una tarea del maestro.
                MPI_Recv(
                    &taskId,
                    1,
                    MPI_INT,
                    0,
                    TAG_TASK,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE
                );

                // Si recibimos -1, terminamos.
                if (taskId == -1)
                    break;

                unsigned int k =
                    minK + taskId / folds;

                unsigned int fold =
                    taskId % folds;

                // Ejecutamos la validación del fold.
                double accuracy =
                    cv.validateFold(k, fold);

                std::cout << "Process " << rank
                          << " completed K=" << k
                          << ", fold=" << fold
                          << ", accuracy=" << accuracy * 100
                          << "%" << std::endl;

                // Enviamos el identificador de la tarea.
                MPI_Send(
                    &taskId,
                    1,
                    MPI_INT,
                    0,
                    TAG_RESULT,
                    MPI_COMM_WORLD
                );

                // Enviamos la precisión obtenida.
                MPI_Send(
                    &accuracy,
                    1,
                    MPI_DOUBLE,
                    0,
                    TAG_RESULT,
                    MPI_COMM_WORLD
                );
            }
        }
    }

    // 5. Mostrar el tiempo total desde el maestro.
    if (rank == 0)
    {
        std::cout << "Dynamic fine-tuning completed in "
                  << MPI_Wtime() - start
                  << " seconds" << std::endl;
    }

    // 6. Todos los procesos reciben el mejor K.
    MPI_Bcast(
        &bestK,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    return bestK;
}



/// <summary>
/// Loads the MNIST dataset.
/// </summary>
/// <param name="train">The output training data.</param>
/// <param name="test">The output test data.</param>
/// <returns>
/// True if the dataset was loaded successfully, false otherwise.
/// </returns>
bool loadDataset(std::vector<TSample> &train, std::vector<TSample> &test)
{
    // Load training data.
    std::string img_path = "train/train-images-idx3-ubyte";
    std::string label_path = "train/train-labels-idx1-ubyte";
    if(!loadMnist(img_path.c_str(), label_path.c_str(), train))
    {
        std::cout<<"The training data could not be read"<<std::endl;
        
        return false;
    }
    
    // Load testing data.
    img_path = "test/t10k-images-idx3-ubyte";
    label_path = "test/t10k-labels-idx1-ubyte";
    if(!loadMnist(img_path.c_str(), label_path.c_str(), test))
    {
        std::cout<<"The test data could not be read"<<std::endl;
        
        return false;
    }
    
    return true;
}



bool broadcastDataset(
    std::vector<TSample>& dataset,
    int root,
    MPI_Comm communicator)
{
    int rank;
    MPI_Comm_rank(communicator, &rank);

    int numSamples = 0;
    int rows = 0;
    int cols = 0;
    int valid = 1;

    // El proceso raíz prepara los metadatos.
    if (rank == root)
    {
        numSamples = static_cast<int>(dataset.size());

        if (numSamples == 0)
            valid = 0;
        else
        {
            rows = dataset[0].image.rows;
            cols = dataset[0].image.cols;

            for (const auto& sample : dataset)
            {
                if (sample.image.rows != rows ||
                    sample.image.cols != cols ||
                    sample.image.pixels.size() !=
                        static_cast<std::size_t>(rows * cols))
                {
                    valid = 0;
                    break;
                }
            }
        }
    }

    MPI_Bcast(&valid, 1, MPI_INT, root, communicator);

    if (!valid)
        return false;

    MPI_Bcast(&numSamples, 1, MPI_INT, root, communicator);
    MPI_Bcast(&rows, 1, MPI_INT, root, communicator);
    MPI_Bcast(&cols, 1, MPI_INT, root, communicator);

    const int pixelsPerSample = rows * cols;
    const int blockSize = 256;

    // Los procesos secundarios preparan el vector de muestras.
    if (rank != root)
        dataset.resize(numSamples);

    // Transmisión por bloques para limitar la memoria temporal.
    for (int first = 0; first < numSamples; first += blockSize)
    {
        const int currentBlock = std::min(blockSize, numSamples - first);
        const int blockPixels = currentBlock * pixelsPerSample;

        std::vector<float> pixels(blockPixels);
        std::vector<int> labels(currentBlock);

        // El proceso raíz aplana únicamente el bloque actual.
        if (rank == root)
        {
            for (int i = 0; i < currentBlock; ++i)
            {
                const int sampleIndex = first + i;

                labels[i] = dataset[sampleIndex].label;

                std::copy(
                    dataset[sampleIndex].image.pixels.begin(),
                    dataset[sampleIndex].image.pixels.end(),
                    pixels.begin() + i * pixelsPerSample
                );
            }
        }

        MPI_Bcast(
            pixels.data(),
            blockPixels,
            MPI_FLOAT,
            root,
            communicator
        );

        MPI_Bcast(
            labels.data(),
            currentBlock,
            MPI_INT,
            root,
            communicator
        );

        // Los procesos secundarios reconstruyen el bloque recibido.
        if (rank != root)
        {
            for (int i = 0; i < currentBlock; ++i)
            {
                const int sampleIndex = first + i;

                dataset[sampleIndex].image.rows = rows;
                dataset[sampleIndex].image.cols = cols;
                dataset[sampleIndex].label = labels[i];

                dataset[sampleIndex].image.pixels.assign(
                    pixels.begin() + i * pixelsPerSample,
                    pixels.begin() + (i + 1) * pixelsPerSample
                );
            }
        }
    }

    return true;
}

double evaluateTestParallel(
    const std::vector<TSample>& train,
    const std::vector<TSample>& test,
    int k,
    int rank,
    int size)
{
    // Cada proceso crea el clasificador con el entrenamiento completo.
    KNN knn(train, k);

    const int totalSamples = static_cast<int>(test.size());

    // Reparto equilibrado de las muestras de test.
    const int base = totalSamples / size;
    const int remainder = totalSamples % size;

    const int localSamples = base + (rank < remainder ? 1 : 0);
    const int startIndex = rank * base + std::min(rank, remainder);
    const int endIndex = startIndex + localSamples;

    int localCorrect = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // Cada proceso clasifica únicamente sus muestras.
    for (int i = startIndex; i < endIndex; ++i)
    {
        int predictedLabel = knn.classify(test[i]);

        if (predictedLabel == test[i].label)
            ++localCorrect;
    }

    double localTime = MPI_Wtime() - start;

    // Sumamos los aciertos de todos los procesos.
    int globalCorrect = 0;

    MPI_Reduce(
        &localCorrect,
        &globalCorrect,
        1,
        MPI_INT,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    // Obtenemos el tiempo del proceso más lento.
    double globalTime = 0.0;

    MPI_Reduce(
        &localTime,
        &globalTime,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        MPI_COMM_WORLD
    );

    double accuracy = 0.0;

    if (rank == 0)
    {
        accuracy = static_cast<double>(globalCorrect) / totalSamples;

        std::cout << "Parallel test accuracy: "
                  << accuracy
                  << " completed in "
                  << globalTime
                  << " seconds"
                  << std::endl;
    }

    return accuracy;
}

/// <summary>
/// The entry point.
/// </summary>
/// <param name="argc">The number of arguments.</param>
/// <param name="argv">The arguments.</param>

// Obtiene el tamaño del conjunto de entrenamiento desde KNN_TRAIN_SIZE.
// Si no se define la variable, se mantiene el valor predeterminado de 20000.
bool getTrainingSizeFromEnv(size_t& requestedSize)
{
    const char* value = std::getenv("KNN_TRAIN_SIZE");
    requestedSize = 20000;

    if (value == nullptr || *value == '\0')
        return true;

    char* end = nullptr;
    unsigned long long parsed = std::strtoull(value, &end, 10);

    if (end == value || *end != '\0' || parsed == 0)
    {
        std::cerr << "KNN_TRAIN_SIZE must be a positive integer." << std::endl;
        return false;
    }

    requestedSize = static_cast<size_t>(parsed);
    return true;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Load MNIST dataset.
    double totalStart = elapsedTime();

    vector<TSample> train;
    vector<TSample> test;
    vector<TSample> trainingSubset;

    int datasetReady = 1;
    unsigned long long trainSize = 20000;

    // Only process 0 loads MNIST and reads the requested training size.
    if (rank == 0)
    {
        datasetReady = loadDataset(train, test) ? 1 : 0;
        size_t requestedTrainSize = 20000;
        if (datasetReady && !getTrainingSizeFromEnv(requestedTrainSize))
            datasetReady = 0;

        if (datasetReady)
        {
            trainSize = static_cast<unsigned long long>(
                min(requestedTrainSize, train.size())
            );
            train.resize(static_cast<size_t>(trainSize));
            cout << "Training samples: " << train.size() << endl;
        }
    }

    // Inform all processes whether loading/configuration was successful.
    MPI_Bcast(&datasetReady, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (!datasetReady)
    {
        MPI_Finalize();
        return 1;
    }

    // All ranks receive the selected training-set size.
    MPI_Bcast(&trainSize, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    // Fine-tuning and final test use the same selected training subset.
    if (rank == 0)
        trainingSubset = train;

    // Distribute the training subset to all processes.
    if (!broadcastDataset(trainingSubset, 0, MPI_COMM_WORLD))
    {
        if (rank == 0)
            cerr << "Error distributing training dataset." << endl;

        MPI_Finalize();
        return 1;
    }

    // Fine tune the KNN classifier with training data.
    double start = elapsedTime();

    int k = fineTuningDynamic(
        1,
        10,
        trainingSubset,
        5,
        rank,
        size
    );

    // Distribuimos los datasets completos para la evaluación final.
    bool trainReady = broadcastDataset(train, 0, MPI_COMM_WORLD);
    bool testReady = broadcastDataset(test, 0, MPI_COMM_WORLD);

    if (!trainReady || !testReady)
    {
        if (rank == 0)
            cerr << "Error distributing evaluation datasets." << endl;

        MPI_Finalize();
        return 1;
    }

    // Test the classifier with testing data.
    // Only process 0 performs the final evaluation.
    // Evaluación final paralela.
    if (rank == 0)
    {
        cout << "Best k: " << k
            << " found in " << elapsedTime() - start
            << " seconds" << endl;
    }

    double accuracy = evaluateTestParallel(
        train,
        test,
        k,
        rank,
        size
    );

    if (rank == 0)
    {
        cout << "Total elapsed time "
            << elapsedTime() - totalStart
            << " seconds" << endl;
    }

    MPI_Finalize();
    return 0;
}
