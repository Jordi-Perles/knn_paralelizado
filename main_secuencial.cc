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
#include <cstdlib>

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
int fineTuning(unsigned int minK, unsigned int maxK, const std::vector<TSample> &dataset, unsigned int folds)
{
    assert(minK >= 1 && maxK > minK);
    
    CrossValidation crossValidation = CrossValidation::NFold(dataset, folds);
    std::cout<<"Starting search for best K parameter in range ["<<minK<<", "<<maxK<<"] using "<<crossValidation.getName()<<" with "<<dataset.size()<<" samples."<<std::endl;
    
    unsigned int bestK = minK;
    double bestAccuracy = 0.0;
    for(int k = minK; k <= maxK; k++)
    {
        std::cout<<"Testing k "<<k<<std::endl;
        double start = elapsedTime();
        
        double accuracy = crossValidation.validate(k);
        std::cout<<"K: "<<k<<" Accuracy: "<<accuracy * 100<<"% completed in "<<elapsedTime() - start<<" seconds"<<std::endl;
        
        if(accuracy > bestAccuracy)
        {
            bestAccuracy = accuracy;
            bestK = k;
        }
    }
    
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

/// <summary>
/// The entry point.
/// </summary>
/// <param name="argc">The number of arguments.</param>
/// <param name="argv">The arguments.</param>
int main(int argc, char** argv) 
{
    // Load MNIST dataset.
    double totalStart = elapsedTime();
    std::vector<TSample> train;
    std::vector<TSample> test;
    if(!loadDataset(train, test))
    {
        return 1;
    }

    size_t requestedTrainSize = 20000;
    if (!getTrainingSizeFromEnv(requestedTrainSize))
        return 1;

    const size_t actualTrainSize = std::min(requestedTrainSize, train.size());
    train.resize(actualTrainSize);
    std::cout << "Training samples: " << train.size() << std::endl;

    // Fine tune the KNN classifier with the selected training data.
    double start = elapsedTime();
    int k = fineTuning(1, 10, train, 5);
    std::cout<<"Best k: "<<k<<" found in "<<elapsedTime() - start<<" seconds"<<std::endl;
    
    // Test the classifier with testing data.
    start = elapsedTime();
    std::vector<int> labels;
    KNN knn(train, k);
    double accuracy = knn.classifyAndEvaluate(test, labels);
    
    std::cout<<"Test accuracy: "<<accuracy<<" completed in "<<elapsedTime() - start<<" seconds"<<std::endl;
    
    std::cout<<"Total elapsed time "<<elapsedTime() - totalStart<<" seconds"<<std::endl;
    
    return 0;
}
