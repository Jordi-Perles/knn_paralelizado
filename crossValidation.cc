#include "crossValidation.h"
#include "sample.h"
#include "knn.h"

#include "timer.h"

#include <vector>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <string>
#include <sstream>

using namespace knn;
using namespace crossValidation;

/// <summary>
/// Performs an N-Fold Cross Validation with the specified KNN parameter.
/// </summary>
/// <param name="k">The K parameter of the KNN classifier.</param>
/// <returns>
/// A value in the range [0.0, 1.0] indicating the averaged accuracy of the KNN in the N folds.
/// </returns>
double CrossValidation::validate(unsigned int k) const
{
    double accuracy = 0.0;

    for (unsigned int fold = 0; fold < this->folds; ++fold)
    {
        accuracy += validateFold(k, fold);
    }

    return accuracy / this->folds;
}

double CrossValidation::validateFold(unsigned int k, unsigned int fold) const
{
    if (fold >= this->folds)
        throw std::invalid_argument("Invalid fold");

    std::vector<TSample> train, test;
    this->splitDataset(train, test, fold);

    KNN knn(train, k);
    std::vector<int> labels;

    return knn.classifyAndEvaluate(test, labels);
}

/// <summary>
/// Splits the dataset into train and validation subsets for an N-Fold Cross Validation.
/// </summary>
/// <param name="train">The output training data.</param>
/// <param name="validation">The output validation data.</param>
/// <param name="fold">The fold to be used as validation data.</param>
void CrossValidation::splitDataset(std::vector<TSample> &train, std::vector<TSample> &validation, unsigned int fold) const
{
    assert(fold < this->folds && "Invalid fold");

    int start = this->dataset.size() * fold / this->folds;
    int end = this->dataset.size() * (fold + 1) / this->folds;

    for(int i=0;i<this->dataset.size();i++)
    {
        if(i<start || i>=end)
            train.push_back(this->dataset[i]);
        else
            validation.push_back(this->dataset[i]);
    }
}

/// <summary>
/// Gets the name of the Cross Validation, either N-Fold or leave-one-out.
/// </summary>
std::string CrossValidation::getName() const
{
    std::stringstream ss("");
    if(this->folds < this->dataset.size())
        ss<<folds<<"-Folds Cross Validation";
    else
        ss<<"leave-one-out Cross Validation";
        
    return ss.str();
}
