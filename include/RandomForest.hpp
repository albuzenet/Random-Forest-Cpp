# ifndef RANDOMFOREST_HPP
# define RANDOMFORESTE_HPP

#include <memory>
#include <vector>
#include "DecisionTreeClassifier.hpp"

class RandomForest
{
public:
    RandomForest(int n_estimators_);

    void Fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y);
    std::vector<int> Predict(const std::vector<std::vector<double>>& X);
    double Score(const std::vector<std::vector<double>>& X, const std::vector<int>& y);

    int n_estimators;
    std::vector<DecisionTreeClassifier> estimators;
};

# endif