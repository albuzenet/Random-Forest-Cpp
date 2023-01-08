#include <map>
#include <algorithm>

#include "../include/DecisionTreeClassifier.hpp"
#include "../include/RandomForest.hpp"

RandomForest::RandomForest(int n_estimators_) : n_estimators(n_estimators_)
{
    estimators.reserve(n_estimators);

    for (int i = 0; i < n_estimators; i++)
    {
        estimators.push_back(DecisionTreeClassifier("sqrt"));
    }
}

void RandomForest::Fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y)
{
    for (DecisionTreeClassifier& tree : estimators)
    {
        tree.Fit(X, y);
    }
}

std::vector<int> RandomForest::Predict(const std::vector<std::vector<double>>& X)
{
    std::vector<std::vector<int>> predictions;
    std::vector<int> predict_proba;

    for (DecisionTreeClassifier& tree : estimators)
    {
        predictions.push_back(tree.Predict(X));
    }

    for (int j = 0; j < X.size(); j++)
    {
        double proba = 0;

        std::map<int, int> counter;

        for (int i = 0; i < n_estimators; i++)
        {
            counter[predictions[i][j]]++;
        }

        int mode = std::max_element(
            counter.begin(),
            counter.end(),
            [&](auto& lhs, auto& rhs) {return lhs.second < rhs.second;}
        )->first;

        predict_proba.push_back(mode);
    }

    return predict_proba;
}

double RandomForest::Score(const std::vector<std::vector<double>>& X, const std::vector<int>& y)
{

    std::vector<int> predictions = Predict(X);

    double true_positives = 0;

    for (int i = 0; i < y.size(); i++)
    {
        true_positives += predictions[i] == y[i];
    }

    return true_positives / y.size();
};