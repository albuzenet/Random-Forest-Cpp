#include <memory>
#include <array>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstring>
#include "DecisionTreeClassifier.hpp"

Node::Node() :
    feature(0),
    threshold(0),
    impurity(1),
    prediction(-1),
    left(nullptr),
    right(nullptr) {};

bool Node::is_leaf()
{
    return left == nullptr && right == nullptr;
};

int Node::Predict(const std::vector<int>& y, const std::vector<int>& samples)
{
    std::unordered_map<int, int> counter;

    for (int i = 0; i < samples.size(); i++)
    {
        int class_ = y[samples[i]];
        counter[class_]++;
    }

    int max_count = 0;
    int result;

    for (auto [class_, count] : counter) {
        if (max_count < count) {
            result = class_;
            max_count = count;
        }
    }

    return result;
};

double Node::Impurity(const std::vector<int>& y, const std::vector<int>& samples)
{

    unsigned int n_class = *std::max_element(y.begin(), y.end()) + 1;

    std::vector<int> count_class(n_class, 0);

    for (int i = 0; i < samples.size(); i++)
    {
        count_class[y[samples[i]]] += 1;
    }

    double gini = 1.0;

    for (int class_ = 0; class_ < n_class; class_++)
    {
        double proportion = count_class[class_] / static_cast<double>(samples.size());
        gini -= proportion * proportion;
    }

    return gini;
};


double Node::Impurity(
    const std::vector<int>& y,
    const std::vector<int>& samples_left,
    const std::vector<int>& samples_right
)
{
    double n_left = samples_left.size();
    double n_right = samples_right.size();
    double n_tot = n_left + n_right;

    return n_left / n_tot * Impurity(y, samples_left) + n_right / n_tot * Impurity(y, samples_right);

};

std::tuple<std::vector<int>, std::vector<int>> Node::Split(
    unsigned int feature,
    double threshold,
    const std::vector<std::vector<double>>& X,
    std::vector<int>& samples
)
{
    std::vector<int> samples_false;
    std::vector<int> samples_true;

    for (int i = 0; i < samples.size(); i++)
    {
        if (X[samples[i]][feature] <= threshold)
        {
            samples_false.push_back(samples[i]);
        }
        else
        {
            samples_true.push_back(samples[i]);
        }
    }

    return { samples_false, samples_true };
};


std::tuple<std::vector<int>, std::vector<int>> Node::FindBestSplit(
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& y,
    std::vector<int>& samples)
{
    int n_features = X[0].size();
    int n_samples = samples.size();
    std::vector<int> sample_left_best;
    std::vector<int> sample_right_best;
    double min_gini = 1.0;

    for (int i = 0; i < n_samples; i++)
    {
        for (int j = 0; j < n_features; j++)
        {
            auto [samples_left, samples_right] = Split(j, X[samples[i]][j], X, samples);

            double gini = Impurity(y, samples_left, samples_right);

            if (gini < min_gini)
            {
                min_gini = gini;
                feature = j;
                threshold = X[samples[i]][j];
                sample_left_best = samples_left;
                sample_right_best = samples_right;
            }
        }
    }

    return { sample_left_best, sample_right_best };
}

TreeBuilder::TreeBuilder(const std::vector<std::vector<double>>& X_, const std::vector<int>& y_)
    : X(X_), y(y_) {};

void TreeBuilder::Build(std::unique_ptr<Node>& node, std::vector<int>& samples)
{
    if (samples.size() <= 1)
    {
        node->impurity = 0;
        node->prediction = node->Predict(y, samples);
        return;
    }
    auto [left_samples, right_samples] = node->FindBestSplit(X, y, samples);

    node->impurity = node->Impurity(y, samples);

    node->left = std::make_unique<Node>();
    node->right = std::make_unique<Node>();

    Build(node->left, left_samples);
    Build(node->right, right_samples);
}

DecisionTreeClassifier::DecisionTreeClassifier() :root(std::make_unique<Node>()) {};

void DecisionTreeClassifier::Fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y)
{
    TreeBuilder builder(X, y);

    std::vector<int> samples(y.size(), 0);

    for (int i = 0; i < samples.size(); i++)
    {
        samples[i] = i;
    }

    builder.Build(root, samples);
}

std::vector<int> DecisionTreeClassifier::Predict(const std::vector<std::vector<double>>& X)
{
    std::vector<int> predictions;

    for (const std::vector<double>& sample : X)
    {
        predictions.push_back(Predict(root, sample));
    };

    return predictions;
}

int DecisionTreeClassifier::Predict(
    std::unique_ptr<Node>& node,
    const std::vector<double>& sample
)
{
    if (node->is_leaf())
    {
        return node->prediction;
    };

    if (sample[node->feature] <= node->threshold)
    {
        return Predict(node->left, sample);
    }
    else
    {
        return Predict(node->right, sample);
    }
};

double DecisionTreeClassifier::Score(std::vector<std::vector<double>> X, std::vector<int> y)
{
    // Return the accuracy

    std::vector<int> predictions = Predict(X);

    double true_positives = 0;

    for (int i = 0; i < y.size(); i++)
    {
        true_positives += predictions[i] == y[i];
    }

    return true_positives / y.size();
};