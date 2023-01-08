#include <memory>
#include <array>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstring>
#include <string>
#include <cmath>
#include <numeric>
#include <iostream>
#include <random>

#include "../include/DecisionTreeClassifier.hpp"


DataSet::DataSet(const std::vector<std::vector<double>>& X_, const std::vector<int>& y_, std::string max_features_)
    : X(X_), y(y_), Xf(X.size(), 0), samples(X.size()), features(X[0].size())
{

    /**
     * Constructs a DataSet object from the given input data.
     *
     * @param X A matrix of input features, where each row represents a sample and each column represents a feature.
     * @param y A vector of class labels, where each element represents the class label of the corresponding sample in X_.
     * @param max_features The maximum number of features to consider when looking for the best split point at each node.
     *                      The value can be either "sqrt"  or "all"
     */

    std::iota(samples.begin(), samples.end(), 0);
    std::iota(features.begin(), features.end(), 0);

    n_class = *std::max_element(y.begin(), y.end()) + 1;
    n_features = X[0].size();

    if (max_features_ == "sqrt")
    {
        max_features = floor(sqrt(n_features));
        max_features = max_features > 1 ? max_features : 1;
    }
    else if (max_features_ == "all")
    {
        max_features = n_features;
    }
    else
    {
        max_features = n_features;
    };
};


void DataSet::SortFeature(std::size_t start, std::size_t end)
{
    // Sorts the samples and the values in Xf according to the values in Xf.

    std::vector<std::pair<double, std::size_t>> sorter(Xf.size(), std::make_pair(0, 0));

    for (int i = start; i <= end; i++)
    {
        sorter[i].first = Xf[i];
        sorter[i].second = samples[i];
    }

    std::sort(
        sorter.begin() + start,
        sorter.end() + (end + 1 - Xf.size()),
        [](std::pair<double, std::size_t>& a, std::pair<double, std::size_t>& b) {return a.first < b.first;}
    );

    for (int i = start; i <= end; i++)
    {
        Xf[i] = sorter[i].first;
        samples[i] = sorter[i].second;
    }
};


void Node::SplitSamples(DataSet& data)
{
    /**
     * Reorganizes the samples in the data set according to the threshold and feature stored in this node.
     * The samples with values in the feature lower or equal to the threshold will be moved to the left of the split point,
     * and the samples with values higher than the threshold will be moved to the right.
     */

    std::size_t left = start;
    std::size_t right = end;

    while (left < right)
    {
        if (data.X[data.samples[left]][feature] <= threshold)
        {
            left++;
        }
        else
        {
            std::swap(data.samples[left], data.samples[right]);
            right--;
        }
    }
};

double Node::Impurity(double n_samples, const std::vector<int>& node_class_)
{
    // Calculate the Gini impurity of a node

    if ((start >= end) || (n_samples == 0))
    {
        return 0.0;
    };

    double gini = 1.0;

    for (int k = 0; k < n_class; k++)
    {
        double proportion = node_class_[k] / n_samples;
        gini -= proportion * proportion;
    }

    return gini;
};


Node::Node(std::size_t start_, std::size_t end_, int n_class_)
    :
    start(start_),
    end(end_),
    feature(0),
    threshold(0),
    impurity(1),
    prediction(-1),
    n_samples(end - start + 1),
    n_class(n_class_),
    is_leaf(false),
    node_class(n_class, 0),
    left(nullptr),
    right(nullptr) {};


int Node::Predict(const DataSet& data)
{
    // Return the most frequent class in the y vector between start and end

    std::vector<int> count_class(data.n_class, 0);

    for (int i = start; i <= end; i++)
    {
        int k = data.y[data.samples[i]];
        count_class[k]++;
    }

    int max = 0;
    int prediction;

    for (int k = 0; k < data.n_class; k++) {

        if (max < count_class[k]) {
            prediction = k;
            max = count_class[k];
        }
    }

    return prediction;
};


double Node::ChildsImpurity(int n_left, int n_right, const std::vector<int>& left_class, const std::vector<int>& right_class)
{
    // Calculate the mean Gini impurity of two childs
    // Used to evalute the quality of a split in Node::BestSplit

    double n_samples = n_left + n_right;

    return n_left / n_samples * Impurity(n_left, left_class) + n_right / n_samples * Impurity(n_right, right_class);
};

void Node::CountClassFrequency(const DataSet& data)
{
    // Count the class frequency in the target vector (between start and end)

    for (int i = start; i <= end; i++)
    {
        int k = data.y[data.samples[i]];
        node_class[k]++;
    }
};

std::size_t Node::BestSplit(DataSet& data)
{
    // Find the best split, ie. determine the feature & threshold of the node that minimize the mean Gini of its child. Most of the work happens here.
    // The trick is to sort Xf before the inner loop on the sample axis. This reduce the complexity regarding the number of samples from O(n²) to O(nlog(n)) 
    // Since we sort, it is no longer necessary to determine which sample is below or above the currently evaluated.

    double best_gini = 1.0;
    std::size_t best_split = start;

    std::shuffle(data.features.begin(), data.features.end(), std::mt19937{ std::random_device{}() });

    for (int m = 0; m < data.max_features; m++)
    {
        int i = data.features[m];

        for (int j = start; j <= end; j++)
        {
            data.Xf[j] = data.X[data.samples[j]][i];
        }

        data.SortFeature(start, end);

        int n_left = 0;
        int n_right = n_samples;
        std::vector<int> left_class(n_class, 0);
        std::vector<int> right_class = node_class;

        for (int split = start; split < end; split++)
        {
            int k = data.y[data.samples[split]];

            n_left++;
            left_class[k]++;
            n_right--;
            right_class[k]--;

            double mean_gini = ChildsImpurity(n_left, n_right, left_class, right_class);

            if (mean_gini < best_gini)
            {
                best_split = split;
                best_gini = mean_gini;
                feature = i;
                threshold = data.Xf[split];
            }
        }
    }
    return best_split;
}


DecisionTreeClassifier::DecisionTreeClassifier(std::string max_features_)
    : root(nullptr), max_features(max_features_) {};



void DecisionTreeClassifier::Fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y)
{
    // Fits a Decision Tree Classifier model to the input data

    DataSet data(X, y, max_features);
    root = Build(data, 0, y.size() - 1);
}


std::unique_ptr<Node> DecisionTreeClassifier::Build(DataSet& data, std::size_t start, std::size_t end)
{
    // Recursivly build the decision tree using the data provided
    // Each node is built using a subset of the data (data.samples[start:end])

    std::unique_ptr<Node> node = std::make_unique<Node>(start, end, data.n_class);

    node->CountClassFrequency(data);
    node->impurity = node->Impurity(end - start + 1, node->node_class);

    // Check if the node is a leaf node
    if (node->n_samples <= 1 || node->impurity == 0.0)
    {
        node->impurity = 0.0;
        node->is_leaf = true;
        node->prediction = node->Predict(data);
    }
    else
    {
        std::size_t split = node->BestSplit(data);

        node->SplitSamples(data);

        node->left = Build(data, start, split);

        if (split < end)
        {
            node->right = Build(data, split + 1, end);
        }
    }

    return node;
}


std::vector<int> DecisionTreeClassifier::Predict(const std::vector<std::vector<double>>& X)
{
    // Makes predictions for a set of samples.

    std::vector<int> predictions;

    for (const std::vector<double>& sample : X)
    {
        predictions.push_back(_Predict(root, sample));
    };

    return predictions;
}


int DecisionTreeClassifier::_Predict(const std::unique_ptr<Node>& node, const std::vector<double>& sample)
{
    // Make a prediction recursively for single sample.

    if (node->is_leaf)
    {
        return node->prediction;
    };

    if (sample[node->feature] <= node->threshold)
    {
        return _Predict(node->left, sample);
    }
    else
    {
        return _Predict(node->right, sample);
    }
};


double DecisionTreeClassifier::Score(const std::vector<std::vector<double>>& X, const std::vector<int>& y)
{
    // Calculates the mean accuracy of the model on the given data set.

    std::vector<int> predictions = Predict(X);

    double true_positives = 0;

    for (int i = 0; i < y.size(); i++)
    {
        true_positives += predictions[i] == y[i];
    }

    return true_positives / y.size();
};