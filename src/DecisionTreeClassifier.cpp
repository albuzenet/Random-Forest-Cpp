#include <memory>
#include <array>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstring>
#include <numeric>
#include <iostream>
#include "../include/Profiling.hpp"
#include "../include/DecisionTreeClassifier.hpp"

#define PROFILING 1
#if PROFILING
#define PROFILE_SCOPE(name) InstrumentationTimer timmer##__LINE__(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)
#else
#define PROFILE_SCOPE(name)
#define PROFILE_FUNCTION()
#endif


DataSet::DataSet(const std::vector<std::vector<double>>& X_, const std::vector<int>& y_)
    : X(X_), y(y_), Xf(X.size(), 0)
{
    std::iota(samples.begin(), samples.end(), 0);

    n_class = *std::max_element(y.begin(), y.end()) + 1;
    n_features = X[0].size();
};


void DataSet::SortFeature(std::size_t start, std::size_t end)
{
    PROFILE_FUNCTION();

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
    PROFILE_FUNCTION();

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

    if (start >= end)
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
    double n_samples = n_left + n_right;

    return n_left / n_samples * Impurity(n_left, left_class) + n_right / n_samples * Impurity(n_right, right_class);
};

void Node::CountClassFrequency(const DataSet& data)
{
    for (int i = start; i <= end; i++)
    {
        int k = data.y[data.samples[i]];
        node_class[k]++;
    }
};

std::size_t Node::BestSplit(DataSet& data)
{
    PROFILE_FUNCTION();

    double best_gini = 1.0;
    std::size_t best_split = start;

    for (int i = 0; i < data.n_features; i++)
    {
        for (int j = start; j <= end; j++)
        {
            data.Xf[j] = data.X[data.samples[j]][i];
        }

        data.SortFeature(start, end);
        {
            PROFILE_SCOPE("AfterSort");
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
    }
    return best_split;
}


DecisionTreeClassifier::DecisionTreeClassifier()
    : root(nullptr) {};


void DecisionTreeClassifier::Fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y)
{
    Instrumentor::Get().BeginSession("Profile");

    DataSet data(X, y);
    root = Build(data, 0, y.size() - 1);

    Instrumentor::Get().EndSession();
}


std::unique_ptr<Node> DecisionTreeClassifier::Build(DataSet& data, std::size_t start, std::size_t end)
{
    PROFILE_FUNCTION();

    std::unique_ptr<Node> node = std::make_unique<Node>(start, end, data.n_class);

    node->CountClassFrequency(data);
    node->impurity = node->Impurity(end - start + 1, node->node_class);

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
        node->right = Build(data, split + 1, end);
    }

    return node;
}


std::vector<int> DecisionTreeClassifier::Predict(const std::vector<std::vector<double>>& X)
{
    std::vector<int> predictions;

    for (const std::vector<double>& sample : X)
    {
        predictions.push_back(_Predict(root, sample));
    };

    return predictions;
}


int DecisionTreeClassifier::_Predict(const std::unique_ptr<Node>& node, const std::vector<double>& sample)
{
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
    // Return the accuracy

    std::vector<int> predictions = Predict(X);

    double true_positives = 0;

    for (int i = 0; i < y.size(); i++)
    {
        true_positives += predictions[i] == y[i];
    }

    return true_positives / y.size();
};