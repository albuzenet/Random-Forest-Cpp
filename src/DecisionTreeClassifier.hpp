# ifndef NODE_HPP
# define NODE_HPP

#include <memory>
#include <vector>

class Node

{
public:
    Node();

    int Predict(const std::vector<int>& y, const std::vector<int>& samples);
    double Impurity(const std::vector<int>& y, const std::vector<int>& samples);
    double Impurity(
        const std::vector<int>& y,
        const std::vector<int>& samples_left,
        const std::vector<int>& samples_right
    );
    std::tuple<std::vector<int>, std::vector<int>> Split(
        unsigned int feature,
        double threshold,
        const std::vector<std::vector<double>>& X,
        std::vector<int>& samples
    );
    std::tuple<std::vector<int>, std::vector<int>> FindBestSplit(
        const std::vector<std::vector<double>>& X,
        const std::vector<int>& y,
        std::vector<int>& samples
    );
    bool is_leaf();

    int feature;
    double threshold;
    double impurity;
    int prediction;

    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
};

class TreeBuilder
{
public:
    TreeBuilder(const std::vector<std::vector<double>>& X_, const std::vector<int>& y_);
    void Build(std::unique_ptr<Node>& node, std::vector<int>& samples);

private:

    std::vector<std::vector<double>> X;
    std::vector<int> y;
};


class DecisionTreeClassifier
{
public:
    DecisionTreeClassifier();
    void Fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y);
    std::vector<int> Predict(const std::vector<std::vector<double>>& X);
    double Score(std::vector<std::vector<double>> X, std::vector<int> y);
    std::unique_ptr<Node> root;
private:
    int Predict(std::unique_ptr<Node>& node, const std::vector<double>& sample);
};

# endif
