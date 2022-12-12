# ifndef NODE_HPP
# define NODE_HPP

#include <memory>
#include <vector>


class DataSet
{
public:
    DataSet(const std::vector<std::vector<double>>& X, const std::vector<int>& y);
    void SortFeature(std::size_t start, std::size_t end);
    void SplitSamples(std::size_t start, std::size_t end, std::size_t split);

    std::vector<std::vector<double>> X;
    std::vector<int> y;
    std::vector<double> Xf;
    std::vector<std::size_t> samples;
    int n_class;
    int n_features;
};


class Node
{
public:
    Node(std::size_t start, std::size_t end, int n_class);
    double Impurity(double n_samples, const std::vector<int>& node_class);
    double ChildsImpurity(int n_left, int n_right, const std::vector<int>& left_class, const std::vector<int>& right_class);
    std::size_t BestSplit(DataSet& data);
    void SplitSamples(DataSet& data);
    void CountClassFrequency(const DataSet& data);
    int Predict(const DataSet& data);

    std::size_t start;
    std::size_t end;
    int feature;
    double threshold;
    double impurity;
    int prediction;
    int n_samples;
    int n_class;
    bool is_leaf;

    std::vector<int> node_class;

    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
};


class DecisionTreeClassifier
{
public:
    DecisionTreeClassifier();
    void Fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y);
    std::vector<int> Predict(const std::vector<std::vector<double>>& X);
    double Score(const std::vector<std::vector<double>>& X, const std::vector<int>& y);
private:
    std::unique_ptr<Node> root;
    int _Predict(const std::unique_ptr<Node>& node, const std::vector<double>& sample);
    std::unique_ptr<Node> Build(DataSet& data, std::size_t start, std::size_t end);
};

# endif
