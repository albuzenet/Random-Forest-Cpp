#ifndef PYBIND11

#include <iostream>
#include <string>
#include "../include/DecisionTreeClassifier.hpp"

int main(int argc, char* argv[])
{

    std::vector<std::vector<double>> X{ {5, 3}, {2, 4}, {9, 7}, {1, 8}, {15, 14}, {17, 13}, {10, 6}, {11, 12}, {33, 44} };
    std::vector<int> y{ 0, 1, 2, 3, 4, 5, 6, 7, 8 };

    DecisionTreeClassifier tree = DecisionTreeClassifier();
    tree.Fit(X, y);
    std::cout << "Score = " << tree.Score(X, y) << std::endl;

    // for (auto& value : tree.Predict(X))
    // {
    //     std::cout << value << std::endl;
    // }

    DecisionTreeClassifier tree2 = DecisionTreeClassifier();
    tree2.Fit(X, y);
    std::cout << "Score = " << tree2.Score(X, y) << std::endl;

    // for (auto& value : tree2.Predict(X))
    // {
    //     std::cout << value << std::endl;
    // }

    DecisionTreeClassifier tree3 = DecisionTreeClassifier();
    tree3.Fit(X, y);
    std::cout << "Score = " << tree3.Score(X, y) << std::endl;

    // for (auto& value : tree3.Predict(X))
    // {
    //     std::cout << value << std::endl;
    // }

    DecisionTreeClassifier tree4 = DecisionTreeClassifier();
    tree4.Fit(X, y);
    std::cout << "Score = " << tree4.Score(X, y) << std::endl;

    // for (auto& value : tree4.Predict(X))
    // {
    //     std::cout << value << std::endl;
    // }
    // std::cout << tree.root->impurity << std::endl;
    // std::cout << tree.root->left->impurity << std::endl;
    // std::cout << tree.root->right->impurity << std::endl;
    // std::cout << tree.root->left->left->impurity << std::endl;
    // std::cout << tree.root->left->right->impurity << std::endl;
    // std::cout << tree.root->right->left->impurity << std::endl;
    // std::cout << tree.root->right->right->impurity << std::endl;

    // std::cout << "----------------------" << std::endl;

    // std::cout << tree.root->feature << " <= " << tree.root->threshold << std::endl;
    // std::cout << tree.root->left->feature << " <= " << tree.root->left->threshold << " prediction = " << tree.root->left->prediction << std::endl;
    // std::cout << tree.root->right->feature << " <= " << tree.root->right->threshold << std::endl;
    // std::cout << tree.root->left->left->feature << " <= " << tree.root->left->left->threshold << std::endl;
    // std::cout << tree.root->left->right->feature << " <= " << tree.root->left->right->threshold << std::endl;
    // std::cout << tree.root->right->left->feature << " <= " << tree.root->right->left->threshold << " prediction = " << tree.root->right->left->prediction << std::endl;
    // std::cout << tree.root->right->right->feature << " <= " << tree.root->right->right->threshold << " prediction = " << tree.root->right->right->prediction << std::endl;

    // std::cout << "----------------------" << std::endl;

    // std::vector<int> y_pred = tree.Predict(X);


    // std::cout << "----------------------" << std::endl;

}

# endif