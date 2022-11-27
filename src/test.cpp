#include "DecisionTreeClassifier.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    DecisionTreeClassifier tree = DecisionTreeClassifier();

    std::vector<std::vector<double>> X{ {5, 3}, {2, 4}, {9, 7}, {1, 8}, {15, 14}, {17, 13}, {10, 6}, {11, 12} };
    std::vector<int> y{ 0, 1, 2, 3, 4, 5, 6, 7 };

    tree.Fit(X, y);
    std::cout << tree.Score(X, y) << std::endl;

    // std::cout << tree.root->impurity << std::endl;
    // std::cout << tree.root->left->impurity << std::endl;
    // std::cout << tree.root->right->impurity << std::endl;
    // std::cout << tree.root->left->left->impurity << std::endl;
    // std::cout << tree.root->left->right->impurity << std::endl;
    // std::cout << tree.root->right->left->impurity << std::endl;
    // std::cout << tree.root->right->right->impurity << std::endl;

    // std::cout << "----------------------" << std::endl;

    // std::cout << tree.root->feature << " <= " << tree.root->threshold << std::endl;
    // std::cout << tree.root->left->feature << " <= " << tree.root->left->threshold << std::endl;
    // std::cout << tree.root->right->feature << " <= " << tree.root->right->threshold << std::endl;
    // std::cout << tree.root->left->left->feature << " <= " << tree.root->left->left->threshold << std::endl;
    // std::cout << tree.root->left->right->feature << " <= " << tree.root->left->right->threshold << std::endl;
    // std::cout << tree.root->right->left->feature << " <= " << tree.root->right->left->threshold << std::endl;
    // std::cout << tree.root->right->right->feature << " <= " << tree.root->right->right->threshold << std::endl;

    // std::cout << "----------------------" << std::endl;

    // std::vector<int> y_pred = tree.Predict(X);

    // for (auto& value : y_pred)
    // {
    //     std::cout << value << std::endl;
    // }

    // std::cout << "----------------------" << std::endl;

}