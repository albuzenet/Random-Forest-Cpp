#ifndef PYBIND11

#include <iostream>
#include <string>
#include "../include/DecisionTreeClassifier.hpp"
#include "../include/Profiling.hpp"


int main(int argc, char* argv[])
{

    std::vector<std::vector<double>> X{ {5, 3},{2, 4},{9, 7},{1, 8},{15, 14},{17, 13},{10, 6},{11, 12} };
    std::vector<int> y{ 0, 1, 2, 3, 4, 5, 6, 7 };

    DecisionTreeClassifier tree = DecisionTreeClassifier();

    // Instrumentor::Get().BeginSession("Profile");
    tree.Fit(X, y);
    // Instrumentor::Get().EndSession();


    std::cout << "Score = " << tree.Score(X, y) << std::endl;

}

# endif