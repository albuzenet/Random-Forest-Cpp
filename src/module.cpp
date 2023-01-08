# ifdef PYBIND11

#include <Windows.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../include/DecisionTreeClassifier.hpp"
#include "../include/RandomForest.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cppclassifier, m) {
    py::class_<DecisionTreeClassifier>(m, "DecisionTreeClassifier")
        .def(py::init<std::string>(), py::arg("max_features") = "all")
        .def("fit", &DecisionTreeClassifier::Fit)
        .def("predict", &DecisionTreeClassifier::Predict)
        .def("score", &DecisionTreeClassifier::Score);

    py::class_<RandomForest>(m, "RandomForest")
        .def(py::init<int>(), py::arg("n_estimators") = 100)
        .def("fit", &RandomForest::Fit)
        .def("predict", &RandomForest::Predict)
        .def("score", &RandomForest::Score);
}

#endif