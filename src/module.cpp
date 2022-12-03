# ifdef PYBIND11

#include <Windows.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../include/DecisionTreeClassifier.hpp"

namespace py = pybind11;

PYBIND11_MODULE(example1, m) {
    py::class_<DecisionTreeClassifier>(m, "DecisionTreeClassifier")
        .def(py::init<>())
        .def("Fit", &DecisionTreeClassifier::Fit)
        .def("Predict", &DecisionTreeClassifier::Predict)
        .def("Score", &DecisionTreeClassifier::Score);
}

#endif