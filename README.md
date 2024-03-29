# Random Forest (C++ with a Python API)

An implementation of the Random Forests classifier algorithm using C++.
Implement an sklearn-like API in Python using Pybind11.
This implementation is multiclass and single output.

## Build
You need to have Microsoft Visual Studio 2017 or newer installed to build this package.
Pybind11 will automatically compile and create links between the C++ and Python types.
All dependencies will be automatically installed using pip.

```
cd .\RandomForestCpp
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
pip install .
```

## Example

You can interact with the C++ types directly in Python.<br />
Both the RandomForest and DecisionTreeClassifier implement the classic fit, predict, score methods.
This is similar to the api used by sklearn.

For example, we can test the estimator on one of the toy dataset of sklearn :

```python

from cppclassifier import RandomForest

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', as_frame=False)
mnist.target = mnist.target.astype(int)

X_train, X_test, y_train, y_test = mnist.data[:5_000, :], mnist.data[5_000:10_000, :], mnist.target[:5_000], mnist.target[5_000:10_000]

tree = RandomForest(n_estimators=100)
tree.fit(X_train, y_train)

print(f"Accuracy on the test set = {tree.score(X_test, y_test):.2%}")
```

Output
```
Accuracy on the test set = 93.02%
```
