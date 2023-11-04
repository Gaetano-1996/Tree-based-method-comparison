# Tree-based method comparison

## Introduction
In this project, we explore and compare tree-based methods (CART, bagging, random forest, and boosting) in terms of performance and interpretability of results on the benchmark Boston Housing dataset.

## Dependencies
The project made use of the scikit-learn, matplotlib, and seaborn libraries.
Before proceeding **be sure you have installed the necessary dependencies** running the command:
`pip3 install -r requirements.txt`

## Project structure
The project is divided into:
- **src**: containing the source code for classes, methods and functions
   - *AutoTree.py*: containing the class created to manage decision trees and methods.
   - *AutoEnsemble.py*: containing the class created to manage decision tree ensemble and methods.
   - *utils.py*: containing the support function used in the analysis.
- _BostonHousing.ipynb_: containing the notebook for the analysis (basically you can clone the repo, and run this notebook to see the results yourself)
- _requirements.txt_: self-explicative.
- _Boston.csv_: data used for the comparison.
- **main.py**
 
## Usage
To run the project:
- From Notebook: modify the path variable in the second cell of the notebook with the path of the Boston.csv file and run all cells.
- From main.py: modify the path variable in the main.py file with the path of the Boston.csv file and run the file.
