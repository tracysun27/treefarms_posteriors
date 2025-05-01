from experiments.globals import CP4IM_DATASET_NAMES, SYNTH_NUM_TREES
import time
from typing import Tuple, Dict, Any
from maptree import search as maptree_search
from experiments.searchers.binary_classification_tree import BinaryClassificationTree, split
from experiments.searchers.maptree import *
from experiments.globals import get_stratified_k_folds_cp4im_dataset, get_full_cp4im_dataset, run_search, save_results
import pdb
import numpy as np

# helper function for parse tree 
# (for case where there's an unbalanced branch, ex. left doesn't split but right does)
def predict_by_simulating(X, y, feature, direction):
    # left is false (feature == 0) and right is true(feature == 1)
    mask = (X[:, feature] == int(direction))
    if mask.sum() == 0:
        return 0 # default to predicting 0 if nothing
    counts = np.bincount(y[mask], minlength=2)
    return np.argmax(counts)

# function to convert maptree into json, like gosdt/treefarms does
# input: maptree (binary classification tree) class
# output: recursive dict structure with left and right for each split, 
# as well as prediction labels assigned based on mode (if fit on data, will default to 0 if not)
def parse_tree(tree, X, y, verbose=0):
    # base case: if the number has parentheses directly to left and right of it,
    # like (1), then the splits on this feature are leaves  
    s = str(tree)
    if s.startswith('(') and s.endswith(')') and s[1:-1].isdigit():
        if verbose == 1:
            print("leaf:", int(s[1:-1]))
        return {"feature": int(s[1:-1]), 
                "left": {"prediction": np.argmax(tree.left.label_counts) if tree.left else predict_by_simulating(X, y, int(s[1:-1]), False)},
                "right": {"prediction": np.argmax(tree.right.label_counts) if tree.right else predict_by_simulating(X, y, int(s[1:-1]), True)}}

    # recursive case: if the number has parentheses like this )10( then it's got leaves
    s = s[1:-1]
    i, balance = 0, 0
    while i < len(s): 
    # find the root feature of this tree by counting parentheses 
    # until they get to the middle (balanced number of parentheses on either side)
        if s[i] == '(':
            balance += 1
        elif s[i] == ')':
            balance -= 1
        elif balance == 0 and s[i].isdigit():
            root_feature = ''
            while i < len(s) and s[i].isdigit():
                root_feature += s[i]
                i += 1
            root_feature = int(root_feature)

            if verbose == 1:
                print("Splitting on feature:", root_feature)

            left_indices, right_indices = split(X, root_feature) 
            # use split function in binary classification tree class
            if verbose == 1:
                print(tree.left)
                print(tree.right)

            
            if str(tree.left) != "": # left side continues splitting
                left_tree = parse_tree(tree.left, X[left_indices], y[left_indices], verbose)
            else: # left side is none (empty str)
                pred = predict_by_simulating(X, y, root_feature, False)
                left_tree = {"prediction": pred}
            
            if str(tree.right) != "": # left side continues splitting
                right_tree = parse_tree(tree.right, X[right_indices], y[right_indices], verbose)
            else: # right side is none (empty str)
                pred = predict_by_simulating(X, y, root_feature, True)
                right_tree = {"prediction": pred}

            return {"feature": root_feature, "left": left_tree, "right": right_tree}
        i += 1

    return None

def relabel(tree_dict):
    if not isinstance(tree_dict, dict):
        return tree_dict

    new_dict = {}
    for key, value in tree_dict.items():
        if key == "left":
            new_dict["false"] = relabel(value)
        elif key == "right":
            new_dict["true"] = relabel(value)
        else:
            new_dict[key] = relabel(value)
    return new_dict

# function to convert tree dictionary into visual representation of tree
# input: tree dict from parse_tree function
# output: nice visual representation (formatted string) of tree 
# according to the split function in the BCT code, left is false and right is true
def print_tree(tree, indent="", last=True):
    if tree is None:
        return
    
    prefix = "└── " if last else "├── "
    if "prediction" in tree:
        print(indent + prefix + "| Prediction: " + str(tree["prediction"]))
        return
    else:
        print(indent + prefix + str(tree["feature"]))
    indent += "    " if last else "│   "

    left, right = tree["left"], tree["right"]
    if left is not None:
        print_tree(left, indent, last=False)
    if right is not None:
        print_tree(right, indent, last=True)

# function to convert tree dictionary into binary classification tree form
# input: dictionary (representing tree, could be maptree or otherwise)
    # with keys "feature" representing feature they split on, 
    # "left" or "false" representing what happens when binary feature value is false
    # "right" or "true" representing what happens when binary feature value is true
    # "prediction" as prediction value only being present in leaves
        # currently it doesn't get preserved when converting though, 
        # you would need to refit the output tree of this function to the data again by calling fit function
# output: binary classification tree
def make_bct(tree_dict):
    if tree_dict is None:
        return BinaryClassificationTree()
    if "prediction" in tree_dict:
        return BinaryClassificationTree()
    else:
        if "left" in tree_dict:
            left = make_bct(tree_dict["left"])
        elif "false" in tree_dict:
            left = make_bct(tree_dict["false"])
        else:
            left=None
        if "right" in tree_dict:
            right = make_bct(tree_dict["right"])
        elif "true" in tree_dict:
            right = make_bct(tree_dict["true"])
        else:
            right=None
        return BinaryClassificationTree(feature=tree_dict["feature"], left=left, right=right)

if __name__ == "__main__":
    # default posterior params
    POSTERIOR = {
        'alpha': 0.95,
        'beta': 0.5,
        'rho': [2.5, 2.5],
        }
    
    test_treefarms_dict = {
    "false": {
        "complexity": 0.009999999776482582,
        "loss": 0.03938033804297447,
        "name": "recidivate-within-two-years:1",
        "prediction": 1
    },
    "feature": 7,
    "model_objective": 0.3748675286769867,
    "name": "juvenile-crimes:=0",
    "reference": 1,
    "relation": "==",
    "true": {
        "false": {
        "complexity": 0.009999999776482582,
        "loss": 0.21644708514213562,
        "name": "recidivate-within-two-years:1",
        "prediction": 0
        },
        "feature": 11,
        "name": "priors:>3",
        "reference": 1,
        "relation": "==",
        "true": {
        "complexity": 0.009999999776482582,
        "loss": 0.08904010057449341,
        "name": "recidivate-within-two-years:1",
        "prediction": 1
        },
        "type": "integral"
    },
    "type": "integral"
    }

    treefarms_dict_2 = {
    "feature": 11,
    "relation": "==",
    "reference": "true",
    "true": {
        "prediction": 1,
        "name": "Prediction"
    },
    "false": {
        "feature": 7,
        "relation": "==",
        "reference": "true",
        "true": {
        "prediction": 0,
        "name": "Prediction"
        },
        "false": {
        "feature": 0,
        "relation": "==",
        "reference": "true",
        "true": {
            "feature": 8,
            "relation": "==",
            "reference": "true",
            "true": {
            "prediction": 1,
            "name": "Prediction"
            },
            "false": {
            "prediction": 0,
            "name": "Prediction"
            }
        },
        "false": {
            "prediction": 1,
            "name": "Prediction"
        }
        }
    }
    }

    print(make_bct(treefarms_dict_2))
    print(parse_tree(make_bct(treefarms_dict_2)))
    print(print_tree(parse_tree(make_bct(treefarms_dict_2))))

    for i in range(15,12,-1): # running on the last 3 datasets, which finish in a reasonable time
    # pdb.set_trace()
        dataset_name = CP4IM_DATASET_NAMES[i]
        print(dataset_name)
        data = get_full_cp4im_dataset(dataset_name)
        X, y = data
        res = run(X, y)
        tree_1 = res["tree"]
        print(tree_1) # str rep of original tree
        print(parse_tree(tree_1)) # dict representation of tree
        print_tree(parse_tree(tree_1)) # visualization
        print(make_bct(parse_tree(tree_1))) # str rep of tree reconstructed from dict