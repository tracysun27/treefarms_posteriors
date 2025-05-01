from experiments.globals import CP4IM_DATASET_NAMES, SYNTH_NUM_TREES
import time
from typing import Tuple, Dict, Any
from maptree import search as maptree_search
from experiments.searchers.binary_classification_tree import BinaryClassificationTree
from experiments.searchers.maptree import *
from experiments.globals import get_stratified_k_folds_cp4im_dataset, get_full_cp4im_dataset, run_search, save_results
import pdb
import numpy as np
import json

# running on full dataset
POSTERIOR = {
    'alpha': 0.95,
    'beta': 0.5,
    'rho': [2.5, 2.5],
}

# function to convert parentheses tree into json for gosdt tree
# input: maptree (binary classification tree) class
# output: recursive dict structure with left and right for each split, 
# as well as prediction labels assigned based on mode
def parse_tree(tree, verbose = 0):
    # base case: if the number has parentheses directly to left and right of it,
    # like (1), then it's a leaf
    s = str(tree)
    if s.startswith('(') and s.endswith(')') and s[1:-1].isdigit():
        if verbose == 1:
            print("leaf:", int(s[1:-1]))
        return {"feature": int(s[1:-1]), 
                "left": {"feature": None, "left": None, "right": None, "prediction": np.argmax(tree.left.label_counts)}, 
                "right": {"feature": None, "left": None, "right": None, "prediction": np.argmax(tree.right.label_counts)}}

    s = s[1:-1]
    
    # recursive case: if the number has parentheses like this )10( then its got leaves
    i, balance = 0, 0
    while i < len(s):
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
            
            # Split left and right subtrees
            # string version
            # left_subtree = s[:i - len(str(root_feature))]
            # right_subtree = s[i:]
            left_subtree = tree.left
            right_subtree = tree.right
            if verbose == 1:
                print("Left:", left_subtree)
                print("Right:", right_subtree)
            # Recur on left and right
            left_tree = parse_tree(left_subtree)
            right_tree = parse_tree(right_subtree)
            
            # Return tree structure
            return {"feature": root_feature, "left": left_tree, "right": right_tree}
        i += 1

    return None 

# input: tree dict from parse_tree function
# output: nice visual representation (formatted string) of tree 
# according to the split function in the BCT code, left is false and right is true
def print_tree(tree, indent="", last=True):
    if tree is None:
        return
    
    prefix = "└── " if last else "├── "
    if "prediction" in tree:
        print(indent + prefix + "| Prediction: " + str(tree["prediction"]))
    else:
        print(indent + prefix + str(tree["feature"]))
    indent += "    " if last else "│   "
    
    left, right = tree["left"], tree["right"]
    if left is not None or right is not None:
        # Print left subtree
        print_tree(left, indent, last=False)
        # Print right subtree
        print_tree(right, indent, last=True)

# example output of the print tree function on maptrees
tree_str1 = "(6)"
tree_str2 = "((18)6(31))"
tree_str3 = "(((19((25(22))1(10)))7((25(16))1(4)))13((20((26(23))2(11)))8((26(17))2(5))))"

# print(parse_tree(tree_str1))
# # Output: {'feature': 6, 'left': None, 'right': None}

# print(parse_tree(tree_str2))
# # Output: {'feature': 6, 'left': {'feature': 8, 'left': None, 'right': None}, 'right': {'feature': 31, 'left': None, 'right': None}}

# print(parse_tree(tree_str3))

# now go backwards: construct a maptree class from the dict tree structure 
# input: dict tree structure
# output: tree of the maptree class
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

# getting posterior
# print(mytree.log_posterior(X, y, **POSTERIOR))
# print(mytree)

# getting predictions
# print(mytree.predict(X))

# varun's code
def train_maptree(X_train, y_train, X_test, y_test, return_train_loss = False, **kwargs):
    num_expansions = kwargs['num_expansions']
    X_train, y_train, X_test, y_test = X_train.values, y_train.values, X_test.values, y_test.values
    alpha = 0.95
    beta = 0.01
    rho = (25, 25)

    start = time.perf_counter()
    time_limit = 600
    sol = maptree_search(X_train, y_train, alpha,
                         beta, rho, num_expansions, time_limit)
    tree = BinaryClassificationTree.parse(sol.tree)
    tree.fit(X_train, y_train)
    end = time.perf_counter()
    elapsed_time = end - start
    y_pred = tree.predict(X_test)
    acc = (y_pred == y_test).mean()
    # n_leaves = get_num_leaves_maptree(tree)
    loss = 1 - acc
    train_acc = np.mean(tree.predict(X_train) == y_train)
    train_loss = 1 - train_acc

    if return_train_loss:
        return elapsed_time, loss, train_loss
    return elapsed_time, loss, None


# sample features randomly to run on since it takes too long?

# running using k fold cross validation

# ok update it seemed to work for numbers 13, 14, and 15 so maybe i just had to wait

# for j in range(15):
#     dataset_name = CP4IM_DATASET_NAMES[j]
#     print(dataset_name)
#     for i, fold in enumerate(get_stratified_k_folds_cp4im_dataset(dataset_name)):
#             print(f"Fold: {i}")
#             X_train, y_train, X_test, y_test = fold

#             res = run(X_train, y_train)
#             mytree = res["tree"]

#             print(mytree.log_posterior(X_train, y_train, 
#                                     alpha = 0.95,
#                                     beta = 0.5,
#                                     rho = (2.0, 5.0)))

# I THINK I GOT IT????
# i need to check what alpha, beta, and rho really mean again
# anddd it only returns 1 number (idk why)
# but this is a good step!