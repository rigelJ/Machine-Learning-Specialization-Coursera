import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *

#_ = plot_entropy()

#Generate the data

#Ear Shape: Pointy = 1, Floppy = 0
#Face Shape: Round = 1, Not Round = 0
#Whiskers: Present = 1, Absent = 0

Features = ["Ear Shape","Face Shape","Whiskers"]

X_train = np.array([[1, 1, 1],
[0, 0, 1],
 [0, 1, 0],
 [1, 0, 1],
 [1, 1, 1],
 [1, 1, 0],
 [0, 0, 0],
 [1, 1, 0],
 [0, 1, 0],
 [0, 1, 0]])

y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])


##How to build a tree
# For one times to find the highest info_gain feature and recursive it to build a tree until over
# ONE Times need Get_best_split^| Split_node_indices^ / Cal_info_gain^|| Entropy^
# Recursive to get Build_Tree_Recursive
#split the X by feature

def Get_root_indices(y,root_index):
    for i in range(len(y)):
        root_index.append(i)

def split_node_indices(X,index_feature):
    left_indices=[]
    right_indices=[]
    for i in enumerate(X):
        if x[index_feature]==1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices,right_indices

#caculate the entropy
def entropy(p):

    if p==0 or p==1:
        return 0
    else:
        return -p*np.log2(p)-(1-p)*np.log2(1-p)

#Get the infom_gain
def Cal_inform_gain(X,y,node_indices):
    left_indices,right_indices = split_node_indices(X,y,node_entropy)
    w_left = len(left_indices)/sum(y)
    w_right = len(right_indices)/sum(y)
    p_left = sum(y[left_indices])/len(left_indices)
    p_right =sum(y[right_indices])/len(right_indices)

    weight_entropy = w_left*entropy(p_left)+w_right*entropy(p_right)

    node_entropy = entropy(sum(y)/len(y))

    Infor_gain = node_entropy - weight_entropy

    return Infor_gain

def Get_best_split(X,y,node_indices):
    num_features = X.shape[1]
    
    best_feature = -1

    max_info_gain = 0

    for feature in range(num_features):
        now_info_gain = Cal_inform_gain(X,y,node_indices)
        if now_info_gain > max_info_gain:
            max_info_gain = now_info_gain
            best_feature = feature
        else:
            continue
    
    return best_feature


def Build_tree_recursive(X,y,node_indices,branch_name,max_depth,current_depth,tree):
    if current_depth == max_depth:
        formatting = " "*current_depth + "-"*current_depth
        print(formatting,"%s leaf node with indices" % branch_name,node_indices)

    best_feature = Get_best_split(X,y,node_indices)

    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))

    left_indices,right_indices = split_node_indices(X,y,node_indices,best_feature)

    tree.append((left_indices,right_indices,best_feature))

    build_tree_recursive(X,y,left_indices,"Left",max_depth,current_depth+1,tree)
    build_tree_recursive(X,y,right_indices,"Right",max_depth,current_depth+1,tree)

    return tree


tree = []
root_index = []
Get_root_indices(y_train,root_index)
Build_tree_recursive(X_train,y_train,root_index,"Root",max_depth=0,current_depth=0,tree=tree)
