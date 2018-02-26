import numpy as np
import pandas as pd
from contexts_generator import get_context, create_continuous_context, create_binary_context, create_discrete_context, create_category_context

def create_columns_names(continuous,binary,discrete,categorical):
    col_names = [False]*(continuous+binary+discrete+categorical)
    for i in range(continuous):
        col_names[i]='cont_'+str(i+1)
    for i in range(continuous,continuous+binary):
        col_names[i]='bin_'+str(i+1-continuous)
    for i in range(continuous+binary,continuous+binary+discrete):
        col_names[i]='dis_'+str(i+1-continuous-binary)
    for i in range(continuous+binary+discrete,continuous+binary+discrete+categorical):
        col_names[i]='cat_'+str(i+1-continuous-binary-discrete)
    return col_names

def create_context_df(df_size, contextual_space):
    X = [False] * df_size
    col_names = create_columns_names(len(contextual_space["continuous"]),len(contextual_space["binary"]),len(contextual_space["discrete"]),len(contextual_space["categorical"]))
    for i in range(df_size):
        X[i] = get_context(len(contextual_space["continuous"]), contextual_space["continuous"], len(contextual_space["binary"]), contextual_space["binary"], len(contextual_space["discrete"]), contextual_space["discrete"], len(contextual_space["categorical"]), contextual_space["categorical"])
    return pd.DataFrame(X,columns = col_names)

def create_context_df_from_arms_features(df_size, features_df, contextual_space, arms_features_space):
    X = [False] * df_size
    col_names = create_columns_names(continuous,binary,discrete,categorical)
    for i in range(df_size):
        X[i] = get_context(len(contextual_space["continuous"]), contextual_space["continuous"], len(contextual_space["binary"]), contextual_space["binary"], len(contextual_space["discrete"]), contextual_space["discrete"], len(contextual_space["categorical"]), contextual_space["categorical"], arms_features_space)
    return pd.DataFrame(X,columns = col_names)



def context_distance(context_1, context_2, contextual_space):
    # first check contexts come from the same space
    # if len(context_1) != len(context_2):
    #     print("Error: contexts must have same length.")
    #     return
    # compute difference between categorical features which will be added to the l_norm-distance of the other features
    different_categorical_features = 0
    continuous = len(contextual_space["continuous"])
    binary = len(contextual_space["binary"])
    discrete = len(contextual_space["discrete"])
    categorical = len(contextual_space["categorical"])
    norm_type = contextual_space["norm_type"]
    for i in range(categorical):
        if (context_1['cat_'+str(i+1)][0]!=context_2['cat_'+str(i+1)][0]):
            different_categorical_features += 1
    if norm_type == "l1":
        return sum(abs(np.array(context_1[0:(continuous+binary+discrete)]-context_2[0:(continuous+binary+discrete)]) )) + different_categorical_features
        #return sum(abs(np.array(context_1[0:(continuous+binary+discrete)]) - np.array(context_2[0:(continuous+binary+discrete)]) )) + different_categorical_features
    elif norm_type == "l2":
        return np.linalg.norm(np.array(context_1[0:(continuous+binary+discrete)]) - np.array(context_2[0:(continuous+binary+discrete)]), ord=2) + different_categorical_features
        #return np.linalg.norm(np.array(context_1.iloc[0][0:(continuous+binary+discrete)]) - np.array(context_2.iloc[0][0:(continuous+binary+discrete)]), ord=2) + different_categorical_features
    elif norm_type == "l0":
        return np.linalg.norm(np.array(context_1[0:(continuous+binary+discrete)]) - np.array(context_2[0:(continuous+binary+discrete)]), ord=0) + different_categorical_features
    else:
        print("Error: invalid norm.")
        return

def context_similarity(context_1, context_2, contextual_space, gamma = None):
    ''' Computes the similarity between two contexts with e^( - gamma * distance ) '''
    if gamma == None:
        gamma = 1.0/(len(contextual_space["continuous"])+len(contextual_space["binary"])+len(contextual_space["discrete"])+len(contextual_space["categorical"]))
    return np.exp(- gamma * context_distance(context_1, context_2, contextual_space))


def create_regression_coefficients_for_life(arms_features_space):
    if arms_features_space == { "continuous": [], "binary":[1.0], "discrete":[], "categorical": [], "norm_type": "l2"}:
        return None
    else:
        number_of_arms_features = len(arms_features_space["continuous"])+len(arms_features_space["binary"])+len(arms_features_space["discrete"] )+len(arms_features_space["categorical"])
        coeff_list = [False]*number_of_arms_features
        for j in range(number_of_arms_features):
            coeff_list[j] = np.random.uniform(-5.0,15.0)
        return coeff_list

def create_relevance(arms_features_space,contextual_space, method = "Uniform"):
    """ For each arm feature gives the relevance it has for each context  """
    number_of_arms_features = len(arms_features_space["continuous"])+len(arms_features_space["binary"])+len(arms_features_space["discrete"] )+len(arms_features_space["categorical"])
    number_of_contextual_features = len(contextual_space["continuous"])+len(contextual_space["binary"])+len(contextual_space["discrete"])+len(contextual_space["categorical"])
    contextual_relevance = {}
    for j in range(number_of_arms_features):
        contextual_relevance["arm_feature_"+str(j)] = [False]*number_of_contextual_features
        if method == "Uniform":
            for x in range(number_of_contextual_features):
                contextual_relevance["arm_feature_"+str(j)][x] = np.random.uniform(-1.0,1.0)
        elif method == "Normal":
            for x in range(number_of_contextual_features):
                contextual_relevance["arm_feature_"+str(j)][x] = min(max(np.random.normal(0.0,1.0),-1.0),1.0)
        else:
            print("No method found")
    return contextual_relevance    

# ---
# #Example
# X =create_context_df(10,continuous = 3, distributions_continuous = [["Normal", 3, 1],["Normal", 13, 1],["Uniform",0,10]], \
# binary = 5, p=[0.9,0,0,0.8,0.9], discrete = 3, distributions_discrete = [  ["Poisson",3], ["Poisson", 13], ["Uniform",0,10]  ], \
# categorical = 2, categories = [   [['A','B','C'],[0.1,0.5,0.4]],  [['D','E','F'], [0.1,0.5,0.4]]   ])
#
# new_y =create_context_df(1,continuous = 3, distributions_continuous = [["Normal", 3, 1],["Normal", 13, 1],["Uniform",0,10]], \
# binary = 5, p=[0.9,0,0,0.8,0.9], discrete = 3, distributions_discrete = [  ["Poisson",3], ["Poisson", 13], ["Uniform",0,10]  ], \
# categorical = 0, categories = [     ])

# to concatenate:  C=pd.concat([X,new_X])
# to reindex:      C=C.reset_index()
#---





# Example
#context_distance(X.iloc[1], X.iloc[2], continuous = 3, binary = 5, discrete = 3, categorical = 2, norm_type = "l1")



# Example
#context_similarity(X.iloc[1], X.iloc[2],continuous = 3, binary = 5, discrete = 3, categorical = 2, norm_type = "l1")
