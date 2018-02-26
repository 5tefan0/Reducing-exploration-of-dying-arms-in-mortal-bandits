#-------------------------------------------------------------------------------
# Context generator
#-------------------------------------------------------------------------------
import numpy as np
#-------------------------------------------------------------------------------
def get_context(continuous = 0, distributions_continuous = [], binary = 0, p=[], \
discrete = 0, distributions_discrete = [], categorical = 0, categories = []):
    if continuous+binary+discrete+categorical == 0:
        return [1]
    CONTEXT = [False]*(continuous+binary+discrete+categorical)
    CONTEXT[:continuous]=create_continuous_context(continuous,distributions_continuous)
    CONTEXT[continuous:(continuous+binary)]=create_binary_context(binary,p)
    CONTEXT[(continuous+binary):(continuous+binary+discrete)]= create_discrete_context(discrete, distributions_discrete)
    CONTEXT[(continuous+binary+discrete):(continuous+binary+discrete+categorical)]=create_category_context(categorical, categories)
    return CONTEXT

def get_context_old(continuous = 0, distributions_continuous = [], binary = 0, p=[], \
discrete = 0, distributions_discrete = [], categorical = 0, categories = [], features = None):
    if continuous+binary+discrete+categorical == 0:
        return [1]
    CONTEXT = [False]*(continuous+binary+discrete+categorical)
    CONTEXT[:continuous]=create_continuous_context(continuous,distributions_continuous,features_list)
    CONTEXT[continuous:(continuous+binary)]=create_binary_context(binary,p,features_list)
    CONTEXT[(continuous+binary):(continuous+binary+discrete)]= create_discrete_context(discrete, distributions_discrete,features_list )
    CONTEXT[(continuous+binary+discrete):(continuous+binary+discrete+categorical)]=create_category_context(categorical, categories,features_list )
    return CONTEXT
#-------------------------------------------------------------------------------
def create_continuous_context(n,distributions_continuous, features_list = None):
    CONTINUOUS = [False] * n
    # If only one distribution is specified, it is used for all continuous variables
    # Example: distributions_continuous[['Normal', mu, sigma], ['Uniform', a, b]]
    if len(distributions_continuous) == 1:
        if distributions_continuous[0][0] == "Normal":
            mu = distributions_continuous[0][1]
            sigma = distributions_continuous[0][2]
            CONTINUOUS = np.random.normal(mu,sigma,n)
        elif distributions_continuous[0][0] == "Uniform":
            a = distributions_continuous[0][1]
            b = distributions_continuous[0][2]
            CONTINUOUS = np.random.uniform(a,b,n)
        else:
            print("Error")
            return
    # Example: distributions_continuous[['Normal', mu1, sigma1], ['Uniform', a, b]]
    elif len(distributions_continuous) == n:
        for i in range(n):
            if distributions_continuous[i][0] == "Normal":
                mu = distributions_continuous[i][1]
                sigma = distributions_continuous[i][2]
                CONTINUOUS[i] = np.random.normal(mu,sigma)
            elif distributions_continuous[i][0] == "Uniform":
                a = distributions_continuous[i][1]
                b = distributions_continuous[i][2]
                CONTINUOUS[i] = np.random.uniform(a,b)
            else:
                print("Error")
                return
    else:
        print("Error")
        return
    return CONTINUOUS
#-------------------------------------------------------------------------------
def create_binary_context(n,probabilities_of_success, features_list = None):
    # probabilities_of_success is a list of parameters for Bernoulli distributions
    if len(probabilities_of_success)==1:
        BINARY_VARIABLES = np.random.binomial(1,probabilities_of_success[0],n)
    elif len(probabilities_of_success)==n:
        BINARY_VARIABLES = [False] * n
        for i in range(n):
            BINARY_VARIABLES[i] = np.random.binomial(1,probabilities_of_success[i])
    else:
        print("Error")
        return
    return BINARY_VARIABLES
#-------------------------------------------------------------------------------
def create_discrete_context(n,distributions_discrete, features_list = None):
    ORDINAL = [False] * n
    # If only one distribution is specified, it is used for all continuous variables
    # Example: distributions_discrete[['Uniform', a, b]]
    if len(distributions_discrete) == 1:
        if distributions_discrete[0][0] == "Poisson":
            l = distributions_discrete[0][1]
            ORDINAL = np.random.poisson(l,n)
        elif distributions_discrete[0][0] == "Uniform":
            a = distributions_discrete[0][1]
            b = distributions_discrete[0][2]
            ORDINAL = np.random.randint(a,b,n)
        else:
            print("Error")
            return
    # Example: distributions_discrete[['Poisson', lambda], ['Uniform', a, b]]
    elif len(distributions_discrete) == n:
        for i in range(n):
            if distributions_discrete[i][0] == "Poisson":
                l = distributions_discrete[i][1]
                ORDINAL[i] = np.random.poisson(l)
            elif distributions_discrete[i][0] == "Uniform":
                a = distributions_discrete[i][1]
                b = distributions_discrete[i][2]
                ORDINAL[i] = np.random.randint(a,b)
            else:
                print("Error")
                return
    else:
        print("Error")
        return
    return ORDINAL
#-------------------------------------------------------------------------------
def create_category_context(n,categories, features_list = None):
    # Example: categories = [  [    ['A','B','C'], [0.1,0.5,0.4]   ], [    ['D','E','F'], [0.1,0.5,0.4]   ]  ]
    CATEGORICAL = [False] * n
    for i in xrange(n):
        P = categories[i][1]
        CATEGORICAL[i] = categories[i][0][np.argmax(np.random.multinomial(1,P,1))]
    return CATEGORICAL


# Example--
# x=get_context(continuous = 3, distributions_continuous = [["Normal", 3, 1],["Normal", 13, 1],["Uniform",0,10]], \
# binary = 5, p=[0.9,0,0,0.8,0.9], discrete = 3, distributions_discrete = [  ["Poisson",3], ["Poisson", 13], ["Uniform",0,10]  ], \
# categorical = 2, categories = [ [['A','B','C'], [0.1,0.5,0.4]], [['D','E','F'], [0.1,0.5,0.4]]  ])
#
# y=get_context(continuous = 3, distributions_continuous = [["Normal", 3, 1],["Normal", 13, 1],["Uniform",0,10]], \
# binary = 5, p=[0.9,0,0,0.8,0.9], discrete = 3, distributions_discrete = [  ["Poisson",3], ["Poisson", 13], ["Uniform",0,10]  ], \
# categorical = 2, categories = [ [['A','B','C'], [0.1,0.5,0.4]], [['D','E','F'], [0.1,0.5,0.4]]  ])














# old (slower)
# def get_context(continuous = 0, distributions_continuous = [], binary = 0, p=[], \
# discrete = 0, distributions_discrete = [], categorical = 0, categories = []):
#     CONTEXT = []                  #[False]*(continuous+ordinal+categorical) ##!!
#     CONTEXT.extend(create_continuous_context(continuous,distributions_continuous))
#     CONTEXT.extend(create_binary_context(binary,p))
#     CONTEXT.extend(create_discrete_context(discrete, distributions_discrete ))
#     CONTEXT.extend(create_category_context(categorical, categories ))
#     return CONTEXT
