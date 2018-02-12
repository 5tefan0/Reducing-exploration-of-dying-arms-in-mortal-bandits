import numpy as np
from df_context_generator import create_columns_names, create_context_df, context_distance, context_similarity

def create_arms_with_context(number_of_arms, arms_behavior, policies, contextual_space, arms_features_space, mortal, life_method, arms_p_and_type={}, list_true_regression_coefficients = None): ## add option for custom arms_behavior
    arms = [False]*number_of_arms
    for i in range(number_of_arms):
        arms[i] = {}
        arms[i]["arm_id"] = 'arm_'+str(i)
        arms[i]["distribution"] = arms_behavior
        # create combination for each context to decide which influences more the arm [0,1] for each context
        # then for each context also decide where is the center of the peak
        if arms_behavior == "Bernoulli":
            # If behavior is Bernoulli we randomly assign to each arm its probability of success
            arms[i]["p"] = np.random.uniform(0.0,1.0) # this comes from the combination
            arms[i]["features"] = create_context_df(1,arms_features_space)
            arms[i]["features"] = arms[i]["features"].iloc[0]
        elif arms_behavior == "Bernoulli_low":
            # If behavior is Bernoulli we randomly assign to each arm its probability of success
            arms[i]["p"] = np.random.uniform(0.0,0.2) # this comes from the combination
            higher_p = np.random.binomial(1,0.15)
            if higher_p == 1:
                arms[i]["p"] = np.random.uniform(0.8,1.0)
            arms[i]["features"] = create_context_df(1,arms_features_space)
            arms[i]["features"] = arms[i]["features"].iloc[0]
        elif arms_behavior == "Truncated_Normal":
            # If behavior is Normal we randomly assign to each arm mean and std
            arms[i]["mu"] = np.random.uniform(0.5,1.0) # mean of the normal
            arms[i]["sigma"] = 0.1 #np.random.uniform(0.0,5.0)
            arms[i]["features"] = create_context_df(1,arms_features_space)
            arms[i]["features"] = arms[i]["features"].iloc[0]
        else:
            print("Error: Distribution not found")
            return
        for policy in policies: # the following parameters depend on the policy:
            arms[i]["total_pulls_"+policy]   = 0
            arms[i]["rewards_"+policy]       = []
            arms[i]["mean_reward_"+policy]   = 0.0
            arms[i]["rounds_pulled_"+policy] = []
            if policy in ["LinUCB","LinUCB_greedy","LinUCB_life_known","LinUCB_soft"]:
                d = len(contextual_space["continuous"])+len(contextual_space["binary"])+len(contextual_space["discrete"]) # I'm leaving out categorical not suited for this
                arms[i]["A_"+policy]     = np.identity(d)
                arms[i]["b_"+policy]     = np.array([np.zeros(d)]).T # column vector
                arms[i]["theta_"+policy] = np.array([np.zeros(d)]).T #same as np.dot(arms[i]["A_"+policy],arms[i]["b_"+policy])
                arms[i]["UCB_"+policy]   = 0.0
            elif policy in ["LinUCB_bayesian","LinUCB_bayesian_greedy"]:
                d = len(contextual_space["continuous"])+len(contextual_space["binary"])+len(contextual_space["discrete"])
                arms[i]["lambda_"+policy]= 5.0
                arms[i]["D_"+policy]     = np.empty((0,d)) # n*d
                arms[i]["b_"+policy]     = np.array([np.zeros(d)]).T
                arms[i]["A_"+policy]     = np.identity(d)
                arms[i]["theta_"+policy] = np.array([np.zeros(d)]).T
                arms[i]["UCB_"+policy]   = 0.0
    if mortal > 0:   ##! add new newborn_arm
        # add life from a Poisson distribution         ## or inference!
        if life_method == "Poisson":
            for i in range(number_of_arms):
                arms[i]["life"] = np.random.poisson(mortal)
        elif life_method == "long_and_short":
            for i in range(number_of_arms):
                long_life = np.random.binomial(1, 0.5, size=None)
                if long_life == 1:
                    arms[i]["life"] = np.random.poisson(mortal*3)
                else:
                    arms[i]["life"] = np.random.poisson(mortal/3)
        elif life_method == "factor_4":
            for i in range(number_of_arms):
                long_life = np.random.binomial(1, 0.5, size=None)
                if long_life == 1:
                    arms[i]["life"] = np.random.poisson(mortal*4)
                else:
                    arms[i]["life"] = np.random.poisson(mortal/4)
        elif life_method == "few_long":
            for i in range(number_of_arms):
                long_life = np.random.binomial(1, 0.2, size=None)
                if long_life == 1:
                    arms[i]["life"] = np.random.poisson(mortal*3)
                else:
                    arms[i]["life"] = np.random.poisson(mortal/3)
        elif life_method == "set":
            for i in range(number_of_arms):
                arms[i] = set_arms_life(arms[i], arms_p_and_type["p"], arms_p_and_type["type"] , list_true_regression_coefficients = list_true_regression_coefficients)
        # write
        for i in range(number_of_arms):
            file_record=open("arms_history.txt","a")
            file_record.write(arms[i]["arm_id"]+" is born with life "+ str(arms[i]["life"])+"\n")
            file_record.close()
    return arms

def set_arms_life(arm_j, probabilities_for_each_type, distributions, list_true_regression_coefficients = None):
    if list_true_regression_coefficients == None:
        draw = np.random.multinomial(1, probabilities_for_each_type, size=1)
        type = np.argmax(draw)
        if distributions[type][0] == "Poisson":
            arm_j["life"] = np.random.poisson(distributions[type][1])
            arm_j["life_predicted"] = False
        return arm_j
    else:
        arm_j["life"] = np.dot(np.array(arm_j["features"]),np.array(list_true_regression_coefficients))
        arm_j["life_predicted"] = False
        return arm_j


def add_newborn_arms_with_context(arms,dead_arms,total_of_arms_born, poisson_parameter, \
arms_behavior, policies, arms_features_space, mortal, contextual_space,  life_method, arms_p_and_type={}, initialization_method = None,  list_true_regression_coefficients = None): # det how many and initialize
    number_of_newborn_arms = np.random.poisson(poisson_parameter)
    index_of_new_arm = total_of_arms_born
    if number_of_newborn_arms == 0:
        return arms, number_of_newborn_arms
    else:
        for new_arm in range(number_of_newborn_arms):
            arms.append({})
            arms[-1]["arm_id"] = 'arm_'+str(index_of_new_arm)
            arms[-1]["distribution"] = arms_behavior
            if arms_behavior == "Bernoulli":
                # If behavior is Bernoulli we randomly assign to each arm its probability of success
                arms[-1]["p"] = np.random.uniform(0.0,1.0)
                arms[-1]["features"] = create_context_df(1,arms_features_space)
                arms[-1]["features"]=arms[-1]["features"].iloc[0]
            elif arms_behavior == "Bernoulli_low":
                # If behavior is Bernoulli we randomly assign to each arm its probability of success
                arms[-1]["p"] = np.random.uniform(0.0,0.2) #this comes from the combination
                higher_p = np.random.binomial(1,0.15)
                if higher_p == 1:
                    arms[-1]["p"] = np.random.uniform(0.8,1.0)
                arms[-1]["features"] = create_context_df(1,arms_features_space)
                arms[-1]["features"]=arms[-1]["features"].iloc[0]
            elif arms_behavior == "Truncated_Normal":
                # If behavior is Normal we randomly assign to each arm mean and std
                arms[-1]["mu"] = np.random.uniform(0.5,1.0) # mean of the normal
                arms[-1]["sigma"] = 0.1 #np.random.uniform(0.0,2.5) # std of the normal
                arms[-1]["features"] = create_context_df(1,arms_features_space)
                arms[-1]["features"]=arms[-1]["features"].iloc[0]
            else:
                arms[-1]["p"] = "set parameters"
            if initialization_method == None:
                for policy in policies: # the following parameters depend on the policy:
                    arms[-1]["total_pulls_"+policy]   = 0 ##!ucb
                    arms[-1]["rewards_"+policy]       = []
                    arms[-1]["mean_reward_"+policy]   = 0.0 ##! vp
                    arms[-1]["rounds_pulled_"+policy] = []
                    if policy in ["LinUCB","LinUCB_greedy","LinUCB_life_known","LinUCB_soft"]:
                        d = len(contextual_space["continuous"])+len(contextual_space["binary"])+len(contextual_space["discrete"]) # I'm leaving out categorical not suited for this
                        arms[-1]["A_"+policy]     = np.identity(d)
                        arms[-1]["b_"+policy]     = np.array([np.zeros(d)]).T
                        arms[-1]["theta_"+policy] = np.array([np.zeros(d)]).T #same as np.dot(arms[i]["A_"+policy],arms[i]["b_"+policy])
                        arms[-1]["UCB_"+policy]   = 0.0
                    elif policy in ["LinUCB_bayesian","LinUCB_bayesian_greedy"]:
                        d = len(contextual_space["continuous"])+len(contextual_space["binary"])+len(contextual_space["discrete"])
                        arms[-1]["lambda_"+policy]= 5.0
                        arms[-1]["D_"+policy]     = np.empty((0,d)) # n*d
                        arms[-1]["b_"+policy]     = np.array([np.zeros(d)]).T
                        arms[-1]["A_"+policy]     = arms[-1]["A_"+policy]     = np.identity(d)
                        arms[-1]["theta_"+policy] = np.array([np.zeros(d)]).T
                        arms[-1]["UCB_"+policy]   = 0.0
            elif initialization_method == "Contextual":
                for policy in policies: # the following parameters depend on the policy:
                    arms[-1]["total_pulls_"+policy]   = 1 ##!ucb
                    arms[-1]["rewards_"+policy]       = []
                    arms[-1]["mean_reward_"+policy]   = get_mean_of("mean_reward",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False,contextual=True,continuous = continuous, binary = binary, discrete = discrete, categorical = categorical)
                    arms[-1]["rounds_pulled_"+policy] = []
                    if policy in ["LinUCB","LinUCB_greedy","LinUCB_life_known","LinUCB_soft"]:
                        d = len(contextual_space["continuous"])+len(contextual_space["binary"])+len(contextual_space["discrete"]) # I'm leaving out categorical not suited for this
                        arms[-1]["A_"+policy]     = np.identity(d)
                        arms[-1]["b_"+policy]     = get_mean_of("b",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False,contextual=True,continuous = continuous, binary = binary, discrete = discrete, categorical = categorical)
                        arms[-1]["theta_"+policy] = get_mean_of("theta",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False,contextual=True,continuous = continuous, binary = binary, discrete = discrete, categorical = categorical) #same as np.dot(arms[i]["A_"+policy],arms[i]["b_"+policy])
                        arms[-1]["UCB_"+policy]   = get_mean_of("UCB",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False,contextual=True,continuous = continuous, binary = binary, discrete = discrete, categorical = categorical)
                    elif policy in ["LinUCB_bayesian","LinUCB_bayesian_greedy"]:
                        d = len(contextual_space["continuous"])+len(contextual_space["binary"])+len(contextual_space["discrete"])
                        arms[-1]["lambda_"+policy]= 5.0
                        arms[-1]["D_"+policy]     = np.empty((0,d)) # n*d
                        arms[-1]["b_"+policy]     = get_mean_of("b",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False,contextual=True,continuous = continuous, binary = binary, discrete = discrete, categorical = categorical)
                        arms[-1]["A_"+policy]     = np.identity(d)
                        arms[-1]["theta_"+policy] = get_mean_of("theta",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False,contextual=True,continuous = continuous, binary = binary, discrete = discrete, categorical = categorical)
                        arms[-1]["UCB_"+policy]   = get_mean_of("UCB",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False,contextual=True,continuous = continuous, binary = binary, discrete = discrete, categorical = categorical)
            elif initialization_method == "Mean":
                for policy in policies: # the following parameters depend on the policy:
                    arms[-1]["total_pulls_"+policy]   = 1 ##!ucb
                    arms[-1]["rewards_"+policy]       = []
                    arms[-1]["mean_reward_"+policy]   = get_mean_of("mean_reward",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False)
                    arms[-1]["rounds_pulled_"+policy] = []
                    if policy in ["LinUCB","LinUCB_greedy","LinUCB_life_known","LinUCB_soft"]:
                        d = len(contextual_space["continuous"])+len(contextual_space["binary"])+len(contextual_space["discrete"]) # I'm leaving out categorical not suited for this
                        arms[-1]["A_"+policy]     = np.identity(d)
                        arms[-1]["b_"+policy]     = get_mean_of("b",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False)
                        arms[-1]["theta_"+policy] = get_mean_of("theta",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False) #same as np.dot(arms[i]["A_"+policy],arms[i]["b_"+policy])
                        arms[-1]["UCB_"+policy]   = get_mean_of("UCB",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False)
                    elif policy in ["LinUCB_bayesian","LinUCB_bayesian_greedy"]:
                        d = len(contextual_space["continuous"])+len(contextual_space["binary"])+len(contextual_space["discrete"])
                        arms[-1]["lambda_"+policy]= 5.0
                        arms[-1]["D_"+policy]     = np.empty((0,d)) # n*d
                        arms[-1]["b_"+policy]     = get_mean_of("b",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False)
                        arms[-1]["A_"+policy]     = np.identity(d)
                        arms[-1]["theta_"+policy] = get_mean_of("theta",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False)
                        arms[-1]["UCB_"+policy]   = get_mean_of("UCB",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False)
            elif initialization_method == "Weighted_Mean":
                for policy in policies: # the following parameters depend on the policy:
                    arms[-1]["total_pulls_"+policy]   = 1 ##!ucb
                    arms[-1]["rewards_"+policy]       = []
                    arms[-1]["mean_reward_"+policy]   = get_mean_of("mean_reward",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=False)
                    arms[-1]["rounds_pulled_"+policy] = []
                    if policy in ["LinUCB","LinUCB_greedy","LinUCB_life_known","LinUCB_soft"]:
                        d = len(contextual_space["continuous"])+len(contextual_space["binary"])+len(contextual_space["discrete"]) # I'm leaving out categorical not suited for this
                        arms[-1]["A_"+policy]     = np.identity(d)
                        arms[-1]["b_"+policy]     = get_mean_of("b",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=False)
                        arms[-1]["theta_"+policy] = get_mean_of("theta",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=False) #same as np.dot(arms[i]["A_"+policy],arms[i]["b_"+policy])
                        arms[-1]["UCB_"+policy]   = get_mean_of("UCB",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=False)
                    elif policy in ["LinUCB_bayesian","LinUCB_bayesian_greedy"]:
                        d = len(contextual_space["continuous"])+len(contextual_space["binary"])+len(contextual_space["discrete"])
                        arms[-1]["lambda_"+policy]= 5.0
                        arms[-1]["D_"+policy]     = np.empty((0,d)) # n*d
                        arms[-1]["b_"+policy]     = get_mean_of("b",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=False)
                        arms[-1]["A_"+policy]     = np.identity(d)
                        arms[-1]["theta_"+policy] = get_mean_of("theta",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=False)
                        arms[-1]["UCB_"+policy]   = get_mean_of("UCB",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=False)
            elif initialization_method == "Mean_Std":
                for policy in policies: # the following parameters depend on the policy:
                    arms[-1]["total_pulls_"+policy]   = 1 ##!ucb
                    arms[-1]["rewards_"+policy]       = []
                    M,v   = get_mean_of("mean_reward",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=True)
                    arms[-1]["mean_reward_"+policy] = M + v
                    arms[-1]["rounds_pulled_"+policy] = []
                    if policy in ["LinUCB","LinUCB_greedy","LinUCB_life_known","LinUCB_soft"]:
                        d = len(contextual_space["continuous"])+len(contextual_space["binary"])+len(contextual_space["discrete"]) # I'm leaving out categorical not suited for this
                        arms[-1]["A_"+policy]     = np.identity(d)
                        arms[-1]["b_"+policy] ,v    = get_mean_of("b",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=True)
                        M,v = get_mean_of("theta",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=True) #same as np.dot(arms[i]["A_"+policy],arms[i]["b_"+policy])
                        arms[-1]["theta_"+policy] = M+v
                        arms[-1]["UCB_"+policy],v   = get_mean_of("UCB",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=True)
                    elif policy in ["LinUCB_bayesian","LinUCB_bayesian_greedy"]:
                        d = len(contextual_space["continuous"])+len(contextual_space["binary"])+len(contextual_space["discrete"])
                        arms[-1]["lambda_"+policy]= 5.0
                        arms[-1]["D_"+policy]     = np.empty((0,d)) # n*d
                        arms[-1]["b_"+policy]  ,v   = get_mean_of("b",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=True)
                        arms[-1]["A_"+policy]     = np.identity(d)
                        M,v = get_mean_of("theta",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=True)
                        arms[-1]["theta_"+policy] = M+v
                        arms[-1]["UCB_"+policy],v   = get_mean_of("UCB",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=True)
            elif initialization_method == "Weighted_Mean_Std":
                for policy in policies: # the following parameters depend on the policy:
                    arms[-1]["total_pulls_"+policy]   = 1 ##!ucb
                    arms[-1]["rewards_"+policy]       = []
                    M,v  = get_mean_of("mean_reward",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=True)
                    arms[-1]["mean_reward_"+policy] = M+v
                    arms[-1]["rounds_pulled_"+policy] = []
                    if policy in ["LinUCB","LinUCB_greedy","LinUCB_life_known","LinUCB_soft"]:
                        d = len(contextual_space["continuous"])+len(contextual_space["binary"])+len(contextual_space["discrete"]) # I'm leaving out categorical not suited for this
                        arms[-1]["A_"+policy]     = np.identity(d)
                        arms[-1]["b_"+policy]  ,v   = get_mean_of("b",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=True)
                        M,v = get_mean_of("theta",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=True) #same as np.dot(arms[i]["A_"+policy],arms[i]["b_"+policy])
                        arms[-1]["theta_"+policy] = M + v
                        arms[-1]["UCB_"+policy] ,v  = get_mean_of("UCB",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=True)
                    elif policy in ["LinUCB_bayesian","LinUCB_bayesian_greedy"]:
                        d = len(contextual_space["continuous"])+len(contextual_space["binary"])+len(contextual_space["discrete"])
                        arms[-1]["lambda_"+policy]= 5.0
                        arms[-1]["D_"+policy]     = np.empty((0,d)) # n*d
                        arms[-1]["b_"+policy] ,v    = get_mean_of("b",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=True)
                        arms[-1]["A_"+policy]     = np.identity(d)
                        M,v= get_mean_of("theta",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=True)
                        arms[-1]["theta_"+policy] = M + v
                        arms[-1]["UCB_"+policy] ,v  = get_mean_of("UCB",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=True)
            #arms[-1]["life"] = np.random.poisson(mortal, 1)
            if life_method == "Poisson":
                arms[-1]["life"] = np.random.poisson(mortal,1)
            elif life_method == "set":
                arms[-1] = set_arms_life(arms[-1], arms_p_and_type["p"], arms_p_and_type["type"] , list_true_regression_coefficients = list_true_regression_coefficients)
            elif life_method == "long_and_short":
                long_life = np.random.binomial(1, 0.5, size=None)
                if long_life == 1:
                    arms[-1]["life"] = np.random.poisson(mortal*3.0)
                else:
                    arms[-1]["life"] = np.random.poisson(mortal/3)
            elif life_method == "factor_4":
                long_life = np.random.binomial(1, 0.5, size=None)
                if long_life == 1:
                    arms[-1]["life"] = np.random.poisson(mortal*4)
                else:
                    arms[-1]["life"] = np.random.poisson(mortal/4)
            elif life_method == "few_long":
                long_life = np.random.binomial(1, 0.2, size=None)
                if long_life == 1:
                    arms[-1]["life"] = np.random.poisson(mortal*3)
                else:
                    arms[-1]["life"] = np.random.poisson(mortal/3)
            file_record=open("arms_history.txt","a")
            file_record.write(arms[-1]["arm_id"]+" is born with life "+ str(arms[-1]["life"])+"\n")
            file_record.close()
            index_of_new_arm += 1
        return arms, number_of_newborn_arms

################################################################################
################################################################################
################################################################################
################################################################################


# def create_arms_with_context_old(number_of_arms, arms_behavior, policies, mortal = 0, continuous = 0, distributions_continuous = [], \
# binary = 0, p=[], discrete = 0, distributions_discrete = [], \
# categorical = 0, categories = []): ## add option for custom arms_behavior
#     arms = [False]*number_of_arms
#     for i in range(number_of_arms):
#         arms[i] = {}
#         arms[i]["arm_id"] = 'arm_'+str(i)
#         arms[i]["distribution"] = arms_behavior
#         # create combination for each context to decide which influences more the arm [0,1] for each context
#         # then for each context also decide where is the center of the peak
#         if arms_behavior == "Bernoulliold":
#             # If behavior is Bernoulli we randomly assign to each arm its probability of success
#             arms[i]["p"] = np.random.uniform(0.0,1.0) #this comes from the combination
#             arms[i]["center"] = create_context_df(1,continuous, distributions_continuous, binary, p, \
#             discrete, distributions_discrete, categorical, categories)
#             arms[i]["center"]=arms[i]["center"].iloc[0]
#         elif arms_behavior == "Bernoulli_low":
#             # If behavior is Bernoulli we randomly assign to each arm its probability of success
#             arms[i]["p"] = np.random.uniform(0.0,0.2) #this comes from the combination
#             higher_p = np.random.binomial(1,0.15)
#             if higher_p == 1:
#                 arms[i]["p"] = np.random.uniform(0.8,1.0)
#             arms[i]["center"] = create_context_df(1,continuous, distributions_continuous, binary, p, \
#             discrete, distributions_discrete, categorical, categories)
#             arms[i]["center"]=arms[i]["center"].iloc[0]
#         elif arms_behavior == "Bernoulli":
#             # If behavior is Bernoulli we randomly assign to each arm its probability of success
#             arms[i]["p"] = np.random.uniform(0.0,1.0) #this comes from the combination
#             arms[i]["center"] = create_context_df(1,continuous, distributions_continuous, binary, p, \
#             discrete, distributions_discrete, categorical, categories)
#             arms[i]["center"]=arms[i]["center"].iloc[0]
#         elif arms_behavior == "Truncated_Normal":
#             # If behavior is Normal we randomly assign to each arm mean and std
#             arms[i]["mu"] = np.random.uniform(0.5,1.0) # mean of the normal
#             arms[i]["sigma"] = 0.1 #np.random.uniform(0.0,5.0)
#             arms[i]["center"] = create_context_df(1,continuous, distributions_continuous, binary, p, \
#             discrete, distributions_discrete, categorical, categories)
#             arms[i]["center"]=arms[i]["center"].iloc[0]
#         else:
#             print("Error: Distribution not found")
#             return
#         for policy in policies: # the following parameters depend on the policy:
#             arms[i]["total_pulls_"+policy]   = 0
#             arms[i]["rewards_"+policy]       = []
#             arms[i]["mean_reward_"+policy]   = 0.0
#             arms[i]["rounds_pulled_"+policy] = []
#             if policy in ["LinUCB","LinUCB_greedy"]:
#                 d = continuous+binary+discrete # I'm leaving out categorical not suited for this
#                 arms[i]["A_"+policy]     = np.identity(d)
#                 arms[i]["b_"+policy]     = np.array([np.zeros(d)]).T # column vector
#                 arms[i]["theta_"+policy] = np.array([np.zeros(d)]).T #same as np.dot(arms[i]["A_"+policy],arms[i]["b_"+policy])
#                 arms[i]["UCB_"+policy]   = 0.0
#             elif policy in ["LinUCB_bayesian","LinUCB_bayesian_greedy"]:
#                 d = continuous+binary+discrete
#                 arms[i]["lambda_"+policy]= 5.0
#                 arms[i]["D_"+policy]     = np.empty((0,d)) # n*d
#                 arms[i]["b_"+policy]     = np.array([np.zeros(d)]).T
#                 arms[i]["A_"+policy]     = np.identity(d)
#                 arms[i]["theta_"+policy] = np.array([np.zeros(d)]).T
#                 arms[i]["UCB_"+policy]   = 0.0
#     if mortal > 0:   ##! add new newborn_arm
#         # add life from a Poisson distribution         ## or inference!
#         for i in range(number_of_arms):
#             arms[i]["life"] = np.random.poisson(mortal)
#     return arms

# def add_similarities_between_arms(new_arms, existing_arms, NonContextual= False): #both inputs are a list of dictionaries
#     if NonContextual == True:
#         for i in range(len(new_arms)):
#             new_arms[i]["Similarity"]
#             for j in range(len(existing_arms)):
#                 new_arms[i]["Similarity"] = np.random.uniform(0.0,1.0)


#initialization_method = "None", "Mean", "Weighted_Mean", "Mean_Std", "Weighted_Mean_Std", "Similarity"

# def add_newborn_arms_with_context_old2(arms,dead_arms,total_of_arms_born, poisson_parameter, \
# arms_behavior, policies, mortal, continuous = 0, distributions_continuous = [], \
# binary = 0, p=[], discrete = 0, distributions_discrete = [], \
# categorical = 0, categories = [], \
# initialization_method = None): # det how many and initialize
#     number_of_newborn_arms = np.random.poisson(poisson_parameter)
#     index_of_new_arm = total_of_arms_born
#     if number_of_newborn_arms == 0:
#         return arms, number_of_newborn_arms
#     else:
#         for new_arm in range(number_of_newborn_arms):
#             arms.append({})
#             arms[-1]["arm_id"] = 'arm_'+str(index_of_new_arm)
#             arms[-1]["distribution"] = arms_behavior
#             arms[-1]["life"] = np.random.poisson(mortal, 1)
#             if arms_behavior == "Bernoulli":
#                 # If behavior is Bernoulli we randomly assign to each arm its probability of success
#                 arms[-1]["p"] = np.random.uniform(0.5,1.0)
#                 arms[-1]["center"] = create_context_df(1,continuous, distributions_continuous, binary, p, \
#                 discrete, distributions_discrete, categorical, categories)
#                 arms[-1]["center"]=arms[-1]["center"].iloc[0]
#             elif arms_behavior == "Bernoulli_low":
#                 # If behavior is Bernoulli we randomly assign to each arm its probability of success
#                 arms[-1]["p"] = np.random.uniform(0.0,0.2) #this comes from the combination
#                 higher_p = np.random.binomial(1,0.15)
#                 if higher_p == 1:
#                     arms[-1]["p"] = np.random.uniform(0.8,1.0)
#                 arms[-1]["center"] = create_context_df(1,continuous, distributions_continuous, binary, p, \
#                 discrete, distributions_discrete, categorical, categories)
#                 arms[-1]["center"]=arms[-1]["center"].iloc[0]
#             elif arms_behavior == "Truncated_Normal":
#                 # If behavior is Normal we randomly assign to each arm mean and std
#                 arms[-1]["mu"] = np.random.uniform(0.5,1.0) # mean of the normal
#                 arms[-1]["sigma"] = 0.1 #np.random.uniform(0.0,2.5) # std of the normal
#                 arms[-1]["center"] = create_context_df(1,continuous, distributions_continuous, binary, p, \
#                 discrete, distributions_discrete, categorical, categories)
#                 arms[-1]["center"]=arms[-1]["center"].iloc[0]
#             else:
#                 arms[-1]["p"] = "set parameters"
#             if initialization_method == None:
#                 for policy in policies: # the following parameters depend on the policy:
#                     arms[-1]["total_pulls_"+policy]   = 0 ##!ucb
#                     arms[-1]["rewards_"+policy]       = []
#                     arms[-1]["mean_reward_"+policy]   = 0.0 ##! vp
#                     arms[-1]["rounds_pulled_"+policy] = []
#                     if policy in ["LinUCB","LinUCB_greedy"]:
#                         d = continuous+binary+discrete # I'm leaving out categorical not suited for this
#                         arms[-1]["A_"+policy]     = np.identity(d)
#                         arms[-1]["b_"+policy]     = np.array([np.zeros(d)]).T
#                         arms[-1]["theta_"+policy] = np.array([np.zeros(d)]).T #same as np.dot(arms[i]["A_"+policy],arms[i]["b_"+policy])
#                         arms[-1]["UCB_"+policy]   = 0.0
#                     elif policy in ["LinUCB_bayesian","LinUCB_bayesian_greedy"]:
#                         d = continuous+binary+discrete
#                         arms[-1]["lambda_"+policy]= 5.0
#                         arms[-1]["D_"+policy]     = np.empty((0,d)) # n*d
#                         arms[-1]["b_"+policy]     = np.array([np.zeros(d)]).T
#                         arms[-1]["A_"+policy]     = arms[-1]["A_"+policy]     = np.identity(d)
#                         arms[-1]["theta_"+policy] = np.array([np.zeros(d)]).T
#                         arms[-1]["UCB_"+policy]   = 0.0
#             elif initialization_method == "Contextual":
#                 for policy in policies: # the following parameters depend on the policy:
#                     arms[-1]["total_pulls_"+policy]   = 1 ##!ucb
#                     arms[-1]["rewards_"+policy]       = []
#                     arms[-1]["mean_reward_"+policy]   = get_mean_of("mean_reward",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False,contextual=True,continuous = continuous, binary = binary, discrete = discrete, categorical = categorical)
#                     arms[-1]["rounds_pulled_"+policy] = []
#                     if policy in ["LinUCB","LinUCB_greedy"]:
#                         d = continuous+binary+discrete # I'm leaving out categorical not suited for this
#                         arms[-1]["A_"+policy]     = np.identity(d)
#                         arms[-1]["b_"+policy]     = get_mean_of("b",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False,contextual=True,continuous = continuous, binary = binary, discrete = discrete, categorical = categorical)
#                         arms[-1]["theta_"+policy] = get_mean_of("theta",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False,contextual=True,continuous = continuous, binary = binary, discrete = discrete, categorical = categorical) #same as np.dot(arms[i]["A_"+policy],arms[i]["b_"+policy])
#                         arms[-1]["UCB_"+policy]   = get_mean_of("UCB",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False,contextual=True,continuous = continuous, binary = binary, discrete = discrete, categorical = categorical)
#                     elif policy in ["LinUCB_bayesian","LinUCB_bayesian_greedy"]:
#                         d = continuous+binary+discrete
#                         arms[-1]["lambda_"+policy]= 5.0
#                         arms[-1]["D_"+policy]     = np.empty((0,d)) # n*d
#                         arms[-1]["b_"+policy]     = get_mean_of("b",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False,contextual=True,continuous = continuous, binary = binary, discrete = discrete, categorical = categorical)
#                         arms[-1]["A_"+policy]     = np.identity(d)
#                         arms[-1]["theta_"+policy] = get_mean_of("theta",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False,contextual=True,continuous = continuous, binary = binary, discrete = discrete, categorical = categorical)
#                         arms[-1]["UCB_"+policy]   = get_mean_of("UCB",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False,contextual=True,continuous = continuous, binary = binary, discrete = discrete, categorical = categorical)
#             elif initialization_method == "Mean":
#                 for policy in policies: # the following parameters depend on the policy:
#                     arms[-1]["total_pulls_"+policy]   = 1 ##!ucb
#                     arms[-1]["rewards_"+policy]       = []
#                     arms[-1]["mean_reward_"+policy]   = get_mean_of("mean_reward",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False)
#                     arms[-1]["rounds_pulled_"+policy] = []
#                     if policy in ["LinUCB","LinUCB_greedy"]:
#                         d = continuous+binary+discrete # I'm leaving out categorical not suited for this
#                         arms[-1]["A_"+policy]     = np.identity(d)
#                         arms[-1]["b_"+policy]     = get_mean_of("b",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False)
#                         arms[-1]["theta_"+policy] = get_mean_of("theta",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False) #same as np.dot(arms[i]["A_"+policy],arms[i]["b_"+policy])
#                         arms[-1]["UCB_"+policy]   = get_mean_of("UCB",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False)
#                     elif policy in ["LinUCB_bayesian","LinUCB_bayesian_greedy"]:
#                         d = continuous+binary+discrete
#                         arms[-1]["lambda_"+policy]= 5.0
#                         arms[-1]["D_"+policy]     = np.empty((0,d)) # n*d
#                         arms[-1]["b_"+policy]     = get_mean_of("b",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False)
#                         arms[-1]["A_"+policy]     = np.identity(d)
#                         arms[-1]["theta_"+policy] = get_mean_of("theta",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False)
#                         arms[-1]["UCB_"+policy]   = get_mean_of("UCB",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False)
#             elif initialization_method == "Weighted_Mean":
#                 for policy in policies: # the following parameters depend on the policy:
#                     arms[-1]["total_pulls_"+policy]   = 1 ##!ucb
#                     arms[-1]["rewards_"+policy]       = []
#                     arms[-1]["mean_reward_"+policy]   = get_mean_of("mean_reward",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=False)
#                     arms[-1]["rounds_pulled_"+policy] = []
#                     if policy in ["LinUCB","LinUCB_greedy"]:
#                         d = continuous+binary+discrete # I'm leaving out categorical not suited for this
#                         arms[-1]["A_"+policy]     = np.identity(d)
#                         arms[-1]["b_"+policy]     = get_mean_of("b",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=False)
#                         arms[-1]["theta_"+policy] = get_mean_of("theta",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=False) #same as np.dot(arms[i]["A_"+policy],arms[i]["b_"+policy])
#                         arms[-1]["UCB_"+policy]   = get_mean_of("UCB",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=False)
#                     elif policy in ["LinUCB_bayesian","LinUCB_bayesian_greedy"]:
#                         d = continuous+binary+discrete
#                         arms[-1]["lambda_"+policy]= 5.0
#                         arms[-1]["D_"+policy]     = np.empty((0,d)) # n*d
#                         arms[-1]["b_"+policy]     = get_mean_of("b",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=False)
#                         arms[-1]["A_"+policy]     = np.identity(d)
#                         arms[-1]["theta_"+policy] = get_mean_of("theta",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=False)
#                         arms[-1]["UCB_"+policy]   = get_mean_of("UCB",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=False)
#             elif initialization_method == "Mean_Std":
#                 for policy in policies: # the following parameters depend on the policy:
#                     arms[-1]["total_pulls_"+policy]   = 1 ##!ucb
#                     arms[-1]["rewards_"+policy]       = []
#                     M,v   = get_mean_of("mean_reward",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=True)
#                     arms[-1]["mean_reward_"+policy] = M + v
#                     arms[-1]["rounds_pulled_"+policy] = []
#                     if policy in ["LinUCB","LinUCB_greedy"]:
#                         d = continuous+binary+discrete # I'm leaving out categorical not suited for this
#                         arms[-1]["A_"+policy]     = np.identity(d)
#                         arms[-1]["b_"+policy] ,v    = get_mean_of("b",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=True)
#                         M,v = get_mean_of("theta",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=True) #same as np.dot(arms[i]["A_"+policy],arms[i]["b_"+policy])
#                         arms[-1]["theta_"+policy] = M+v
#                         arms[-1]["UCB_"+policy],v   = get_mean_of("UCB",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=True)
#                     elif policy in ["LinUCB_bayesian","LinUCB_bayesian_greedy"]:
#                         d = continuous+binary+discrete
#                         arms[-1]["lambda_"+policy]= 5.0
#                         arms[-1]["D_"+policy]     = np.empty((0,d)) # n*d
#                         arms[-1]["b_"+policy]  ,v   = get_mean_of("b",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=True)
#                         arms[-1]["A_"+policy]     = np.identity(d)
#                         M,v = get_mean_of("theta",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=True)
#                         arms[-1]["theta_"+policy] = M+v
#                         arms[-1]["UCB_"+policy],v   = get_mean_of("UCB",arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=True)
#             elif initialization_method == "Weighted_Mean_Std":
#                 for policy in policies: # the following parameters depend on the policy:
#                     arms[-1]["total_pulls_"+policy]   = 1 ##!ucb
#                     arms[-1]["rewards_"+policy]       = []
#                     M,v  = get_mean_of("mean_reward",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=True)
#                     arms[-1]["mean_reward_"+policy] = M+v
#                     arms[-1]["rounds_pulled_"+policy] = []
#                     if policy in ["LinUCB","LinUCB_greedy"]:
#                         d = continuous+binary+discrete # I'm leaving out categorical not suited for this
#                         arms[-1]["A_"+policy]     = np.identity(d)
#                         arms[-1]["b_"+policy]  ,v   = get_mean_of("b",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=True)
#                         M,v = get_mean_of("theta",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=True) #same as np.dot(arms[i]["A_"+policy],arms[i]["b_"+policy])
#                         arms[-1]["theta_"+policy] = M + v
#                         arms[-1]["UCB_"+policy] ,v  = get_mean_of("UCB",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=True)
#                     elif policy in ["LinUCB_bayesian","LinUCB_bayesian_greedy"]:
#                         d = continuous+binary+discrete
#                         arms[-1]["lambda_"+policy]= 5.0
#                         arms[-1]["D_"+policy]     = np.empty((0,d)) # n*d
#                         arms[-1]["b_"+policy] ,v    = get_mean_of("b",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=True)
#                         arms[-1]["A_"+policy]     = np.identity(d)
#                         M,v= get_mean_of("theta",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=True)
#                         arms[-1]["theta_"+policy] = M + v
#                         arms[-1]["UCB_"+policy] ,v  = get_mean_of("UCB",arms,dead_arms,number_of_newborn_arms,policy,weighted=True,std=True)
#             file_record=open("arms_history.txt","a")
#             file_record.write(arms[-1]["arm_id"]+" is born\n")
#             file_record.close()
#             index_of_new_arm += 1
#         return arms, number_of_newborn_arms

# def add_newborn_arms_with_context_old(arms,total_of_arms_born, poisson_parameter, \
# arms_behavior, policies, mortal, continuous = 0, distributions_continuous = [], \
# binary = 0, p=[], discrete = 0, distributions_discrete = [], \
# categorical = 0, categories = [], \
# initialization_method = None): # det how many and initialize
#     number_of_newborn_arms = np.random.poisson(poisson_parameter)
#     index_of_new_arm = total_of_arms_born
#     if number_of_newborn_arms == 0:
#         return arms, number_of_newborn_arms
#     else:
#         for new_arm in range(number_of_newborn_arms):
#             arms.append({})
#             arms[-1]["arm_id"] = 'arm_'+str(index_of_new_arm)
#             arms[-1]["distribution"] = arms_behavior
#             arms[-1]["life"] = np.random.poisson(mortal, 1)
#             if arms_behavior == "Bernoulli":
#                 # If behavior is Bernoulli we randomly assign to each arm its probability of success
#                 arms[-1]["p"] = np.random.uniform(0.5,1.0)
#                 arms[-1]["center"] = create_context_df(1,continuous, distributions_continuous, binary, p, \
#                 discrete, distributions_discrete, categorical, categories)
#                 arms[-1]["center"]=arms[-1]["center"].iloc[0]
#             elif arms_behavior == "Truncated_Normal":
#                 # If behavior is Normal we randomly assign to each arm mean and std
#                 arms[-1]["mu"] = np.random.uniform(0.5,1.0) # mean of the normal
#                 arms[-1]["sigma"] = 0.1 #np.random.uniform(0.0,2.5) # std of the normal
#                 arms[-1]["center"] = create_context_df(1,continuous, distributions_continuous, binary, p, \
#                 discrete, distributions_discrete, categorical, categories)
#                 arms[i]["center"]=arms[i]["center"].iloc[0]
#             else:
#                 arms[-1]["p"] = "set parameters"
#             for policy in policies: # the following parameters depend on the policy:
#                 arms[-1]["total_pulls_"+policy]   = 0 ##!ucb
#                 arms[-1]["rewards_"+policy]       = []
#                 arms[-1]["mean_reward_"+policy]   = 0.0 ##! vp
#                 arms[-1]["rounds_pulled_"+policy] = []
#                 if policy in ["LinUCB","LinUCB_greedy"]:
#                     d = continuous+binary+discrete # I'm leaving out categorical not suited for this
#                     arms[-1]["A_"+policy]     = np.identity(d)
#                     arms[-1]["b_"+policy]     = np.array([np.zeros(d)]).T
#                     arms[-1]["theta_"+policy] = np.array([np.zeros(d)]).T #same as np.dot(arms[i]["A_"+policy],arms[i]["b_"+policy])
#                     arms[-1]["UCB_"+policy]   = 0.0
#                 elif policy in ["LinUCB_bayesian","LinUCB_bayesian_greedy"]:
#                     d = continuous+binary+discrete
#                     arms[-1]["lambda_"+policy]= 5.0
#                     arms[-1]["D_"+policy]     = np.empty((0,d)) # n*d
#                     arms[-1]["b_"+policy]     = np.array([np.zeros(d)]).T
#                     arms[-1]["A_"+policy]     = arms[-1]["A_"+policy]     = np.identity(d)
#                     arms[-1]["theta_"+policy] = np.array([np.zeros(d)]).T
#                     arms[-1]["UCB_"+policy]   = 0.0
#             file_record=open("arms_history.txt","a")
#             file_record.write(arms[-1]["arm_id"]+" is born\n")
#             file_record.close()
#             index_of_new_arm += 1
#         return arms, number_of_newborn_arms

# def get_mean_of_mean_rewards(arms,dead_arms,policy):
#     A = [arms[i]["mean_reward_"+policy] for i in range(len(arms))]+[dead_arms[i]["mean_reward_"+policy] for i in range(len(dead_arms))]
#     return np.mean(A)
# def get_weighted_mean_of_mean_rewards(arms,dead_arms,policy):
#     A = [arms[i]["mean_reward_"+policy] for i in range(len(arms))]+[dead_arms[i]["mean_reward_"+policy] for i in range(len(dead_arms))]
#     W = [arms[i]["total_pulls_"+policy] for i in range(len(arms))]+[dead_arms[i]["total_pulls_"+policy] for i in range(len(dead_arms))]
#     return np.average(A,weights=W)
# def get_mean_of_b(arms,dead_arms,policy):
#     A = [arms[i]["b_"+policy] for i in range(len(arms))]+[dead_arms[i]["b_"+policy] for i in range(len(dead_arms))]
#     return np.mean(A,axis=0)
# def get_std_of_mean_rewards(arms,dead_arms,policy):
#     A = [arms[i]["mean_reward_"+policy] for i in range(len(arms))]+[dead_arms[i]["mean_reward_"+policy] for i in range(len(dead_arms))]
#     return np.std(A)
# def get_weighted_std_of_mean_rewards(arms,dead_arms,policy):
#     A = [arms[i]["mean_reward_"+policy] for i in range(len(arms))]+[dead_arms[i]["mean_reward_"+policy] for i in range(len(dead_arms))]
#     W = [arms[i]["total_pulls_"+policy] for i in range(len(arms))]+[dead_arms[i]["total_pulls_"+policy] for i in range(len(dead_arms))]
#     return np.sqrt(np.average((np.array(A)-np.average(A,weights=W))**2,weights=W))
# def get_std_of_b(arms,dead_arms,policy):
#     A = [arms[i]["b_"+policy] for i in range(len(arms))]+[dead_arms[i]["b_"+policy] for i in range(len(dead_arms))]
#     return np.std(A,axis=0)

def get_mean_of(keyword,arms,dead_arms,number_of_newborn_arms,policy,weighted=False,std=False,contextual=False, continuous = 0, binary = 0, discrete = 0, categorical = 0, norm_type = "l2"):
    if weighted==False and std==False and keyword in ["A","b","theta"]:
        A = [arms[i][keyword+"_"+policy] for i in range(len(arms)-number_of_newborn_arms)]   +[dead_arms[i][keyword+"_"+policy] for i in range(len(dead_arms))]
        return np.mean(A,axis = 0)
    elif weighted==True and std==False and keyword in ["A","b","theta"]:
        A = [arms[i][keyword+"_"+policy] for i in range(len(arms)-number_of_newborn_arms)]   +[dead_arms[i][keyword+"_"+policy] for i in range(len(dead_arms))]
        W = [arms[i]["total_pulls_"+policy] for i in range(len(arms)-number_of_newborn_arms)]+[dead_arms[i]["total_pulls_"+policy] for i in range(len(dead_arms))]
        return np.average(A, weights=W, axis = 0)
    elif weighted==False and std==True and keyword in ["A","b","theta"]:
        A = [arms[i][keyword+"_"+policy] for i in range(len(arms)-number_of_newborn_arms)]   +[dead_arms[i][keyword+"_"+policy] for i in range(len(dead_arms))]
        return np.mean(A,axis = 0), np.std(A,axis=0)
    elif weighted==True and std==True and keyword in ["A","b","theta"]:
        A = [arms[i][keyword+"_"+policy] for i in range(len(arms)-number_of_newborn_arms)]   +[dead_arms[i][keyword+"_"+policy] for i in range(len(dead_arms))]
        W = [arms[i]["total_pulls_"+policy] for i in range(len(arms)-number_of_newborn_arms)]+[dead_arms[i]["total_pulls_"+policy] for i in range(len(dead_arms))]
        return np.average(A, weights=W, axis = 0) , np.sqrt(np.average((np.array(A)-np.average(A,weights=W,axis=0))**2,weights=W,axis=0))
    elif weighted==False and std==False:
        A = [arms[i][keyword+"_"+policy] for i in range(len(arms)-number_of_newborn_arms)]    +[dead_arms[i][keyword+"_"+policy] for i in range(len(dead_arms))]
        return np.mean(A)
    elif weighted==True and std==False:
        A = [arms[i][keyword+"_"+policy] for i in range(len(arms)-number_of_newborn_arms)]+[dead_arms[i][keyword+"_"+policy] for i in range(len(dead_arms))]
        W = [arms[i]["total_pulls_"+policy] for i in range(len(arms)-number_of_newborn_arms)]+[dead_arms[i]["total_pulls_"+policy] for i in range(len(dead_arms))]
        return np.average(A, weights=W, axis = 0)
    elif weighted==False and std==True:
        A = [arms[i][keyword+"_"+policy] for i in range(len(arms)-number_of_newborn_arms)]+[dead_arms[i][keyword+"_"+policy] for i in range(len(dead_arms))]
        return np.mean(A), np.std(A)
    elif weighted==True and std==True:
        A = [arms[i][keyword+"_"+policy] for i in range(len(arms)-number_of_newborn_arms)]+[dead_arms[i][keyword+"_"+policy] for i in range(len(dead_arms))]
        W = [arms[i]["total_pulls_"+policy] for i in range(len(arms)-number_of_newborn_arms)]+[dead_arms[i]["total_pulls_"+policy] for i in range(len(dead_arms))]
        return np.average(A, weights=W, axis = 0) , np.sqrt(np.average((np.array(A)-np.average(A,weights=W, axis = 0))**2,weights=W, axis = 0))
    elif weighted==False and std==False and contextual==True:
        A = [arms[i][keyword+"_"+policy] for i in range(len(arms)-number_of_newborn_arms)]+[dead_arms[i][keyword+"_"+policy] for i in range(len(dead_arms))]
        W = [context_similarity(arms[-1]["center"], arms[i]["center"], continuous = continuous, binary = binary, discrete = discrete, categorical = categorical, norm_type = norm_type) for i in range(len(arms)-number_of_newborn_arms)] \
        +[context_similarity(arms[-1]["center"],dead_arms[i]["center"], continuous = continuous, binary = binary, discrete = discrete, categorical = categorical, norm_type = norm_type) for i in range(len(dead_arms))]
        return np.average(A, weights=W, axis = 0)
#
def get_real_best_arm_index(arms, arms_behavior):
    if arms_behavior == "Bernoulli":
        A = [arms[i]["p"] for i in range(len(arms))]
    elif arms_behavior == "Truncated_Normal":
        A = [arms[i]["mu"] for i in range(len(arms))]
    else:
        print("Distribution not found")
        return
    return np.argmax(A)
#
def get_best_arm_parameter(arms, arms_behavior):
    if arms_behavior == "Bernoulli":
        A = [arms[i]["p"] for i in range(len(arms))]
    elif arms_behavior == "Truncated_Normal":
        A = [arms[i]["mu"] for i in range(len(arms))]
    return max(A)
#
def get_best_estimate_arm_index(arms,policy):
    A = [arms[i]["mean_reward_"+policy] for i in range(len(arms))]
    return np.argmax(A)
#
def get_best_mean_estimate(arms,policy):
    A = [arms[i]["mean_reward_"+policy] for i in range(len(arms))]
    return max(A)
#
def remove_arms_that_died(arms,dead_arms): ##
    """ Remove arms that have died and subtracts 1 from remaining life span """
    number_of_arms_deleted = 0
    for j in range(len(arms)):
        if arms[j]["life"] <= 0:
            file_record = open("arms_history.txt","a")
            file_record.write(arms[j]["arm_id"]+" died\n")
            file_record.close()
            dead_arms.append(arms[j])
            arms[j] = 'dead' # flag as 'dead'
            number_of_arms_deleted += 1
        else:
            arms[j]["life"] -= 1
    for j in range(0,number_of_arms_deleted):
        arms.remove('dead')
    return arms, dead_arms, number_of_arms_deleted

def get_percentile_life(arms,percentile):
    A = [arms[i]["life"] for i in range(len(arms))]
    return np.percentile(A,percentile)


# def remove_arms_that_died(arms): ##
#     """ Remove arms that have died and subtracts 1 from remaining life span """
#     number_of_arms_deleted = 0
#     for j in range(len(arms)):
#         if arms[j]["life"] == 0:
#             file_record = open("arms_history.txt","a")
#             file_record.write(arms[j]["arm_id"]+" died\n")
#             file_record.close()
#             arms[j] = 'dead' # flag as 'dead'
#             number_of_arms_deleted += 1
#         else:
#             arms[j]["life"] -= 1
#     for j in range(0,number_of_arms_deleted):
#         arms.remove('dead')
#     return arms, number_of_arms_deleted
#
# poisson parameter can be estimated by #new_arms/turns?
#
# non contextual functions:
# def create_arms(number_of_arms, arms_behavior, policies, mortal = 0): ## add option for custom arms_behavior
#     arms = [0 for i in range(number_of_arms)]
#     for i in range(number_of_arms):
#         arms[i] = {}
#         arms[i]["arm_id"] = 'arm_'+str(i)
#         arms[i]["distribution"] = arms_behavior
#         if arms_behavior == "Bernoulli":
#             # If behavior is Bernoulli we randomly assign to each arm its probability of success
#             arms[i]["p"] = np.random.uniform(0.0,0.9)
#         elif arms_behavior == "Truncated_Normal":
#             # If behavior is Normal we randomly assign to each arm mean and std
#             arms[i]["mu"] = np.random.uniform(0.0,1.0) # mean of the normal
#             arms[i]["sigma"] = 0.1#np.random.uniform(0.0,5.0)
#         else:
#             print("Distribution not found")
#             return
#         for policy in policies: # the following parameters depend on the policy:
#             arms[i]["total_pulls_"+policy]   = 0
#             arms[i]["rewards_"+policy]       = []
#             arms[i]["mean_reward_"+policy]   = 0.0
#             arms[i]["rounds_pulled_"+policy] = []
#     if mortal > 0:
#         # add life from a Poisson distribution         ## or inference!
#         for i in range(number_of_arms):
#             arms[i]["life"] = np.random.poisson(mortal)
#     return arms
# def add_newborn_arms(arms,total_of_arms_born, poisson_parameter, arms_behavior, policies, mortal): # det how many and initialize
#     number_of_newborn_arms = np.random.poisson(poisson_parameter)
#     index_of_new_arm = total_of_arms_born
#     if number_of_newborn_arms == 0:
#         return arms, number_of_newborn_arms
#     else:
#         for new_arm in range(number_of_newborn_arms):
#             arms.append({})
#             arms[-1]["arm_id"] = 'arm_'+str(index_of_new_arm)
#             arms[-1]["distribution"] = arms_behavior
#             arms[-1]["life"] = np.random.poisson(mortal, 1)
#             if arms_behavior == "Bernoulli":
#                 # If behavior is Bernoulli we randomly assign to each arm its probability of success
#                 arms[-1]["p"] = np.random.uniform(0.4,0.9)                       ## new arms are better
#             elif arms_behavior == "Truncated_Normal":
#                 # If behavior is Normal we randomly assign to each arm mean and std
#                 arms[-1]["mu"] = np.random.uniform(0.0,10.0) # mean of the normal
#                 arms[-1]["sigma"] = np.random.uniform(0.0,2.5) # std of the normal
#             else:
#                 arms[-1]["p"] = "set parameters"
#             for policy in policies: # the following parameters depend on the policy:
#                 arms[-1]["total_pulls_"+policy]   = 0 ##!ucb
#                 arms[-1]["rewards_"+policy]       = []
#                 arms[-1]["mean_reward_"+policy]   = 0.0 ##! vp
#                 arms[-1]["rounds_pulled_"+policy] = []
#             file_record=open("arms_history.txt","a")
#             file_record.write(arms[-1]["arm_id"]+" is born\n")
#             file_record.close()
#             index_of_new_arm += 1
#         return arms, number_of_newborn_arms


#-example
# B=create_arms_with_context(10, "Bernoulli", ["LinUCB"], mortal = 10, continuous = 0, distributions_continuous = [], \
# binary = 1, p=[1.0], discrete = 0, distributions_discrete = [  ], \
# categorical = 0, categories = [  ])
#
#
# # Example
# add_newborn_arms_with_context(B, 5, 4, "Bernoulli", ["UCB"], mortal =4,  continuous = 3, distributions_continuous = [["Normal", 3, 1],["Normal", 13, 1],["Uniform",0,10]], \
# binary = 5, p=[0.9,0,0,0.8,0.9], discrete = 3, distributions_discrete = [  ["Poisson",3], ["Poisson", 13], ["Uniform",0,10]  ], \
# categorical = 2, categories = [   [['A','B','C'],[0.1,0.5,0.4]],  [['D','E','F'], [0.1,0.5,0.4]]   ])
#
# get_best_estimate_arm_index(arms, policy)

###########

#
