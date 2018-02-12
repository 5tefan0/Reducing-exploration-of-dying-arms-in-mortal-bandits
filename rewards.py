import numpy as np
from df_context_generator import create_columns_names, create_context_df, context_distance, context_similarity
# add context
# def get_reward_with_context_old(arm_j,context_t, contextual_space):
#     behavior = arm_j["distribution"]
#     if behavior == "Bernoulli" or behavior == "Bernoulli_low":
#         return np.random.binomial(1,arm_j["p"]*context_similarity(context_t, arm_j["center"], contextual_space))
#     elif behavior == "Truncated_Normal":
#         return min(max(0.0,np.random.normal(arm_j["mu"]*context_similarity(context_t, arm_j["center"], contextual_space),arm_j["sigma"])),1.0)
# #
# def contextual_parameter(arm_j, context_t, contextual_relevance, coefficients_context, coefficients_arms):
#     behavior = arm_j["distribution"]
#     p = arm_j["p"]
#     for i in range(len(context_t)):
#         for j in range(len(coefficients_arms)):
#             p += contextual_relevance[i]*(float(arm_j["features"][j])/coefficients_arms[j])*( float(context_t[i])/ coefficients_context[i])
#     return p

# def get_reward_with_context(arm_j,context_t, contextual_space, contextual_relevance, coefficients_context, coefficients_arms):
#     behavior = arm_j["distribution"]
#     if behavior == "Bernoulli" or behavior == "Bernoulli_low":
#         return np.random.binomial(1,contextual_parameter(arm_j, context_t, contextual_relevance, coefficients_context, coefficients_arms))
#     elif behavior == "Truncated_Normal":
#         return min(max(0.0,contextual_parameter(arm_j, context_t, contextual_relevance, coefficients_context, coefficients_arms)),1.0)


# def initialize_mean_reward_with_context(arms, dead_arms,G,rewards_history,tradeoff_history,contexts_history,policy,mortal, contextual_space):
#     for j in range(len(arms)):
#         context_t = contexts_history.iloc[j]
#         reward = get_reward_with_context(arms[j],context_t, contextual_space)#
#         update_arm_with_context(j,arms, dead_arms, reward, j, policy,context_t)
#         rewards_history[policy][j] = reward * G[j]
#         tradeoff_history[policy][j] = False
#         if mortal > 0:
#             arms[j]["life"] -= 1
#     return arms

def initialize_mean_reward_contextual(arms, dead_arms,G, NonContextual,rewards_history,tradeoff_history,contexts_history,policy,mortal, life_threshold, contextual_space, contextual_relevance):
    if policy in ["Adaptive_greedy_life"]:
        for j in range(len(arms)):
            context_t = contexts_history.iloc[j]
            if arms[j]["life"] >= life_threshold:
                reward = get_reward_contextual(arms[j], NonContextual, context_t=context_t, contextual_relevance=contextual_relevance)#
                update_arm_with_context(j,arms, dead_arms, reward, j, policy,context_t)
                rewards_history[policy][j] = reward * G[j]
                tradeoff_history[policy][j] = False
            else:
                A = [arms[i]["life"] for i in range(len(arms))]
                # the following gets the 50% highest INDECES of list A, and chooses the one in the random position. ##
                arm_played = heapq.nlargest(len(arms)/2, xrange(len(A)), key=A.__getitem__)[np.random.randint(0,len(arms)/2)]
                reward = get_reward_contextual(arms[arm_played], NonContextual, context_t=context_t, contextual_relevance=contextual_relevance)#
                update_arm_with_context(arm_played,arms, dead_arms, reward, j, policy,context_t)
                rewards_history[policy][j] = reward * G[j]
                tradeoff_history[policy][j] = False
            # if mortal > 0: #of course it's mortal
            #     arms[j]["life"] -= (len(arms)-j) #drop life   #####
        return arms
    else:
        for j in range(len(arms)):
            context_t = contexts_history.iloc[j]
            reward = get_reward_contextual(arms[j], NonContextual, context_t=context_t, contextual_relevance=contextual_relevance)#
            update_arm_with_context(j,arms, dead_arms, reward, j, policy,context_t)
            rewards_history[policy][j] = reward * G[j]
            tradeoff_history[policy][j] = False
            # if mortal > 0:
            #     arms[j]["life"] -= (len(arms)-j) #drop life
        return arms

def get_reward_contextual(arm_j, NonContextual, context_t = None, contextual_relevance = None):
    behavior = arm_j["distribution"]
    if behavior == "Bernoulli":
        reward = np.random.binomial(1,arm_j["p"])
    elif behavior == "Truncated_Normal":
        reward = min(max(0.0,np.random.normal(arm_j["mu"],arm_j["sigma"])),1.0)
    return reward*get_reward_modifier(arm_j, NonContextual, context_t = context_t, contextual_relevance = contextual_relevance)

def get_reward_modifier(arm_j, NonContextual, context_t = None, contextual_relevance = None):
    if NonContextual == True:
        return 1.0
    else:
        reward_modifier = 0.0
        # print(len(context_t))
        # print(len(arm_j["features"]))
        for i in range(len(context_t)):
            # print("i:\n")
            # print(i)
            for j in range(len(arm_j["features"])):
                # print("j:\n")
                # print(j)
                # print(arm_j["features"][j])
                # print(contextual_relevance["arm_feature_"+str(j)][i])
                # print(context_t[i])
                reward_modifier += arm_j["features"][j] * contextual_relevance["arm_feature_"+str(j)][i] * context_t[i]
        return max(0.0,reward_modifier)


#
def initialize_mean_reward(arms,G,rewards_history,tradeoff_history,policy,mortal):
    for j in range(len(arms)):
        reward = get_reward(arms[j])
        update_arm(arms[j] , reward, j, policy)
        rewards_history[policy][j] = reward * G[j]
        tradeoff_history[policy][j] = False
        if mortal > 0:
            arms[j]["life"] -= 1
    return arms

def get_reward(arm_j):
    behavior = arm_j["distribution"]
    if behavior == "Bernoulli":
        reward = np.random.binomial(1,arm_j["p"])
    elif behavior == "Truncated_Normal":
        reward = min(max(0.0,np.random.normal(arm_j["mu"],arm_j["sigma"])),1.0)
    return reward
#
def initialize_mean_reward(arms,G,rewards_history,tradeoff_history,policy,mortal):
    for j in range(len(arms)):
        reward = get_reward(arms[j])
        update_arm(arms[j] , reward, j, policy)
        rewards_history[policy][j] = reward * G[j]
        tradeoff_history[policy][j] = False
        if mortal > 0:
            arms[j]["life"] -= 1
    return arms




def update_arm_with_context(j,arms,dead_arms,reward,time,policy,context_t):
    arms[j]["total_pulls_"+policy] += 1
    arms[j]["rounds_pulled_"+policy].append(time)
    arms[j]["rewards_"+policy].append(reward)
    arms[j]["mean_reward_"+policy] = np.mean(arms[j]["rewards_"+policy])
    if policy in ["LinUCB","LinUCB_greedy"]:
        x = np.array([context_t]).T
        arms[j]["A_"+policy]     = arms[j]["A_"+policy] + np.dot(x,x.T)
        arms[j]["b_"+policy]     = arms[j]["b_"+policy] + reward*x
    elif policy in ["LinUCB_bayesian","LinUCB_bayesian_greedy"]:
        x = np.array([context_t]).T
        d = len(x)
        arms[j]["D_"+policy]     = np.vstack([arms[j]["D_"+policy],x.T]) # n*d
        arms[j]["b_"+policy]     = np.dot(arms[j]["D_"+policy].T,np.array([arms[j]["rewards_"+policy]]).T)
        arms[j]["A_"+policy]     = np.dot(arms[j]["D_"+policy].T,arms[j]["D_"+policy])+(1.0/arms[j]["lambda_"+policy])*np.identity(d)
        arms[j]["theta_"+policy] = np.dot( np.linalg.inv(arms[j]["A_"+policy])  , arms[j]["b_"+policy])
    return

   # if time >= len(arms)+len(dead_arms):
   #     x = np.array([context_t.iloc[0]]).T
   # else:
   #     x = np.array([context_t]).T
#
def create_reward_history(number_of_turns, policies):
    """ Creates a dictionary that stores the rewards history for each
    policy applied """
    rewards_history = dict()
    for policy in policies:
        rewards_history[policy]=[False]*number_of_turns
    return rewards_history
#
def create_tradeoff_history(number_of_turns, policies):
    """ Creates a dictionary that stores the trade-off history
    (exploration VS exploitation) for each policy applied """
    tradeoff_history = dict()
    for policy in policies:
        tradeoff_history[policy]=[False]*number_of_turns
    return tradeoff_history
