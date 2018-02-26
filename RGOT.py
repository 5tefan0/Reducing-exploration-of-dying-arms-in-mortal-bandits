""" Regulating Exploration Over Dying Arms in Moratal Bandits"""

#-------------------------------------------------------------------------------
# RGODA library for Python - Mortal setting
#-------------------------------------------------------------------------------

from pprint import pprint  #pretty print for dictionaries
#-------------------------------------------------------------------------------
# Import useful modules
#-------------------------------------------------------------------------------
# module used to get the highest values from a list:
import heapq
# bandit functions:
from arms import create_arms_with_context, add_newborn_arms_with_context, set_arms_life
from greed import set_G_as, compute_z, compute_z2
from df_context_generator import create_columns_names, create_context_df, context_distance, context_similarity, create_relevance, create_regression_coefficients_for_life
from arms import create_arms_with_context, get_best_estimate_arm_index, remove_arms_that_died, get_mean_of
from rewards import get_reward_contextual, initialize_mean_reward_contextual, get_reward_modifier, update_arm_with_context, create_reward_history, create_tradeoff_history
from policy import choose_arm_and_tradeoff_with_context
from contexts_generator import get_context, create_continuous_context, create_binary_context, create_discrete_context, create_category_context
# pandas to handle data frames:
import pandas as pd
# bokeh for visualize results:
from bokeh.charts import Histogram, Bar, defaults, vplot, hplot, show, output_file
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.palettes import Spectral7, Blues
from bokeh.models import HoverTool, ColumnDataSource
# for math functions we use Numpy:
import numpy as np


#-------------------------------------------------------------------------------
# The following functions plays one istance of the multi-armed bandit algorithm
#-------------------------------------------------------------------------------

# G list of 1's



def RGOT(G, number_of_turns, number_of_arms, \
    arms_behavior, toPredict = None, NonContextual = False, policies =  [], \
    mortal = False, life_method ="long_and_short", arms_p_and_type={}, poisson_parameter = 0.2, initialization_method = None, greed_type = "None", \
    contextual_space = { "continuous": [], "binary":[1.0], "discrete":[], "categorical": [], "norm_type": "l2"}, \
    arms_features_space = { "continuous": [], "binary":[1.0], "discrete":[], "categorical": [], "norm_type": "l2"} ):
    # if NonContextual == True:
    #     binary = 1
    #     p = [1.0]
    # # open file of arms:
    # else:
    contextual_relevance = create_relevance(arms_features_space, contextual_space, method = "Uniform")
    #if NonContextual == True:
    a_low, b_high = 0,  1#compute_range_of_rewards(arms_behavior, contextual_relevance, contextual_space, arms_features_space)
    #coefficients_context = get_coefficients(contextual_space)
    #coefficients_arms = get_coefficients(arms_features_space)
    file_record = open("arms_history.txt","w")
    # create arms in the context space:
    list_true_regression_coefficients = create_regression_coefficients_for_life(arms_features_space) #life length depends on this
    arms = create_arms_with_context(number_of_arms, arms_behavior, policies, contextual_space, arms_features_space, mortal, life_method, arms_p_and_type,   list_true_regression_coefficients = list_true_regression_coefficients)
    dead_arms = [] # list of arms that die
    total_of_arms_born = len(arms)
    rewards_history    = create_reward_history(number_of_turns, policies)
    tradeoff_history   = create_tradeoff_history(number_of_turns, policies)
    # create the context dataframe
    contexts_history = create_context_df(len(arms), contextual_space)
    # parameters for time series estimation:
    alpha_smoothing_parameter = 0.9
    beta_smoothing_parameter  = 0.9
    # Initialization:
    life_threshold = 90#get_percentile_life(arms, 50.0)  ####
    for policy in policies: # for each policy we initialize the arms
        # print("before initialization")
        # for j in range(len(arms)):
        #     print(arms[j]['life'])
        arms = initialize_mean_reward_contextual(arms, dead_arms,G, NonContextual,rewards_history,tradeoff_history,contexts_history,policy,mortal,life_threshold, contextual_space, contextual_relevance)          ## add option to not initialize
    for j in range(len(arms)):
        if mortal > 0: #of course it's mortal
            arms[j]["life"] -= (len(arms)-j) #drop life
        else:
            continue
    if toPredict != None:
        past_G_and_predicted_Gt              = [False]*number_of_turns
        past_G_and_predicted_Gt[0:len(arms)] = G[0:len(arms)]
        predicted_G                          = [False]*number_of_turns # storage for predicted values
        predicted_G[0:len(arms)]             = G[0:len(arms)]
        trend_G                              = [False]*number_of_turns
        beta_smoothing_parameter = 0.9 #
        if toPredict == "double_exponential_smoothing":
            predicted_G[2] = G[1]
            trend_G[2]     = G[1] - G[0]
            alpha_smoothing_parameter = 0.9
            beta_smoothing_parameter  = 0.2
            for s in range(3,len(arms)):
                predicted_G[s] = alpha_smoothing_parameter*G[s-1] + (1.0-alpha_smoothing_parameter)*(predicted_G[s-1]+trend_G[s-1])
                trend_G[s]     = beta_smoothing_parameter*(predicted_G[s]-predicted_G[s-1]) + (1.0-beta_smoothing_parameter)*trend_G[s-1]
        # elif toPredict == "triple_exponential_smoothing":
        #     predicted_G[1] = G[0] #ok
        #     trend_G[2]     = G[1] - G[0]
        #     alpha_smoothing_parameter = 0.9
        #     beta_smoothing_parameter  = 0.9
        #     for s in range(3,len(arms)):
        #         predicted_G[s] = alpha_smoothing_parameter*G[s-1] + (1.0-alpha_smoothing_parameter)*(predicted_G[s-1]+trend_G[s-1])
        #         PREDICTED = alpha_smoothing_parameter*G[t-1] + (1.0-alpha_smoothing_parameter)*(predicted_G[t-1]+trend_G[t-1])
        #         trend_G[s]     = beta_smoothing_parameter*(PREDICTED-predicted_G[s-1]) + (1.0-beta_smoothing_parameter)*trend_G[s-1]
    # Game after Initialization:
    for t in range(number_of_arms,number_of_turns):
        context_t = create_context_df(1, contextual_space)
        if toPredict != None:
            past_G_and_predicted_Gt[t] = predict_Gt(t, G, past_G_and_predicted_Gt, predicted_G, trend_G, toPredict, lag=4, alpha_smoothing_parameter=0.9)
            predicted_G[t]             = past_G_and_predicted_Gt[t] #update the predicted values for G
            PREDICTED = alpha_smoothing_parameter*past_G_and_predicted_Gt[t-1] + (1.0-alpha_smoothing_parameter)*(predicted_G[t-1]+trend_G[t-1])
            trend_G[t] = beta_smoothing_parameter*(PREDICTED-predicted_G[t-1]) + (1.0-beta_smoothing_parameter)*trend_G[t-1]
        if mortal > 0: # update arms if you are playing a mortal game
            arms, dead_arms, number_of_arms_dead = remove_arms_that_died(arms, dead_arms)                                                     # remove arms that died and subtracts 1 from remaining life
            arms, number_of_arms_born = add_newborn_arms_with_context(arms, t, dead_arms, total_of_arms_born,  poisson_parameter, arms_behavior, policies, arms_features_space, mortal, contextual_space, life_method, arms_p_and_type,  initialization_method = initialization_method,  list_true_regression_coefficients = list_true_regression_coefficients)
            total_of_arms_born        += number_of_arms_born
            if len(arms) == 0: #nothing to play
                continue # go to next t and hopefully you add an arm
        else:
            number_of_arms_born = 0
            number_of_arms_dead = 0
        for policy in policies:
            # if some arms are dead we can start prediction:
            #best_arm_so_far = get_best_estimate_arm_index(arms, policy)
            if toPredict != None:
                z=compute_z2(past_G_and_predicted_Gt, greed_type) # may depend on policy  ##!!
                arm_to_play, tradeoff = choose_arm_and_tradeoff(t, policy, arms, best_arm_so_far, past_G_and_predicted_Gt, z, number_of_arms_born,total_of_arms_born, mortal, a_low, b_high)
            else:
                z = compute_z2(G, greed_type)#
                arm_to_play, tradeoff = choose_arm_and_tradeoff_with_context(t, policy, arms, G, z, \
                      number_of_arms_born, total_of_arms_born, mortal, contexts_history,  context_t, rewards_history, contextual_space, a_low, b_high) ##mortal
            x_t = get_reward_contextual(arms[arm_to_play],NonContextual, context_t.iloc[0], contextual_relevance)                                 # reward for the arm played
            rewards_history[policy][t]  = x_t * G[t]                             # actual reward you get modified by the greed function
            tradeoff_history[policy][t] = tradeoff * x_t * G[t]
            update_arm_with_context(arm_to_play,arms,dead_arms,x_t,t,policy,context_t.iloc[0])    ###                      # update the arm performance under this policy
            if toPredict != None:
                past_G_and_predicted_Gt[t]=G[t]
        contexts_history=pd.concat([contexts_history,context_t])
        contexts_history=contexts_history.reset_index(drop=True)
    if toPredict != None:
        return pd.DataFrame(arms), rewards_history, tradeoff_history, total_of_arms_born, past_G_and_predicted_Gt, predicted_G, trend_G ##
    else:
        return arms, rewards_history, tradeoff_history, total_of_arms_born, dead_arms, contextual_relevance

# contextual_space = {}
# contextual_space["continuous"]   = [  ["Normal", 3, 1], ["Normal", 13, 1], ["Uniform",0,10]  ]
# contextual_space["binary"]       = [  0.9,   0,  0,  0.8,   0.9  ]
# contextual_space["discrete"]     = [  ["Poisson",3], ["Poisson", 13], ["Uniform",0,10]  ]
# contextual_space["categorical"]  = [ ]
# contextual_space["norm_type"]    = "l2"
#
# arms_features_space = {}
# arms_features_space["continuous"]   = [  ["Normal", 31, 5],["Normal", 13, 5],["Uniform",4,20]  ]
# arms_features_space["binary"]       = [  0.7,  0.2,  0.3 ]
# arms_features_space["discrete"]     = [  ["Poisson",2], ["Uniform",5,11]  ]
# arms_features_space["categorical"]  = [    ]
# arms_features_space["norm_type"]    = "l2"
#
# arms_features_space = {}
# arms_features_space["continuous"]   = [   ]
# arms_features_space["binary"]       = [   0.5, 0.1, 0.9, 0.3 , 0.5, 0.1, 0.9, 0.7,0.7,  0.2,  0.3 , 0.5, 0.1, 0.9, 0.3 ,0.7,  0.2,  0.3 , 0.5, 0.1, 0.9, 0.7,  0.2,  0.3 ]
# arms_features_space["discrete"]     = [   ]
# arms_features_space["categorical"]  = [    ]
# arms_features_space["norm_type"]    = "l2"
#
# contextual_space = {}
# contextual_space["continuous"]   = [   ]
# contextual_space["binary"]       = [  0.7,  0.2,  0.3 , 0.5, 0.1, 0.9, 0.7,  0.2,  0.3 , 0.5, 0.1, 0.9, 0.3 , 0.5, 0.1, 0.9, 0.7,0.7,  0.2,  0.3 , 0.5, 0.1, 0.9, 0.3 ]
# contextual_space["discrete"]     = [  ]
# contextual_space["categorical"]  = [ ]
# contextual_space["norm_type"]    = "l2"
#
# no_arms_feature = { "continuous": [], "binary":[1.0], "discrete":[], "categorical": [], "norm_type": "l2"}
#
