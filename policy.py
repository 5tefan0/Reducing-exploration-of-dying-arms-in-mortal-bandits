import numpy as np
import heapq


#-------------------------------------------------------------------------------
# Core engine of the algorithms. Outputs the index of the arm to play.
#-------------------------------------------------------------------------------

def choose_arm_and_tradeoff_with_context(t, policy, arms, G, z, \
      number_of_arms_born, total_of_arms_born, mortal,contexts_history,  context_t, rewards_history, contextual_space, a_low=0, b_high=1):
    """ Returns True if the policy exploited the best arm found so far. """
    m = len(arms)
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------
    if policy == "LinUCB":
        alpha = 5.0
        x = np.array(context_t).T # column vector of context
        average_expected_reward = [False]*len(arms)
        UCB_values              = [False]*len(arms)
        for i in range(len(arms)):
            arms[i]["theta_"+policy]   = np.dot(np.linalg.inv(arms[i]["A_"+policy]),arms[i]["b_"+policy])
            average_expected_reward[i] = np.dot(arms[i]["theta_"+policy].T ,x)
            arms[i]["UCB_"+policy]     = average_expected_reward[i] + alpha*np.sqrt(np.dot(x.T,np.dot(np.linalg.inv(arms[i]["A_"+policy]),x)))
            UCB_values[i]              = arms[i]["UCB_"+policy]
        arm_to_play = np.argmax(UCB_values)
        if arm_to_play == np.argmax(average_expected_reward):
            Exploitation = True
        else:
            Exploitation = False
        return arm_to_play, Exploitation
    elif policy == "LinUCB_bayesian":
        alpha = 5.0
        x = np.array(context_t).T # column vector of context
        average_expected_reward = [False]*len(arms)
        UCB_values              = [False]*len(arms)
        for i in range(len(arms)):
            average_expected_reward[i] = np.dot(arms[i]["theta_"+policy].T ,x)
            arms[i]["UCB_"+policy]     = average_expected_reward[i] + alpha*np.sqrt(np.dot(x.T,np.dot(np.linalg.inv(arms[i]["A_"+policy]),x)))
            UCB_values[i]              = arms[i]["UCB_"+policy]
        arm_to_play = np.argmax(UCB_values)
        if arm_to_play == np.argmax(average_expected_reward):
            Exploitation = True
        else:
            Exploitation = False
        return arm_to_play, Exploitation
    elif policy == "UCB-L":
        arm_to_play = -1
        second_best_arm = -1
        best_UCB_found = -1
        for j in xrange(len(arms)):
            T_j = arms[j]["total_pulls_"+policy]
            UCB_j = arms[j]["mean_reward_"+policy] + np.sqrt( 2.0 * np.log( t - arms[j]["born_time"]) / T_j )
            if UCB_j > best_UCB_found:
                second_best_arm = arm_to_play # the arm you thought you were playing now is second best
                arm_to_play = j # update best arm
                best_UCB_found = UCB_j
        if arms[arm_to_play]["mean_reward_"+policy] > arms[second_best_arm]["mean_reward_"+policy] and second_best_arm !=-1: #\
        #+ np.sqrt( 2.0 * np.log( t ) / arms[second_best_arm]["total_pulls_"+policy] ):
            Exploitation = True
        else:
            Exploitation = False
        return arm_to_play, Exploitation
    elif policy == "VMAB":
        arm_to_play = -1
        second_best_arm = -1
        best_UCB_found = -1
        for j in xrange(len(arms)):
            T_j = arms[j]["total_pulls_"+policy]
            if T_j == 0: #always play new arms
                return j, False
            UCB_j = arms[j]["mean_reward_"+policy] + np.sqrt( 2.0 * np.log( t - arms[j]["born_time"]) / T_j )
            if UCB_j > best_UCB_found:
                second_best_arm = arm_to_play # the arm you thought you were playing now is second best
                arm_to_play = j # update best arm
                best_UCB_found = UCB_j
        if arms[arm_to_play]["mean_reward_"+policy] > arms[second_best_arm]["mean_reward_"+policy] and second_best_arm !=-1: #\
        #+ np.sqrt( 2.0 * np.log( t ) / arms[second_best_arm]["total_pulls_"+policy] ):
            Exploitation = True
        else:
            Exploitation = False
        return arm_to_play, Exploitation
    elif policy == "AG":
        # set algorithm parameters
        c = 0.9
        best_arm_so_far = get_best_estimate_arm_index(arms,policy)
        if arms[best_arm_so_far]["distribution"]=="Bernoulli":
            probability_of_random_exploration = 1.0 - min(1.0, c * arms[best_arm_so_far]["mean_reward_"+policy]) # c * p_m
        else: ## add other distributions
            probability_of_random_exploration = 1.0 - min(1.0, c * (arms[best_arm_so_far]["mean_reward_"+policy]-a_low)/(b_high-a_low)) # /max(arms[best_arm_so_far]["rewards_"+policy]) )
        exploration = np.random.binomial(1, probability_of_random_exploration, size=None)
        if exploration == 0:
            return best_arm_so_far , True
        else:
            return np.random.randint(0,len(arms)) , False
    elif policy == "AG-L":
        # set algorithm parameters
        c = 0.9
        best_arm_so_far = get_best_estimate_arm_index(arms,policy)
        if arms[best_arm_so_far]["distribution"] == "Bernoulli":
            probability_of_random_exploration = 1.0 - min(1.0, c * arms[best_arm_so_far]["mean_reward_"+policy]) # c * p_m
        else: # add other distributions
            probability_of_random_exploration = 1.0 - min(1.0, c * (arms[best_arm_so_far]["mean_reward_"+policy]-a_low)/(b_high-a_low)) # max(arms[best_arm_so_far]["rewards_"+policy]) )
        exploration = np.random.binomial(1, probability_of_random_exploration, size=None)
        if exploration == 0:
            return best_arm_so_far , True
        else:
            A = [arms[i]["life"] for i in range(len(arms))]
            # the following gets the 50% highest INDECES of list A, and chooses the one in the random position.
            if len(arms)/2 == 0:
                return 0, False
            else:
                return heapq.nlargest(len(arms)/2, xrange(len(A)), key=A.__getitem__)[np.random.randint(0,len(arms)/2)] , False
    elif policy == "LinUCB-L":
        alpha = 5.0
        c = 1.0
        x = np.array(context_t).T # column vector of context
        average_expected_reward = [False]*len(arms)
        UCB_values              = [False]*len(arms)
        UCB_values2              = [False]*len(arms)
        for i in range(len(arms)):
            arms[i]["theta_"+policy]   = np.dot(np.linalg.inv(arms[i]["A_"+policy]),arms[i]["b_"+policy])
            average_expected_reward[i] = np.dot(arms[i]["theta_"+policy].T ,x)
            arms[i]["UCB_"+policy]     = average_expected_reward[i] + alpha*np.sqrt(np.dot(x.T,np.dot(np.linalg.inv(arms[i]["A_"+policy]),x)))
            UCB_values[i]              = arms[i]["UCB_"+policy]
            UCB_values2[i]             = (c*np.log(arms[i]["life"]))*UCB_values[i]
        arm_to_play = np.argmax(UCB_values2)
        if arm_to_play == np.argmax(average_expected_reward):
            Exploitation = True
        else:
            Exploitation = False
        return arm_to_play, Exploitation
    elif policy == "AG-L(predict)":
        # set algorithm parameters
        c = 0.9
        best_arm_so_far = get_best_estimate_arm_index(arms,policy)
        if arms[best_arm_so_far]["distribution"] == "Bernoulli":
            probability_of_random_exploration = 1.0 - min(1.0, c * arms[best_arm_so_far]["mean_reward_"+policy]) # c * p_m
        else: # add other distributions
            probability_of_random_exploration = 1.0 - min(1.0, c * (arms[best_arm_so_far]["mean_reward_"+policy]-a_low)/(b_high-a_low)) # max(arms[best_arm_so_far]["rewards_"+policy]) )
        exploration = np.random.binomial(1, probability_of_random_exploration, size=None)
        if exploration == 0:
            return best_arm_so_far , True
        else:
            A = [arms[i]["life"] for i in range(len(arms))]
            # the following gets the 50% highest INDECES of list A, and chooses the one in the random position.
            if len(arms)/2 == 0:
                return 0, False
            else:
                return heapq.nlargest(len(arms)/2, xrange(len(A)), key=A.__getitem__)[np.random.randint(0,len(arms)/2)] , False
    elif policy == "LinUCB-L(predict)":
        alpha = 5.0
        c = 1.0
        x = np.array(context_t).T # column vector of context
        average_expected_reward = [False]*len(arms)
        UCB_values              = [False]*len(arms)
        UCB_values2              = [False]*len(arms)
        for i in range(len(arms)):
            arms[i]["theta_"+policy]   = np.dot(np.linalg.inv(arms[i]["A_"+policy]),arms[i]["b_"+policy])
            average_expected_reward[i] = np.dot(arms[i]["theta_"+policy].T ,x)
            arms[i]["UCB_"+policy]     = average_expected_reward[i] + alpha*np.sqrt(np.dot(x.T,np.dot(np.linalg.inv(arms[i]["A_"+policy]),x)))
            UCB_values[i]              = arms[i]["UCB_"+policy]
            UCB_values2[i]             = (c*np.log(arms[i]["life"]))*UCB_values[i]
        arm_to_play = np.argmax(UCB_values2)
        if arm_to_play == np.argmax(average_expected_reward):
            Exploitation = True
        else:
            Exploitation = False
        return arm_to_play, Exploitation
    elif policy == "UCB-L(predict)":
        arm_to_play = -1
        second_best_arm = -1
        best_UCB_found = -1
        for j in xrange(len(arms)):
            T_j = arms[j]["total_pulls_"+policy]
            UCB_j = arms[j]["mean_reward_"+policy] + np.sqrt( 2.0 * np.log( t - arms[j]["born_time"]) / T_j )
            if UCB_j > best_UCB_found:
                second_best_arm = arm_to_play # the arm you thought you were playing now is second best
                arm_to_play = j # update best arm
                best_UCB_found = UCB_j
        if arms[arm_to_play]["mean_reward_"+policy] > arms[second_best_arm]["mean_reward_"+policy] and second_best_arm !=-1: #\
        #+ np.sqrt( 2.0 * np.log( t ) / arms[second_best_arm]["total_pulls_"+policy] ):
            Exploitation = True
        else:
            Exploitation = False
        return arm_to_play, Exploitation
