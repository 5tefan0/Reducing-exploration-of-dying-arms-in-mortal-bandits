#Example
from pprint import pprint  #pretty print for dictionaries
#-------------------------------------------------------------------------------
# Import useful modules
#-------------------------------------------------------------------------------
# module used to get the highest values from a list:
import heapq
# bandit functions:
from RGOT import RGOT
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



#TYPE OF ARMS AND LIFE
arms_p_and_type={}
arms_p_and_type["p"]=[0.7,0.3]
arms_p_and_type["type"]=[["Poisson", 60],["Poisson", 250]]

#CONTEXTUAL AND ARMS SPACES
contextual_space = { "continuous": [], "binary":[1.0], "discrete":[], "categorical": [], "norm_type": "l2"}
arms_features_space = { "continuous": [], "binary":[1.0], "discrete":[], "categorical": [], "norm_type": "l2"}

#PLAY AG-L (or choose from "AG","UCB-L","VMAB","AG-L","LinUCB-L","AG-L(predict)","UCB-L(predict)","LinUCB-L(predict)" )
policies =  ["AG-L"]
a,b,c,d,e,f=RGOT(G,toPredict = None, number_of_turns = 100, number_of_arms = 50, \
    arms_behavior = "Bernoulli",policies =  policies, NonContextual = True, \
    mortal = 80, life_method="set", arms_p_and_type=arms_p_and_type, poisson_parameter = 1.1,\
     contextual_space=contextual_space,arms_features_space =arms_features_space)
