import pandas as pd
import numpy as np

def get_ratings_for_answers(df: pd.DataFrame):
    arr_random = np.random.default_rng().uniform(low=0,high=1,size=[df.shape[0],1])
    df.loc[:, 'ratings'] = arr_random


def get_feedback_for_answers(df: pd.DataFrame):
    df.loc[:, 'feedback'] = 'Some Random Feedback'

def get_overall_feedback():
    return 'Some Overall Feedback'