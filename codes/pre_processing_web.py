import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
from cycler import cycler
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import re
import time
pd.set_option('future.no_silent_downcasting', True)

plt.style.use("bmh")
# Set default color for bars globally
plt.rcParams['axes.prop_cycle'] = cycler(color=['tab:brown'])
# Set global defaults for bar borders
plt.rcParams['patch.edgecolor'] = 'black'  # Border color
plt.rcParams['patch.linewidth'] = 1.5     # Border width

# Convert columns non float to float
def convert_to_float(df):
    for col in df.columns:
        if not pd.api.types.is_float_dtype(df[col]):
            try:
                df[col] = df[col].astype(float)
                #print(f"Column '{col}' converted directly to float.")
            except ValueError:
                # if fail, use LabelEncoder to convert to categóric
                if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    #print(f"Column '{col}' is categorical. Applying LabelEncoder.")
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                else:
                    # if not categorical, mark as NaN
                    #print(f"Column '{col}' Contain non numerical and non categorical data. Defining as NaN.")
                    df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def pre_processing(path):
    
    print(f"[PREPROCESSING] reading the data from '{path}")
    df_init = pd.read_csv(path)

    print(f"[PREPROCESSING] converting the dataset to float")
    
    # remove unnecessary columns
    cols_to_drop = ['ID', 'Start time', 'Completion time', 'Email', 'Name', 'Last modified time', 'Question', "consent"]
    for c in cols_to_drop:
        if c in df_init.columns:
            df_init.drop(columns=c, inplace=True, errors='ignore')
            
    column_names = df_init.columns
    
    df_init.rename(columns={
        'Please select the appropriate consent statement below.':'consent',
        'How do you usually ride your skateboard?':'skate_stance',
        'With which foot forward do you do your highest ollies on a skateboard?\n':'ollie_foot',
        'When you skate an obstacle with transition like a bowl or a half-pipe, which foot forward feels most\xa0comfortable?\n':'bowl_foot',
        'What foot forward do you feel the most safe riding fast down a hill on a skateboard?':'downhill_foot',
        'When you snowboard, which foot forward do you feel most comfortable carving down the mountain with?\n':'snowboard_foot',
        'When you surf, which foot forward do you feel most comfortable when you ride a wave?\n':'surf_foot',
        'Almost all of my friends (more than 90%) share my same stance.':'friends_share_stance',
        'One or more of my parents share my same stance when they skateboard, snowboard, and/or surf.\n':'parents_share_stance',
        'My favorite skateboarder has the same stance as me.\n':'fav_skater_share_stance',
        'I started skateboarding with a different stance than the one I have now.\n':'changed_stance',
        'In recent memory, I recall seeing skateboarding magazine ads or online posts in skate-\nboarding, and I have tried to figure out what stance they are.':'stance_awareness',
        'I have the same stance whether I skate any obstacle, snowboard any hill, or surf any wave.':'consistent_stance',
        'Usually,\xa0when\xa0you\xa0write\xa0with\xa0a\xa0pen\xa0or\xa0pencil,\xa0which\xa0hand\xa0do\xa0you\xa0use?':'hand_write',
        'Usually, when you throw a ball, which hand do you use?\n':'hand_throw',
        'Which hand do you feel more control when swinging a hammer?\n':'hand_hammer',
        'Usually, when you kick a ball into a goal, which foot do you use?\n':'foot_kick',
        'Usually, when you sweep a floor from the side, which foot seems more naturally in front?\n':'foot_sweep',
        'If you were to step on a guitar pedal, which foot would you use?\n':'foot_pedal',
        'To stand on a chair, which foot do you use to get up on the chair?\n':'foot_chair',
        'Take a moment to test your preferred eye. 1) Stretch your arms out in front of you and 2) with your hands make an O shaped hole and 3) line up the O shaped hole to frame an object. Which eye do yo...':'eye_test1',
        'Take a moment to pinpoint an object in the distance and align your thumb over it. Which eye do you need to close to do so?':'eye_test2',
        'How many years have you been skateboarding':'years_skate',
        'In the last few years, how often do you skateboard?':'freq_skate',
        'What would your say your skateboarding expertise level is?':'expertise',
        'Gender:':'gender',
        'Race\xa0(Mark ONLY one with which you MOST CLOSELY identify)':'race',
        'Age':'age',
        'Ethnicity\xa0(Mark ONLY one with which you MOST CLOSELY identify):':'ethnicity',
        'Residence Locality (Mark ONLY one with which you CURRENTLY live)':'residence_locality',
        'Identity Locality (Mark ONLY one with which you IDENTIFY as your home locality)':'home_locality'
    }, inplace=True)
       
    df_init = df_init[(df_init['consent'] != "I am not 18 years of age or older, and/or, I do not agree to participate.")]

    scale_map = {
        'Left foot in front, push with back right foot (regular stance)' : "Regular",
        "Either right foot in front or left foot in front, equally    ": "Both",
        "Left foot in front, push with left right foot (regular mongo stance)": "Regular mongo",
        "Right foot in front, push with front right foot (goofy mongo stance)": "Goofy mongo",
        'Always  Left': "Always Left",
        'Left    ': "Left",
        'Mostly Left    ': "Mostly Left",
        'Both Equally    ': "Both Equally",
        'Mostly Right    ': "Mostly Right",
        'Right    ': "Right",
        'Always Right    ': "Always Right",
        'Right foot in front, push with back left foot (goofy stance)' : "Goofy",
    }

    # mapping the lateralities
    cols_scale = ['skate_stance','ollie_foot','bowl_foot','downhill_foot','snowboard_foot','surf_foot',
                'hand_write','hand_throw','hand_hammer','foot_kick','foot_sweep','foot_pedal',
                'foot_chair']
    for c in cols_scale:
        if c in df_init.columns:
            df_init[c] = df_init[c].map(scale_map)
    
    '''
    # transform some answers to NaN
    df_init.replace({
        'I do not know': np.nan,
        'I do not snowboard enough to know': np.nan,
        'I do not surf enough to know': np.nan
    }, inplace=True)
    '''    

    # mapping the eye tests
    likert_map = {
        'Left    ': "Left",
        'Always Right    ': "Always Right",
        'Mostly Right    ': "Mostly Right",
        'Neither': "Neither",
        'Both Equally    ': "Both Equally",
        'Mostly Left    ': "Mostly Left",
        'Always  Left': "Always Left",
        'Right    ': "Right",
    }
    likert_cols = ['eye_test1','eye_test2']
    for lc in likert_cols:
        if lc in df_init.columns:
            df_init[lc] = df_init[lc].map(likert_map)
            
    

    '''
    # mapping the remain variavles
    likert_map = {
        'Strongly disagree': -10,
        'Disagree': -5,
        'Neutral': 0,
        'Agree': 5,
        'Strongly Agree': 10
    }
    likert_cols = ['friends_share_stance','parents_share_stance','fav_skater_share_stance','changed_stance','stance_awareness','consistent_stance']
    for lc in likert_cols:
        if lc in df_init.columns:
            df_init[lc] = df_init[lc].map(likert_map)
    '''

    # mapping the years of skate
    def parse_years(value):
        if pd.isna(value):
            return np.nan
        # try to extract the first year
        match = re.search(r'\d+', str(value))
        if match:
            return float(match.group(0))
        return np.nan

    if 'years_skate' in df_init.columns:
        df_init['years_skate'] = df_init['years_skate'].apply(parse_years)
        
    
    # possible values found before: adjust with the actual in the new dataset
    df_init['expertise'].unique()

    # in the case of find values like "Advanced (I can occasionally land complex tricks)", etc.
    # map according to the complexity, same as before
    expertise_map = {
        'Beginner (I cannot ride on a board)': "Beginner",
        'Novice (I can push around)': "Novice",
        'Intermediate (I can occasionally land simple tricks)': "Intermediate",
        'Advanced (I can occasionally land complex tricks)': "Advanced",
        'Expert (I can consistently land complex tricks)': "Expert",
        'Sponsored/Pro (I regularly receive skateboard products for free from companies)': "Sponsored/Pro",
    }

    df_init['expertise'] = df_init['expertise'].replace(expertise_map)
    
    df_init = df_init[~df_init['skate_stance'].isna() & (df_init['skate_stance'] != '')]
        
    df_init.to_csv(f"teste.csv", index=False)
        
    # Convert the columns to float
    #df_init = convert_to_float(df_init)
    newPath = "../data/df_converted_web.csv"

    # shows informations about the converted dataframe
    print(f"[PREPROCESSING] finished dataset preprocessing")
    #print(f"[PREPROCESSING] dataFrame after conversion:")
    time.sleep(2)
    #print(df_init.info())
    #time.sleep(2)
    
    #df_init.columns = column_names
    df_init.to_csv(f"{newPath}", index=False)
    print(f"[PREPROCESSING] new dataframe was saved in {newPath}")
    
    return df_init

pre_processing("../data/data.csv")