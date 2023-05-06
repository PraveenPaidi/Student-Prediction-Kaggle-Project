#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import time
from xgboost import XGBClassifier
from sklearn.metrics import f1_score


# Get the start time
start_time = time.time()

# reading the data
train = pd.read_csv(r'D:\Spring 23\EEE 591\Project\train.csv')
# reading the data from the train labels and the session id with 18 questions
train_label = pd.read_csv(r'D:\Spring 23\EEE 591\Project\train_labels.csv')

# Competition specific - separate labels into session_id and question number
session_list = []
q_list = []

# Iterate over each row in the DataFrame
for index, row in train_label.iterrows():
    # Split the session_id string by the underscore character
    session_id_parts = row['session_id'].split('_')
    
    # Extract the first part of the session_id and convert it to an integer
    session = int(session_id_parts[0])
    
    # Extract the last part of the session_id, slice it to remove the 'q' character, and convert it to an integer
    q = int(session_id_parts[-1][1:])
    
    # Append the values to the corresponding lists
    session_list.append(session)
    q_list.append(q)

# Add the new columns to the DataFrame using the lists of values
train_label['session'] = session_list
train_label['q'] = q_list

# Print the first few rows of the updated DataFrame to verify the changes
train_label.head()

#creating extra features and concating the dataset
new_data = pd.get_dummies(train['event_name'])
train = pd.concat([train, new_data], axis=1)


# DATA DISCRIPTION

#session_id - the ID of the session the event took place in
#index - the index of the event for the session
#elapsed_time - how much time has passed (in milliseconds) between the start of the session and when the event was recorded
#event_name - the name of the event type
#name - the event name (e.g. identifies whether a notebook_click is is opening or closing the notebook)
#level - what level of the game the event occurred in (0 to 22)
#page - the page number of the event (only for notebook-related events)
#room_coor_x - the coordinates of the click in reference to the in-game room (only for click events)
#room_coor_y - the coordinates of the click in reference to the in-game room (only for click events)
#screen_coor_x - the coordinates of the click in reference to the player’s screen (only for click events)
#screen_coor_y - the coordinates of the click in reference to the player’s screen (only for click events)
#hover_duration - how long (in milliseconds) the hover happened for (only for hover events)
#text - the text the player sees during this event
#fqid - the fully qualified ID of the event
#room_fqid - the fully qualified ID of the room the event took place in
#text_fqid - the fully qualified ID of the
#fullscreen - whether the player is in fullscreen mode
#hq - whether the game is in high-quality
#music - whether the game music is on or off
#level_group - which group of levels - and group of questions - this row belongs to (0-4, 5-12, 13-22)



# Separate numerical features and categorical features and new_data created.
# not taking some sparse data and some incomplete data
# counting unique number of things for the categorical features
categorical_features = ['event_name', 'fqid', 'room_fqid', 'text', 'text_fqid', 'name']
# As these are the numerical events, counting the mean and the standard deviation and making new features by linear dimension
# Not considering the quadratic and higher dimension for the computational benefit 
numerical_features = ['elapsed_time','level','hq','fullscreen','music']
# Counting the number of same events in the given session id within each question
event_var = ['elapsed_time', 'navigate_click','person_click','cutscene_click','object_click','checkpoint','map_click',
             'map_hover','notebook_click','notification_click']


# In[12]:


print(train_label)


# In[13]:


# Create a histogram of the 'correct' column
plt.hist(train_label['correct'])

# Add axis labels and a title
plt.xlabel('Correct')
plt.ylabel('Frequency')
plt.title('Distribution of Correct Answers')
plt.show()


# In[10]:


print(train_label.shape)


# In[11]:


def Define_features(data):
    
    # Initialize an empty list to store the feature dataframes
    new_frame = []
    
    # Categorical features
    for c in categorical_features:
        # Group by session_id and level_group, and count the unique values of the categorical feature
        cat = pd.DataFrame(data.groupby(['session_id', 'level_group'])[c].nunique())
        cat = cat.rename(columns={c: f"{c}_nunique"})
        new_frame.append(cat)
    
    # Numerical features - mean
    for c in numerical_features:
        # Group by session_id and level_group, and calculate the mean of the numerical feature
        num = pd.DataFrame(data.groupby(['session_id', 'level_group'])[c].mean())
        num = num.rename(columns={c: f"{c}_mean"})
        new_frame.append(num)
        numf = pd.DataFrame(data.groupby(['session_id', 'level_group'])[c].std())
        numf = numf.rename(columns={c: f"{c}_std"})
        new_frame.append(numf)
       
    # Event variables
    for c in event_var:
        # Group by session_id and level_group, and sum the values of the event variable
        eve = pd.DataFrame(data.groupby(['session_id', 'level_group'])[c].sum())
        eve = eve.rename(columns={c: f"{c}_sum"})
        new_frame.append(eve)
    
    new_frame = pd.concat(new_frame,axis=1)
    new_frame = new_frame.fillna(-1)
    new_frame = new_frame.reset_index()
    new_frame = new_frame.set_index('session_id')

    return new_frame

# Feature Engineer Train 
df_tr = Define_features(train)
print(df_tr.shape)

#Splitting the data into train and test
df_tr_train = df_tr.iloc[:63615, :]
train_label_train = train_label.iloc[:381690, :]
df_tr_test = df_tr.iloc[63615:, :]
train_label_test = train_label.iloc[381690:, :]

# Define features 
FEATURES = [c for c in df_tr_train.columns if c != 'level_group']
ALL_USERS = df_tr_train.index.unique()
print('We will train with', len(FEATURES) ,'features and ', len(ALL_USERS) ,'users info')
pred = pd.DataFrame(data=np.zeros((len(ALL_USERS),18)), index=ALL_USERS)

# Define features 
FEATURES1 = [c for c in df_tr_test.columns if c != 'level_group']
ALL_USERS1 = df_tr_test.index.unique()
print('We will test with', len(FEATURES1) ,'features and ', len(ALL_USERS1) ,'users info')
pred_test = pd.DataFrame(data=np.zeros((len(ALL_USERS1),18)), index=ALL_USERS1)


# In[4]:


n_estimators = [100,200,300,400,500,600,700,800,900,1000 ]
THR=[]
acc=[]
bF1=[]
models = {}
for ne in n_estimators:
    xgb_params = {       
        'learning_rate': 0.0001,              # can vary
        'max_depth': 7,                   # can vary 
        'n_estimators': ne,             # can vary 
        'subsample':0.8,
        'colsample_bytree': 0.4, }        # subsampling ratio 

    for t in range(1,19):

        # USE THIS TRAIN DATA WITH THESE QUESTIONS
        if t<=3: grp = '0-4'
        elif t<=13: grp = '5-12'
        elif t<=22: grp = '13-22'

        # TRAIN DATA
        train_x = df_tr_train[df_tr_train.level_group == grp]
        train_users = train_x.index.values
        train_y = train_label_train.loc[train_label.q==t].set_index('session').loc[train_users]
        test_x = df_tr_test[df_tr_test.level_group == grp]
        test_users = test_x.index.values

         # TRAIN MODEL
        clf =  XGBClassifier(**xgb_params)
        clf.fit(train_x[FEATURES].astype('float32'), train_y['correct'])

        # SAVE MODEL, PREDICT VALID OOF
        models[f'{grp}_{t}'] = clf
        pred.loc[train_users, t-1] = clf.predict_proba(train_x[FEATURES].astype('float32'))[:,1]

        #TEST DATA
        pred_test.loc[test_users, t-1] = clf.predict_proba(test_x[FEATURES].astype('float32'))[:,1]

    # PUT TRUE LABELS INTO DATAFRAME WITH 18 COLUMNS
    true = pred.copy()
    true_test=pred_test.copy()
    for k in range(18):
        # GET TRUE LABELS
        tmp = train_label_train.loc[train_label.q == k+1].set_index('session').loc[ALL_USERS]
        tmp1 = train_label_test.loc[train_label.q == k+1].set_index('session').loc[ALL_USERS1]
        true[k] = tmp.correct.values
        true_test[k] = tmp1.correct.values

    # FIND BEST THRESHOLD TO CONVERT PROBS INTO 1s AND 0s

    scores = []
    thresholds = []

    # Loop over a range of thresholds
    for threshold in np.arange(0.4, 0.81, 0.01):

        # Convert predicted probabilities to binary predictions using threshold
        binary_preds = (pred.values.reshape((-1)) > threshold).astype('int')

        # Calculate the macro F1 score between the true labels and the binary predictions
        f1 = f1_score(true.values.reshape((-1)), binary_preds, average='macro')

        # Append the score and threshold to the respective lists
        scores.append(f1)
        thresholds.append(threshold)

    # Find the threshold that maximizes the F1 score
    best_score = max(scores)
    best_threshold = thresholds[scores.index(best_score)]

    print(f"\nBest threshold: {best_threshold:.2f}")
    print(f"Best macro F1 score: {best_score:.5f}")
    THR.append(best_threshold)
    bF1.append(best_score)

    new_df = pred_test.applymap(lambda x: 1 if x >= best_threshold else 0)
    train_label_test['q'] = train_label_test['q'] - 1
    pr_df = train_label_test.pivot(index='session', columns='q', values='correct')

    accuracy=0
    count=0
    for i in range(18):
        for j in range(2357):
            if pr_df.iloc[j,i]==new_df.iloc[j,i]:
                accuracy+=1
            count=count+1

    print('Accuracy is', accuracy/count)
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")
    acc.append(accuracy/count)


# In[5]:


print(acc)
print(THR)
print(bF1)
print(n_estimators)


# In[6]:


plt.figure(figsize=(10, 5))
plt.plot(n_estimators, acc, '-o', color='blue')
plt.xlabel('n_estimators', size=14)
plt.ylabel('Accuracy', size=14)
plt.title('Testing Accuracy vs. n_estimators', size=18)
plt.grid(True)
plt.show()

# Plot threshold vs. learning rate
plt.figure(figsize=(10, 5))
plt.plot(n_estimators, THR, '-o', color='orange')
plt.xlabel('n_estimators', size=14)
plt.ylabel('Threshold', size=14)
plt.title('Training Threshold vs. n_estimators', size=18)
plt.grid(True)
plt.show()

# Plot best F1 score vs. learning rate
plt.figure(figsize=(10, 5))
plt.plot(n_estimators, bF1, '-o', color='green')
plt.xlabel('n_estimators', size=14)
plt.ylabel('Best F1 Score', size=14)
plt.title('Training Best F1 Score vs. n_estimators', size=18)
plt.grid(True)
plt.show()


# In[8]:


plt.figure(figsize=(6, 4))
plt.plot(n_estimators, acc, '-o', label='Accuracy')
plt.plot(n_estimators, bF1, '-o', label='Balanced F1 Score')
plt.plot(n_estimators, THR, '-o', label='Best Threshold')
plt.xlabel('n-estimators')
plt.ylabel('Score')
plt.title('n_estimators vs. Performance Metrics')
plt.grid()
plt.legend()
plt.show()


# In[9]:


plt.figure(figsize=(20,5))
plt.plot(thresholds,scores,'-o',color='blue')
plt.scatter([best_threshold], [best_score], color='red', s=100, alpha=1)
plt.xlabel('Threshold',size=14)
plt.ylabel('Validation F1 Score',size=14)
plt.title(f'Threshold vs. F1_Score with Best F1_Score = {best_score:.3f} at Best Threshold = {best_threshold:.3f}',size=18)
plt.show()


# In[ ]:




