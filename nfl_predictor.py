__author__ = 'Aaron'

import numpy as np
import pandas as pd
import csv

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn import metrics
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')


##  prints a simple baseline where the home team is always selected as the winner
def simple_baseline(test_data_set_info, test_df):

    baseline_df = test_df.loc[:,['Week','Home_Team','Away_Team','W/L']]
    baseline_df.rename(columns = {'W/L':'Actual'}, inplace = True)
    baseline_df.index = pd.RangeIndex(len(baseline_df.index))

    baseline_prediction = {}
    for i in range(len(baseline_df.index)):
        baseline_prediction[i] = 1

    baseline_series = pd.Series(baseline_prediction)
    ## create new column for home team name
    baseline_df['Prediction'] = baseline_series
    ## swap the Prediction and Actual columns
    swapColumns = ['Week','Home_Team', 'Away_Team', 'Prediction', 'Actual']
    baseline_df = baseline_df.reindex(columns = swapColumns)

    ## convert prediction and actual columns to float so they can be used for metrics
    baseline_df['Prediction'] = pd.to_numeric(baseline_df['Prediction'], errors='coerce')
    baseline_df['Actual'] = pd.to_numeric(baseline_df['Actual'], errors='coerce')

    ## store y_true and y_pred for getting metrics
    y_true = baseline_df['Actual']
    y_pred = baseline_df['Prediction']

    ## change the 1 to home and 0 to away before displaying baseline df
    baseline_df.replace({'Actual': {1: 'Home', 0: 'Away'}, 'Prediction': {1: 'Home', 0: 'Away'}},inplace=True)

    print("\n   ***** BASELINE RESULTS ******    \n")
    ## display baseline df
    print(baseline_df)
    ## display metrics for baseline
    display_metrics(test_data_set_info, y_true, y_pred)

def make_predictions_round_one(test_data_set_info, train_df, test_df):

    ## divide the x and y values for training and testing data
    X_train, Y_train, X_test, Y_test = x_y_divide(train_df,test_df)



    ## ***** Default parameters for logistic regression below ***** ##
    ## LogisticRegression(penalty=’l2’, dual=False, tol=0.0001, C=1.0, fit_intercept=True,
    ##  intercept_scaling=1, class_weight=None, random_state=None, solver=’liblinear’, max_iter=100,
    ##  multi_class=’ovr’, verbose=0, warm_start=False, n_jobs=1

    ## logistic regression begin with c = 1, no regularization
    lr = LogisticRegression(C=1., solver='liblinear')

    ## fit the model
    lr.fit(X_train,Y_train.values.ravel())

    ## make game predictions
    Y_pred = lr.predict(X_test)
    Y_pred_df = pd.DataFrame({'Prediction':Y_pred})
    Y_pred_df.index = Y_test.index

    result = test_df
    result.index = Y_test.index
    result = pd.concat([result, Y_pred_df], axis=1, join_axes=[result.index])

    ## find the probabilities of home teams winning
    probability_home_team = lr.predict_proba(X_test)[:,1]
    probability_home_team_df = pd.DataFrame({'Prob_Home':probability_home_team*100})
    probability_home_team_df.index = Y_test.index

    ## find the probabilities of away teams winning
    probability_away_team = lr.predict_proba(X_test)[:,0]
    probability_away_team_df = pd.DataFrame({'Prob_Away':probability_away_team*100})
    probability_away_team_df.index = Y_test.index

    ## concat home and away probability columns to result
    result = pd.concat([result, probability_home_team_df], axis=1, join_axes=[result.index])
    result = pd.concat([result, probability_away_team_df], axis=1, join_axes=[result.index])

    ## create new data frame for clean looking display to user
    display_outcome = result.loc[:,['Week','Home_Team','Away_Team','Prob_Home','Prob_Away','Prediction','W/L']]
    display_outcome.rename(columns = {'W/L':'Actual'}, inplace = True)
    display_outcome.replace({'Actual': {'1': 'Home', '0': 'Away'}, 'Prediction': {'1': 'Home', '0': 'Away'}},inplace=True)

    print(display_outcome)

    Y_test = pd.to_numeric(Y_test['W/L'], errors='coerce')
    Y_pred = pd.to_numeric(Y_pred, errors='coerce')

    display_metrics(test_data_set_info, Y_test, Y_pred)


def make_predictions_round_two(test_data_set_info, train_df, test_df):

    X_train, Y_train, X_test, Y_test = x_y_divide(train_df,test_df)

    ## logistic regression
    #lr = LogisticRegression(C=.001, solver='liblinear')

    lr = LogisticRegression()

    param_grid = {'solver':['newton-cg', 'lbfgs', 'liblinear']}

    ## use grid search to find the best C and penalty parameters for the model
    clf = GridSearchCV(lr,param_grid,refit=True)

    ## fit the model
    clf.fit(X_train,Y_train.values.ravel())

    #clf.fit(X_train,Y_train.values.ravel())

    ## make game predictions
    Y_pred = clf.predict(X_test)

    ## fit the model
    #lr.fit(X_train,Y_train.values.ravel())

    Y_pred_df = pd.DataFrame({'Prediction':Y_pred})

    Y_pred_df.index = Y_test.index

    result = test_df
    result.index = Y_test.index
    result = pd.concat([result, Y_pred_df], axis=1, join_axes=[result.index])

    ## find the probabilities of home teams winning
    probability_home_team = clf.predict_proba(X_test)[:,1]
    probability_home_team_df = pd.DataFrame({'Prob_Home':probability_home_team*100})
    probability_home_team_df.index = Y_test.index

    ## find the probabilities of away teams winning
    probability_away_team = clf.predict_proba(X_test)[:,0]
    probability_away_team_df = pd.DataFrame({'Prob_Away':probability_away_team*100})
    probability_away_team_df.index = Y_test.index

    ## concat home and away probability columns to result
    result = pd.concat([result, probability_home_team_df], axis=1, join_axes=[result.index])
    result = pd.concat([result, probability_away_team_df], axis=1, join_axes=[result.index])

    ## create new data frame for clean looking display to user
    display_outcome = result.loc[:,['Week','Home_Team','Away_Team','Prob_Home','Prob_Away','Prediction','W/L']]
    display_outcome.rename(columns = {'W/L':'Actual'}, inplace = True)
    display_outcome.replace({'Actual': {'1': 'Home', '0': 'Away'}, 'Prediction': {'1': 'Home', '0': 'Away'}},inplace=True)


    print(display_outcome)

    Y_test = pd.to_numeric(Y_test['W/L'], errors='coerce')
    Y_pred = pd.to_numeric(Y_pred, errors='coerce')

    display_metrics(test_data_set_info, Y_test, Y_pred)
    print('\t Best parameter for solver: ', clf.best_params_)

def make_predictions_round_three(train_df, test_df):

    # divide into x and y
    X_train, Y_train, X_test, Y_test = x_y_divide(train_df,test_df)

    y_test_temp = pd.to_numeric(Y_test['W/L'], errors='coerce')

    ## make logistic regression model
    lr = LogisticRegression(solver='liblinear',penalty='l2')

    ## lr c = .001
    lr_001 = LogisticRegression(C=.001,solver='liblinear')
    lr_001.fit(X_train,Y_train)
    pred = lr_001.predict(X_test)
    y_pred_temp = pd.to_numeric(pred, errors='coerce')
    print("\t C = .001 Accuracy: %.3f%%" % (metrics.accuracy_score(y_test_temp, y_pred_temp)*100))

    lr_01 = LogisticRegression(C=.01,solver='liblinear')
    lr_01.fit(X_train,Y_train)
    pred = lr_01.predict(X_test)
    y_pred_temp = pd.to_numeric(pred, errors='coerce')
    print("\t C = .01 Accuracy: %.3f%%" % (metrics.accuracy_score(y_test_temp, y_pred_temp)*100))

    lr_dot_1 = LogisticRegression(C=.1,solver='liblinear')
    lr_dot_1.fit(X_train,Y_train)
    pred = lr_dot_1.predict(X_test)
    y_pred_temp = pd.to_numeric(pred, errors='coerce')
    print("\t C = .1 Accuracy: %.3f%%" % (metrics.accuracy_score(y_test_temp, y_pred_temp)*100))

    lr_1 = LogisticRegression(C=1,solver='liblinear')
    lr_1.fit(X_train,Y_train)
    pred = lr_1.predict(X_test)
    y_pred_temp = pd.to_numeric(pred, errors='coerce')
    print("\t C = 1 Accuracy: %.3f%%" % (metrics.accuracy_score(y_test_temp, y_pred_temp)*100))

    lr_10 = LogisticRegression(C=10,solver='liblinear')
    lr_10.fit(X_train,Y_train)
    pred = lr_10.predict(X_test)
    y_pred_temp = pd.to_numeric(pred, errors='coerce')
    print("\t C = 10 Accuracy: %.3f%%" % (metrics.accuracy_score(y_test_temp, y_pred_temp)*100))

    lr_100 = LogisticRegression(C=100,solver='liblinear')
    lr_100.fit(X_train,Y_train)
    pred = lr_100.predict(X_test)
    y_pred_temp = pd.to_numeric(pred, errors='coerce')
    print("\t C = 100 Accuracy: %.3f%%" % (metrics.accuracy_score(y_test_temp, y_pred_temp)*100))

    lr_1000 = LogisticRegression(C=1000,solver='liblinear')
    lr_1000.fit(X_train,Y_train)
    pred = lr_1000.predict(X_test)
    y_pred_temp = pd.to_numeric(pred, errors='coerce')
    print("\t C = 1000 Accuracy: %.3f%%" % (metrics.accuracy_score(y_test_temp, y_pred_temp)*100))

    '''
    plt.plot(lr_001.coef_.T, 'o', label='C=1')
    plt.xticks(range(len(X_train.columns)),X_train.columns, rotation=90)
    plt.hlines(0,0,X_train._data.shape[1])
    plt.ylim(-5,5)
    plt.xlabel('Coef')
    plt.ylabel('Coef mag')
    plt.legend()
    plt.show()
    '''

    ## dictionary of parameters as the keys and parameters settings to try
    param_grid = {'C':[.001,.01,.1,1,10,100,1000]}

    ## use grid search to find the best C and penalty parameters for the model
    clf = GridSearchCV(lr,param_grid,refit=True)

    ## fit the model
    clf.fit(X_train,Y_train.values.ravel())

    print('\t Best parameter for solver: ', clf.best_params_)



def make_predictions_round_four(test_data_set_info, train_df, test_df):

    X_train, Y_train, X_test, Y_test = x_y_divide(train_df,test_df)

    ## logistic regression
    #lr = LogisticRegression(C=.001, solver='liblinear')

    lr = LogisticRegression(solver='liblinear', C=.001)

    param_grid = {'penalty':['l1','l2']}

    ## use grid search to find the best C and penalty parameters for the model
    clf = GridSearchCV(lr,param_grid,refit=True)

    ## fit the model
    clf.fit(X_train,Y_train.values.ravel())

    #clf.fit(X_train,Y_train.values.ravel())

    ## make game predictions
    Y_pred = clf.predict(X_test)

    ## fit the model
    #lr.fit(X_train,Y_train.values.ravel())

    Y_pred_df = pd.DataFrame({'Prediction':Y_pred})

    Y_pred_df.index = Y_test.index

    result = test_df
    result.index = Y_test.index
    result = pd.concat([result, Y_pred_df], axis=1, join_axes=[result.index])

    ## find the probabilities of home teams winning
    probability_home_team = clf.predict_proba(X_test)[:,1]
    probability_home_team_df = pd.DataFrame({'Prob_Home':probability_home_team*100})
    probability_home_team_df.index = Y_test.index

    ## find the probabilities of away teams winning
    probability_away_team = clf.predict_proba(X_test)[:,0]
    probability_away_team_df = pd.DataFrame({'Prob_Away':probability_away_team*100})
    probability_away_team_df.index = Y_test.index

    ## concat home and away probability columns to result
    result = pd.concat([result, probability_home_team_df], axis=1, join_axes=[result.index])
    result = pd.concat([result, probability_away_team_df], axis=1, join_axes=[result.index])

    ## create new data frame for clean looking display to user
    display_outcome = result.loc[:,['Week','Home_Team','Away_Team','Prob_Home','Prob_Away','Prediction','W/L']]
    display_outcome.rename(columns = {'W/L':'Actual'}, inplace = True)
    display_outcome.replace({'Actual': {'1': 'Home', '0': 'Away'}, 'Prediction': {'1': 'Home', '0': 'Away'}},inplace=True)


    print(display_outcome)

    Y_test = pd.to_numeric(Y_test['W/L'], errors='coerce')
    Y_pred = pd.to_numeric(Y_pred, errors='coerce')

    display_metrics(test_data_set_info, Y_test, Y_pred)
    print('\t Best parameter for intercept scaling: ', clf.best_params_)


def x_y_divide(train_df, test_df):

    #X_train = train_df.loc[:,['HomeAway','HT_Wins','HT_Losses','HT_Avg_Pts','HT_Avg_Pts_Against']]

    X_train = train_df.iloc[:,[3]]
    temp = train_df.iloc[:,6:81]
    X_train = pd.concat([X_train, temp], axis=1, join_axes=[X_train.index])


    Y_train = train_df.loc[:,['Week','W/L']]
    Y_train.index = pd.RangeIndex(len(Y_train.index))
    Y_train = Y_train.drop('Week', axis=1)

    #X_test = test_df.loc[:,['HomeAway','HT_Wins','HT_Losses','HT_Avg_Pts','HT_Avg_Pts_Against']]

    #X_test = test_df.iloc[:,[3,6,7,8,9,10,11]]

    X_test = test_df.iloc[:,[3]]
    temp = test_df.iloc[:,6:81]
    X_test = pd.concat([X_test, temp], axis=1, join_axes=[X_test.index])

    Y_test = test_df.loc[:,['Week','W/L']]
    Y_test.index = pd.RangeIndex(len(Y_test.index))

    return X_train, Y_train, X_test, Y_test


def display_metrics(test_name, y_true, y_pred):

    ## display precision
    print("\n\t",test_name," Precision: %.3f%%" % (metrics.precision_score(y_true, y_pred)*100.00))
    ## display recall
    print("\t",test_name," Recall: %.3f%%" % (metrics.recall_score(y_true, y_pred)*100.00))
    ## display accuracy
    print("\t",test_name," Accuracy: %.3f%%" % (metrics.accuracy_score(y_true, y_pred)*100.00))
    ## display f-1 measure
    print("\t",test_name," F-1 Measure: %.3f%%" % (metrics.f1_score(y_true, y_pred)*100.00))

if __name__ == '__main__':

    ## load nfl season 2014
    season_2014 = pd.read_csv('nfl_schedule_2014.csv',header=[0])

    ## load nfl seasons 2014-2015
    season_2014_2015 =  pd.read_csv('nfl_2014_2015.csv',header=[0])

    ## load nfl season 2014-2016
    season_2014_2015_2016 =  pd.read_csv('nfl_2014_2015_2016.csv',header=[0])

    ## load nfl season 2014-2017
    season_2014_2015_2016_2017 =  pd.read_csv('nfl_2014_2015_2016_2017.csv',header=[0])

    ## index df according by week
    season_2014.index = season_2014.Week

    ####                   NFL Season 2014                ####
    ####                                                  ####
    #### 1) Train Weeks 1 to 15                           ####
    #### 2) Validate Week 16                              ####
    #### 3) Retrain Weeks 1 to 16                         ####
    #### 4) Test Week 17                                  ####
    ####                                                  ####
    ##########################################################
    train_df = season_2014[season_2014['Week'] < 16]
    validation_df = season_2014[season_2014['Week'] == 16]

    print("\n   ***** Validation Data 2014 ******    \n")
    ## validate season 2014 week 16
    make_predictions_round_one('Validation data NFL 2014 Week 16',train_df , validation_df)

    #print("Round Three")
    #make_predictions_round_three(train_df, validation_df)

    train_df = season_2014[season_2014['Week'] < 17]
    test_df = season_2014[season_2014['Week'] == 17]

    print("\n   ***** Test Data 2014 ******    \n")
    ## test season 2014
    make_predictions_round_one('Test data NFL 2014 Week 17',train_df, test_df)

    ## call simple baseline for test data 2014 for comparison
    simple_baseline('Baseline NFL 2014 Week 17',test_df)

    ####                   NFL Season 2015                ####
    ####                                                  ####
    #### 1) Train 2014 Wk 1 - 17  & 2015 Wk 1 - 15        ####
    #### 2) Validate 2015 Wk 16                           ####
    #### 3) Retrain 2015 Wk 1 to 16                       ####
    #### 4) Test 2015 Week 17                             ####
    ####                                                  ####
    ##########################################################
    train_df = season_2014_2015[:480]
    validation_df = season_2014_2015[480:496]
    print("\n   ***** Validation Data 2015 ******    \n")
    make_predictions_round_two('Validation data NFL 2015 Week 16',train_df, validation_df)

    train_df = season_2014_2015[:496]
    test_df = season_2014_2015[496:]
    print("\n   ***** Test Data 2015 ******    \n")
    make_predictions_round_two('Test data NFL 2015 Week 17',train_df,test_df)

    ## call simple baseline for test data 2015 for comparison
    simple_baseline('Baseline NFL 2015 Week 17',test_df)

    ####                   NFL Season 2016                ####
    ####                                                  ####
    #### 1) Train 2014-15 Wk 1-17 & 2016 Wk 1-15          ####
    #### 2) Validate 2016 Wk 16                           ####
    #### 3) Retrain 2016 Wk 1 to 16                       ####
    #### 4) Test 2016 Week 17                             ####
    ####                                                  ####
    ##########################################################
    train_df = season_2014_2015_2016[:737]
    validation_df = season_2014_2015_2016[737:752]
    print("\n   ***** Validation Data 2016 ******    \n")
    #make_predictions_round_three('Validation data NFL 2016 Week 16',train_df, validation_df)
    make_predictions_round_three(train_df, validation_df)

    train_df = season_2014_2015_2016[:752]
    test_df = season_2014_2015_2016[752:]
    print("\n   ***** Test Data 2016 ******    \n")
    #make_predictions_round_three('Test data NFL 2016 Week 17',train_df,test_df)
    make_predictions_round_three(train_df,test_df)

    ## call simple baseline for test data 2016 for comparison
    simple_baseline('Baseline NFL 2016 Week 17',test_df)



    ####                   NFL Season 2017                ####
    ####                                                  ####
    #### 1) Train 2014-16 Wk 1-17 & 2017 Wk 1-10          ####
    #### 2) Validate 2017 Wk 11                           ####
    #### 3) Retrain 2017 Wk 1 to 11                       ####
    #### 4) Test 2016 Week 12                             ####
    ####                                                  ####
    ##########################################################
    train_df = season_2014_2015_2016_2017[:915]
    validation_df = season_2014_2015_2016_2017[915:928]
    print("\n   ***** Validation Data 2017 ******    \n")
    make_predictions_round_four('Validation data NFL 2017 Week 10',train_df, validation_df)

    train_df = season_2014_2015_2016_2017[:928]
    test_df = season_2014_2015_2016_2017[928:]
    print("\n   ***** Test Data 2017 ******    \n")
    make_predictions_round_four('Test data NFL 2017 Week 12',train_df,test_df)

    ## call simple baseline for test data 2017 for comparison
    simple_baseline('Baseline NFL 2017 Week 12',test_df)

    #make_predictions_round_three(train_df,test_df)



    '''
    print('Number columns = ', len(test_df.columns))
    col_test = test_df.iloc[:,[3,6,7,8,9,10,11]]
    print(col_test.head(n=10))
    '''



