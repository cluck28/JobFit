#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 01:39:25 2018

Working version of database is ONET_db_v2

@author: christopherluciuk
"""

import math
import pandas as pd
import numpy
import urllib
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import re
import nltk
from sklearn import linear_model
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
from wordcloud import WordCloud
import random
import pickle

#Model values
model_opts = pickle.load( open( "./flaskapp/static/model.p", "rb" ) )

#Model values
#model_opts = {'coef_': numpy.array([[ 0.27031311,  1.10743968,  1.21184168,  0.24537734, -0.0331161 ,
#         0.92906561,  1.07179237, -0.51962826, -0.41135867,  0.29007684,
#        -5.98340972,  0.3938587 , -1.25777498,  1.24891589,  1.59120592,
#         2.43866736, -2.06619454, -1.16598049,  0.710818  ,  0.22879241,
#        -0.58795895,  0.87304311, -1.5935178 ,  0.20676845,  0.54368865,
#         1.54420561,  0.12662517, -0.12539465,  2.85181239, -0.17144002,
#        -2.16192062, -1.63583483, -1.54464784, -0.23207396,  0.85823911]]), 'intercept_': numpy.array(3.9589530905817183), 'classes_': numpy.array([0, 1])}

#For SQL
user = 'christopherluciuk' #add your Postgres username here      
host = 'localhost'
dbname = 'ONET_db_v2'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)


def score_btw_jobs(job_title1,job_title2,query_results):
    '''
    For two jobs returns the probability of belonging to the positive class
    (i.e., the transition is possible) using a trained logistic regression
    '''
    data_out = query_results
    job1 = data_out[data_out['title']==job_title1]
    job2 = data_out[data_out['title']==job_title2]
    #Limits of df to select
    min_len = 1
    max_len = len(job1.iloc[0,:])-1
    #slice
    job1_vals = job1.iloc[:,min_len:max_len].values[0]
    job2_vals = job2.iloc[:,min_len:max_len].values[0]
    #calculate distance
    distance = job2_vals - job1_vals
    for i in range(len(distance)):
        dist = max(0,distance[i])
        distance[i] = dist
    distance = [distance]
    #Use logistic regression to predict possible/impossible
    logistic = linear_model.LogisticRegression()
    logistic.coef_ = model_opts['coef_']
    logistic.intercept_ = model_opts['intercept_']
    logistic.classes_ = model_opts['classes_']
    score = logistic.predict(distance)
    prob = logistic.predict_proba(distance)
    #Return image for possible vs impossible
    if score == 1:
        return 'smiley.png', prob[0][1]
    else:
        return 'frowney.png', prob[0][1]
    
'''
Determines the skills that the user should learn
Calculates the skills deficit for each skill job2_skill - job1_skill
Weights deficit by importance of skill to job2
Weights deficit by difficulty in skill space
Returns a rank for colour coding progress bars
'''
def skills_rec(job_title1,job_title2,query_results,num_rec,skills_vector):
    data_out = query_results
    job1 = data_out[data_out['title']==job_title1]
    job2 = data_out[data_out['title']==job_title2]
    #Limits of df to select
    min_len = 1
    max_len = len(job1.iloc[0,:])-1
    #slice
    job1_vals = job1.iloc[:,min_len:max_len].values[0]
    job2_vals = job2.iloc[:,min_len:max_len].values[0]
    #calculate deficit and weighted score
    skill_def = numpy.zeros(len(job1_vals))
    score = numpy.zeros(len(job1_vals))
    for i in range(len(job1_vals)):
        distance = max(0,job2_vals[i]-job1_vals[i])
        skill_def[i] = distance*job2_vals[i]*math.fabs(model_opts['coef_'][0][i])#*skills_vector[i])
        score[i] = min(1,(1 - distance/2.5))
    #sort and return indices of largest elements
    indices_max_def = numpy.argsort(skill_def)[-num_rec:][::-1]
    col_head = list(data_out)
    dist_col_head = col_head[min_len:max_len]
    skills_list = []
    scores_list = []
    score_ind = []
    #Colour codes
    for i in indices_max_def:
        skills_list.append(dist_col_head[i])
        scores_list.append(int(score[i]*100))
        if score[i]*100 < 33:
            score_ind.append(1.0)
        elif 33 <= score[i]*100 < 67:
            score_ind.append(2.0)
        else:
            score_ind.append(3.0)
    #Return name of skill, score, colour code
    return skills_list, scores_list, score_ind

'''
Scrapes the first page of Indeed.com given the job title of the transition
and returns a list of job post summaries
SLOW
To speed up could limit to 5, 3, or 1 job post
'''
def scrape_indeed(job_title,loc,start=0):
    #Check that job_title is formated correctly
    #Load landing page
    #NEED TRY EXCEPT
    page = urllib.request.urlopen('https://www.indeed.com/jobs?q='+job_title+'&l='+loc+'&start='+str(start))
    soup =  BeautifulSoup(page, 'html.parser')
    #Get all hyperlinks that correspond to unsponsored content
    links = []
    for item in soup.find_all(name='div', attrs={'data-tn-component':'organicJob'}):
        for tag in item.find_all(name='a', attrs={'data-tn-element':'jobTitle'}):
            #Match specific part of hyperref for links
            n = re.search('/clk?',str(tag['href'])) #regular content
            if n is None:
                pass
            else:
                links.append('https://www.indeed.com/viewjob'+str(tag['href'][7:]))
    #Get summaries from links
    #Kicked out if no user agent
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
    headers={'User-Agent':user_agent,}
    summaries = []
    #Get all summaries -- limit and count
    limit = 10
    count = 0
    for link in links:
        #NEED TRY EXCEPT
        request = urllib.request.Request(link, None, headers)
        response = urllib.request.urlopen(request)
        post_soup = BeautifulSoup(response,'html.parser')
        for item in post_soup.find_all(name='span', attrs={'class':'summary'}):
            if item is None:
                pass
            else:
                summaries.append(str(item.text.strip()))
        count += 1
        if count == limit:
            break
    return summaries

'''
Computes the skills vector from the job summaries
Need to optimize for speed
'''
def summaries_to_activities(summaries):
    DWA_query = """                                                                       
                    SELECT * FROM dwa_table;          
                  """
    df_DWA = pd.read_sql_query(DWA_query,con)
    IWA_query = """                                                                       
                    SELECT * FROM iwa_table;          
                  """
    df_IWA = pd.read_sql_query(IWA_query,con)
    acts_query = """                                                                       
                    SELECT * FROM acts_table;          
                  """
    df_WorkAct = pd.read_sql_query(acts_query,con)
    
    #df_WorkAct = pd.read_excel('/Users/christopherluciuk/Desktop/Insight/JobsData/db_22_1_excel/Work Activities.xlsx')
    #df_DWA = pd.read_excel('/Users/christopherluciuk/Desktop/Insight/JobsData/db_22_1_excel/DWA Reference.xlsx')
    #df_IWA = pd.read_excel('/Users/christopherluciuk/Desktop/Insight/JobsData/db_22_1_excel/IWA Reference.xlsx')

    activities = list(df_WorkAct)[1:len(df_WorkAct.iloc[0,:])-1]
    #activities = df_WorkAct['Element Name'].unique()
    #Load all work activities
    scores = []
    for activity in activities:
        #Build activities dictionary
        IWAs = df_IWA[df_IWA['Element Name'] == activity]['IWA Title'].values
        DWAs = df_DWA[df_DWA['Element Name'] == activity]['DWA Title'].values
        task_dict = numpy.append(IWAs,DWAs)
        task_dict = numpy.append(task_dict,activity)
        #Turn into a tokenized list of words to do anlysis on (plus minus stop words and stuff)
        tokens = []
        for sent in task_dict:
            tokens.extend(nltk.word_tokenize(sent))
        filtered1 = filter_stopwords(tokens)
        WA_dict = nltk.FreqDist(filtered1).most_common(10)
        #Create work activities vector for each summary
        score = []
        for summary in summaries:
            #Remove stop words
            summary = filter_stopwords(clean_summary(summary))
            counter = 0
            for word in summary:
                for item in WA_dict:
                    if word == item[0]:
                        counter += 1
            try:
                score.append(counter/(len(summary)*len(WA_dict)))
            except ZeroDivisionError:
                score.append(0)
        try:
            scores.append(sum(score)/len(score))
        except ZeroDivisionError:
            scores.append(0)
    return scores
            
def filter_stopwords(text):
    '''
    Text preprocessing removing stopwords
    '''
    stopwords = stopwords = set('a about above after again against all am an and any are as at be because been before being below between both but by cannot could did do does doing down during each few for from further had has have having he her here hers herself him himself his how i if in into is it its itself me more most my myself no nor not of off on once only or other ought our ours ourselves out over own same she should so some such than that the their theirs them themselves then there these they this those through to too under until up very was we were what when where which while who whom why with would you your yours yourself yourselves . , : ; / ? ( )'.split())
    filtered_words = [w for w in text if w not in stopwords]
    filtered_words1 = [w for w in filtered_words if len(w)>1]
    return filtered_words1

def clean_summary(summary):
    '''
    Text preprocessing to clean job post
    '''
    sum_test = re.sub(r"([A-Z])", r" \1", summary)
    words_list = nltk.word_tokenize(sum_test.lower())
    return words_list

def get_wordcloud(skills_vector):
    '''
    Creates a word cloud of skills from the job posts
    Not currently used in final product
    '''
    #Save file in static
    filename = '/Users/christopherluciuk/Desktop/Insight/JobFit_app/flaskapp/static/wordcloud.png'
    #Load
    skill_query = """                                                                       
                    SELECT * FROM skills_table;          
                  """
    df_skills = pd.read_sql_query(skill_query,con)
    #Slice
    min_len = 1
    max_len = len(df_skills.iloc[0,:])-1
    skillsList = list(df_skills.iloc[:,min_len:max_len])
    word_cloud_list = {}
    for i in range(len(skillsList)):
        try:
            word_cloud_list.append((skillsList[i],skills_vector[i]))
        except:
            word_cloud_list[skillsList[i]] = 1
    word_cloud = WordCloud().fit_words(word_cloud_list)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(filename)
    return random.randrange(100000)

def parse_title(job_title):
    '''
    Cleaning of job title
    '''
    pieces = job_title.split()
    title = ''
    count = 0
    for piece in pieces:
        if count == 0:
            title += piece.lower()
        else:
            title += '+'+piece.lower()
        count += 1
    return title

def acts_to_skills(acts_vector):
    '''
    Matrix transformation from work activities to skills
    '''
    #Load
    skill_query = """                                                                       
                    SELECT * FROM skills_table;          
                  """
    act_query = """                                                                       
                    SELECT * FROM acts_table;          
                  """
    df_skills = pd.read_sql_query(skill_query,con)
    df_acts = pd.read_sql_query(act_query,con)
    #Slice
    min_len = 1
    max_len = len(df_skills.iloc[0,:])-1
    skillsMat = df_skills.iloc[:,min_len:max_len].values
    min_len = 1
    max_len = len(df_acts.iloc[0,:])-1
    actsMat = df_acts.iloc[:,min_len:max_len].values
    #Normalize
    norm_actsMat = numpy.linalg.norm(actsMat)
    norm_skillsMat = numpy.linalg.norm(skillsMat)
    #Take dot product
    transformation_mat = numpy.dot(skillsMat.T/norm_skillsMat,actsMat/norm_actsMat)
    norm_acts = numpy.linalg.norm(acts_vector)
    return numpy.dot(transformation_mat,acts_vector) #/norm_acts)

def search_titles(test_title,query_results):
    '''
    Find titles in the ONET database
    '''
    title_query = """
                    SELECT * FROM alttitles_table 
                """
    data_out = pd.read_sql_query(title_query,con)
    success_flag = 0
    #Is the searched title in the database?
    for title in data_out['Alternate Title']:
        if title.lower() == test_title.lower():
            common_title = data_out[data_out['Alternate Title']==title]['Title']
            success_flag = 1
            break
    #Check that the title has skills associated
    if success_flag == 1:
        if (query_results['title'] == common_title.values[0]).any():
            success_flag = 1
        else:
            success_flag = 0
    if success_flag == 1:
        return common_title.values[0], success_flag
    else:
        return '', success_flag
    
def get_model_coeff(skills, skills_vector):
    '''
    Load in trained model
    Used to rank the difficulty of skills
    Not used in final product
    '''
    skill_query = """                                                                       
                    SELECT * FROM skills_table;          
                  """
    df_skills = pd.read_sql_query(skill_query,con)
    min_len = 1
    max_len = len(df_skills.iloc[0,:])-1
    skillsList = list(df_skills.iloc[:,min_len:max_len])
    skill_ratings = []
    hot = []
    for i in range(len(skillsList)):
        for skill in skills:
            if skill == skillsList[i]:
                #Get ranking
                x = math.fabs(model_opts['coef_'][0][i])
                if 0 < x < 0.42:
                    skill_ratings.append('stars_1')
                elif 0.5 < x < 1.25:
                    skill_ratings.append('stars_2')
                else:
                    skill_ratings.append('stars_3')
                #Is the skill hot?
                if skills_vector[i] > numpy.mean(skills_vector):
                    hot.append('chilli')
                else:
                    hot.append('none')
    return skill_ratings, hot

def get_activity_from_skill(skills):
    '''
    Get one representative work activity from ONET for each skill
    '''
    activities = []
    for skill in skills:
        if skill == 'Reading Comprehension':
            activities.append('Read documents or materials to inform work processes.')
        elif skill == 'Active Listening':
            activities.append('Provide information or assistance to the public.')
        elif skill == 'Writing': 
            activities.append('Compile records, documentation, or other data.')
        elif skill == 'Speaking':
            activities.append('Interview people to obtain information.')
        elif skill == 'Mathematics':
            activities.append('Analyze scientific or applied data using mathematical principles.')
        elif skill == 'Science': 
            activities.append('Develop scientific or mathematical theories or models.')
        elif skill == 'Critical Thinking': 
            activities.append('Develop research plans or methodologies.')
        elif skill == 'Active Learning':
            activities.append('Attend training to learn new skills or update knowledge.')
        elif skill == 'Learning Strategies': 
            activities.append('Create technology-based learning materials.')
        elif skill == 'Monitoring':
            activities.append('Replenish inventories of materials, equipment, or products.')
        elif skill == 'Social Perceptiveness':
            activities.append('Assess living, work, or social needs or status of individuals or communities.')
        elif skill == 'Coordination':
            activities.append('Communicate with other workers to coordinate activities.')
        elif skill == 'Persuasion':
            activities.append('Resolve employee or contractor problems.')
        elif skill == 'Negotiation':
            activities.append('Negotiate contracts or agreements.')
        elif skill == 'Instructing': 
            activities.append('Teach academic or vocational subjects.')
        elif skill == 'Service Orientation':
            activities.append('Recommend products or services to customers.')
        elif skill == 'Complex Problem Solving':
            activities.append('Analyze data to identify or resolve operational problems.')
        elif skill == 'Operations Analysis':
            activities.append('Gather data about operational or development activities.')
        elif skill == 'Technology Design':
            activities.append('Research technology designs or applications.')
        elif skill == 'Equipment Selection': 
            activities.append('Inspect facilities or equipment.')
        elif skill == 'Installation':
            activities.append('Install energy or heating equipment.')
        elif skill == 'Programming':
            activities.append('Program computer systems or production equipment.')
        elif skill == 'Operation Monitoring':
            activities.append('Maintain operational records.')
        elif skill == 'Operation and Control':
            activities.append('Prepare reports of operational or procedural activities.')
        elif skill == 'Equipment Maintenance':
            activities.append('Replenish inventories of materials, equipment, or products.')
        elif skill == 'Troubleshooting':
            activities.append('Troubleshoot equipment or systems operation problems.')
        elif skill == 'Repairing':
            activities.append('Calculate requirements for equipment installation or repair projects.')
        elif skill == 'Quality Control Analysis':
            activities.append('Evaluate the quality or accuracy of data.')
        elif skill == 'Judgment and Decision Making':
            activities.append('Determine resource needs of projects or operations.')
        elif skill == 'Systems Analysis': 
            activities.append('Develop data analysis or data management procedures.')
        elif skill == 'Systems Evaluation':
            activities.append('Develop procedures to evaluate organizational activities.')
        elif skill == 'Time Management':
            activities.append('Estimate time or monetary resources needed to complete projects.')
        elif skill == 'Management of Financial Resources':
            activities.append('Assess financial status of clients.')
        elif skill == 'Management of Material Resources':
            activities.append('Develop plans to manage natural or renewable resources.')
        elif skill == 'Management of Personnel Resources':
            activities.append('Maintain personnel records.')
    return activities

if __name__ == "__main__":
    #summaries = scrape_indeed('chief+executives','new+york')
    #Calculate activity vector
    #activity_vector = summaries_to_activities(summaries)
    #Map to skills vector
    #skills_vector = acts_to_skills(activity_vector)
    #Send to visualization
    #get_wordcloud(skills_vector)
    #skills = ['Mathematics', 'Reading Comprehension', 'Active Learning']
    #print(get_model_coeff(skills))
    #print(type(model_opts['coef_'][0]))
    print(numpy.sort(numpy.abs(model_opts['coef_'][0])))
    