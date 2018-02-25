#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 10:10:34 2018

Working version of database is ONET_db_v1

ONET_db_v2 is going to include alternate titles for SQL search

@author: christopherluciuk
"""

from flask import render_template, request, url_for
from flaskapp import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from flaskapp.clustermodel import score_btw_jobs, skills_rec, scrape_indeed, summaries_to_activities, parse_title, acts_to_skills, get_wordcloud, search_titles, get_model_coeff, get_activity_from_skill

# Python code to connect to Postgres
# You may need to modify this based on your OS
user = 'christopherluciuk' #add your Postgres username here      
host = 'localhost'
dbname = 'ONET_db_v2'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET','POST'])
def index():
    '''
    Load the home page
    '''
    return render_template("index.html")

@app.route('/jobFit', methods=['GET','POST'])
def jobFit():
    '''
    Display the results of the transition
    '''
    #Colour code progress bar
    progress_dict = {1.0:'progress-bar-danger',2.0:'progress-bar-warning',3.0:'progress-bar-success'}
    #Number of recommendations to return
    num_rec = 3
    #Query database
    #NEED TRY EXCEPT
    skill_query = """                                                                       
                SELECT * FROM skills_table;          
                """
    skill_query_results = pd.read_sql_query(skill_query,con)
    #Get posted jobs
    job1_post = request.form.get('Job_1')
    job2_post = request.form.get('Job_2')
    #Search
    job1, flag1 = search_titles(job1_post,skill_query_results)
    job2, flag2 = search_titles(job2_post,skill_query_results)
    #Check for matching categories
    if job1 == job2:
        return render_template("index.html",error_msg='Jobs are too similar')
    if (flag1 == 0):
        if (flag2 == 0):
            return render_template("index.html",error_msg=job1_post+' and '+job2_post+' not found')
        else:
            return render_template("index.html",error_msg=job1_post+' not found')
    else:
        if (flag2 == 0):
            return render_template("index.html",error_msg=job2_post+' not found')
        else:
            pass
    #Calculation for redirect
    score, prob = score_btw_jobs(job1,job2,skill_query_results)
    #Scrape job posts
    search_title = parse_title(job2_post)
    summaries = scrape_indeed(search_title,'new+york')
    #Calculate activity vector
    activity_vector = summaries_to_activities(summaries)
    #Map to skills vector
    skills_vector = acts_to_skills(activity_vector)
    #Get recommendation
    skills, score_list, score_ind = skills_rec(job1,job2,skill_query_results,num_rec,skills_vector)
    #Get scores
    skill_ratings, hot = get_model_coeff(skills, skills_vector)
    #Colour code
    progress_val = []
    for i in range(len(score_ind)):
        progress_val.append(progress_dict[score_ind[i]])
    #Send to visualization
    img_alias = get_wordcloud(skills_vector) #Not used in this version
    work_act = get_activity_from_skill(skills)
    #work_act = ['Something', 'something else', 'the last something']
    return render_template("jobFit.html", job1=job1_post, job2=job2_post, score=score, skills_out = zip(skills,score_list,progress_val,skill_ratings,hot,work_act), img_alias = img_alias, prob = prob) 

@app.route('/aboutJobFit')
def aboutJobFit():
    '''
    About the program
    '''
    return render_template("aboutJobFit.html")

@app.route('/aboutMe')
def aboutMe():
    '''
    About me
    '''
    return render_template("aboutMe.html")

@app.route('/contact')
def contact():
    '''
    Contact information
    '''
    return render_template("contact.html")

@app.route('/TED1')
def ted1():
    '''
    Info about my TED talk project
    '''
    return render_template("about-datascience-TED.html")

@app.route('/TED2')
def ted2():
    '''
    Continued Info about my TED talk project
    '''
    return render_template("about-datascience-TED2.html")

@app.route('/Fermi')
def fermi():
    '''
    Info about my image analysis using neural networks
    '''
    return render_template("about-datascience-fermi.html")   

@app.route('/Hedge')
def hedge():
    '''
    Info about the hedging project
    '''
    return render_template("about-datascience-hedge.html")                  