# -*- coding: utf-8 -*-
"""
I'm modifying the web scraping code from JobFit to do market research on DS jobs
in Toronto, CA.
"""

#Import libraries
from bs4 import BeautifulSoup
from selenium  import webdriver
import datetime
import re
import pandas as pd

def get_soup(field,loc,start):
    '''
    Get html from landing page for a starting page index
    Pass keyword for job title, location of job and starting index
    '''
    driver = webdriver.Chrome('/Users/chris/Downloads/chromedriver')
    #loc is the location
    #field is the search term
    #The last concatenation lets us navigate through pages
    driver.get('https://www.indeed.ca/jobs?q='+field+'&l='+loc+'&start='+str(start))
    soup = BeautifulSoup(driver.page_source,"lxml")
    driver.close()
    return soup
    

def get_soup_post(url):
    '''
    Get html from a specific job post
    '''
    driver = webdriver.Chrome('/Users/chris/Downloads/chromedriver')
    driver.get(url)
    soup = BeautifulSoup(driver.page_source,"lxml")
    driver.close()
    return soup


def get_links(soup):
    '''
    Get the job post urls from the landing page html
    '''
    links = []
    for item in soup.find_all('a', href=True):
        m = re.search('/pagead/',str(item['href'])) #sponsored ads
        n = re.search('/rc/',str(item['href'])) #regular content
        o = re.search('/clk?',str(item['href']))
        if m is None:
            pass
        else:
            links.append('https://www.indeed.ca'+str(item['href']))
        if n is None:
            pass
        else:
            links.append('https://www.indeed.ca'+str(item['href']))
    return links


def get_jobsum(soup):
    '''
    Get job info from a specific url
    '''
    for item in soup.find_all(name='span', attrs={'class':'summary'}):
        if item is None:
            summary = ''
        else:
            summary = str(item.text.encode('utf-8').strip())
    return summary


def get_metadata(soup,field,loc):
    '''
    Get metadata from landing page html
    '''
    data = []
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d")
    for item in soup.find_all(name='div', attrs={'class':'row result clickcard'}):
        if item.find(name='span',attrs={'class':'company'}) is None:
            company_name = ''
        else:
            company_name = str(item.find(name='span',attrs={'class':'company'}).text.encode('utf-8').strip())
        if item.find(name='a', attrs={'data-tn-element':'jobTitle'}) is None:
            job_title= ''
        else:
            job_title = str(item.find(name='a', attrs={'data-tn-element':'jobTitle'})['title'])
        if item.find(name='span',attrs={'class':'date'}) is None:
            age = 'None'
        else:    
            age = str(item.find(name='span',attrs={'class':'date'}).text.encode('utf-8').strip())
        
        n = re.search('/rc/',str(item.find(name='a',attrs={'data-tn-element':'jobTitle'})['href']))
        if n is None:
            url = ''
        else:
            url_snip = item.find(name='a',attrs={'data-tn-element':'jobTitle'})['href']
            url_split = url_snip.split('?')
            url = 'https://www.indeed.ca/viewjob?'+url_split[1]
        
        entry = [job_title, company_name, age, date, field, loc, url]
        data.append(entry)
    return data


def get_summaries(data):
    '''
    Get job summaries from a list of urls/metadata
    '''
    for i in range(len(data)):
        URL_INDEX = 6 #Hardcoded in
        url = data[i][URL_INDEX] 
        if url == '':
            summary = ''
        else:
            #Open this page and scrape
            soup = get_soup_post(url)
            summary = get_jobsum(soup)
        data[i].append(summary)


def get_numposts(field,loc):
    '''
    Get the total number of posts
    '''
    driver = webdriver.Chrome('/Users/chris/Downloads/chromedriver')
    driver.get('https://www.indeed.ca/jobs?q='+field+'&l='+loc+'&start=0')
    soup = BeautifulSoup(driver.page_source,"lxml")
    driver.close()
    tot_num = str(soup.find(name='div', attrs={'id':'searchCount'}).text.encode('utf-8').strip())
    return int(tot_num.split()[3].replace(',',''))



if __name__ == "__main__":
    loc = 'toronto'
    field = 'data+scientist'
    #Set range for scraping posts
    start = 0
    end1 = get_numposts(field,loc)
    end2 = 200
    end = min(end1,end2)
    num_per_page = 10
    indices = range(start,end,num_per_page)
    #Scrape pages
    data = []
    for i in indices:
        soup = get_soup(field, loc, i)
        data.extend(get_metadata(soup,field,loc))
    
    #Get summaries of posts
    get_summaries(data)
    df = pd.DataFrame(data)
    print(df.head())