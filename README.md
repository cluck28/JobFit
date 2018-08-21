# JobFit
The tool that I built is called JobFit. It is a web application that identifies the skills deficit separating a user’s current job from their dream job.

The pipeline is as follows:
<ol>
  <li>Extracted relevant information from the ONET database and store it in a series of tables in a postgreSQL database.</li>
  <li>Scrape resumes from Indeed.com to label job transitions.</li>
  <li>Scraped job posts from Indeed.com to identify trending skills.</li>
  <li>Train a model to predict whether a transition would be successful or not.</li>
  <li>Wrap model into engine of web application using Flask.</li>
  <li>User interface for input of jobs and display of results of model using Bootstrap (and Flask).</li>
  <li>Host on AWS.</li>
</ol>

<h1>Extract Relevant ONET Data</h1>
After performing data exploration and transforming some of the relevant features in the ONET database I saved the relevant features in a number of tables in a postgreSQL database. I had been working with separate .csv files and the python script to process the relevant data is ./DataProcessing/WriteDataForWeb.ipynb. A brief description of the script is contained in the notebook.

<h1>Scrape Resumes and Label Transitions</h1>
In this problem I needed a label for each job transition if it was possible or impossible. To generate this dataset I scraped resumes from Indeed.com. The script that scrapes the resumes is contained in ./DataProcessing/ScrapeResumes.ipynb.

<h1>Scrape Job Posts</h1>
The function that scrapes job posts is contained in the web application. I have included a stand alone script that processes job posts and saves them to a database in ./DataProcessing/IndeedScraper_v5.ipynb.

<h1>Train a Model</h1>
I train the model in the file ./DataProcessing/Logistic_v2.ipynb. The model results are pickled and can be uploaded to the server for use in the web application.

<h1>Web Application</h1>
The web application itself and the required html and scripts are contained in ./JobFit_app/
