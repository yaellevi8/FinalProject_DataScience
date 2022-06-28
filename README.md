# <font size="20"> <center> News article classification 

# Authors:  <a class="anchor" id="authors"></a>

<font size="4"> Yael Levi and Matan Maimon.

<font size="4"> Data Science Course, HIT  | Semester B | 29/06/2022

# Abstract <a class="anchor" id="abstract"></a>
## Can you classify articles using their content?

<font size="4"> <p style="color:SlateBlue">  News article classification is classifying news articles into various categories like travel, health, finance, etc. based on the content of the article. It is useful for people reading news to read all news of a particular category in one place or for people who are undecided whether to read an article.
    
<font size="4"> <p style="color:SlateBlue"> In this project, we are scraping a LOT of data (42K articles!) and train some machine learning models that will predict the category of an article using it content.

<font size="4"> <p style="color:SlateBlue">This has been achieved by representing articles as vectors through the use of methods like TF-IDF Vectorizer.

<font size="4"> <p style="color:SlateBlue"> After this, news articles can be classified using any suitable machine learning classification model, such as Logistic Regression, Multinomial Naive Bayes, Gaussian Naive Bayes and Linear SVC.

# Full Project is here:
 
 This readme does not have the graphs and images of our project.
 So we recommand to open this link and enjoy our full project:
 https://nbviewer.org/github/yaellevi8/FinalProject_DataScience/blob/main/final_project_text_classification.ipynb
 It might take a few minutes.
 OR open the file called "final_project_text_classification.html" ,Enjoy!
 
 
# Table of Content

## [Abstract](#abstract)
## [Import Necessary Libraries](#import)
## [Part 1. Scraping the data](#scrapingthedata)
   * [Html into beautiful soup object](#Htmlintobeautifulsoupobject)
   * [Inserting the raw data from the websites into a formatted CSV file](#InsertingtherawdatafromthewebsitesintoaformattedCSVfile)
   * [Scraping the data from each website](#Scrapingthedatafromeachwebsite)

## [Part 2. Initial cleaning data](#Part2.Initialcleaningdata)
   * [Removing rows and columns with Null/NaN values](#RemovingrowsandcolumnswithNullNaNvalues)
   * [Dropping duplicate rows](#Droppingduplicaterows)
   * [Dropping short data](#Droppingshortdata)

## [Part 3. EDA & Visualization](#eda)
   * [How many articles per category?](#articlesPerCategory1)
   * [Distribution of content lengths](#distribution1)
   * [Common words in dataset](#commonwords1)
       * [Barplot](#barplot1)
       * [WordCloud](#wordcloud1)
   * [Conclusion of EDA](#conclusioneda)
   
## [Part 4. Cleaning the data after EDA](#cleaningaftereda)
   * [Capitalization](#capitalization)
   * [Stop words](#stopwords)
   * [Noise Removel](#noiseremovel)
   * [Lemmatization](#lemmatization)
   * [Remove categories with some articles](#removecategories)
   * [Visualization of the 'cleaned' DataFrame](#visualization)
       * [How many articles per category?](#articlesPerCategory2)
       * [Distribution of content lengths](#distribution2)
       * [Common words in dataset](#commonwords2)
           * [Barplot](#barplot2)
           * [WordCloud](#wordcloud2)

## [Part 5. Bag of Words and Machine Learning](#advancheddataanalysis)
   * [Create and Fit Bag of Words Model](#bagofwords)
   * [Train Test and Split the Dataset](#trainandtest)
   * [Machine Learning Models:](#mlmodels)
       * [Multinomial Naive Bayes](#Mnaivebayes)
       * [Logistic Regression](#logisticregression)
       * [Gaussian Naive Bayes](#Gnaivebayes)
       * [SVC](#svc)
   * [Results](#results)
   * [Best model](#bestmodel)
   * [Worst model](#worstmodel)
   * [Confusion Matrix](#confusionmatrix)
   * [Insert your data and we will tell you it category!](#woweffect)

## [Part 6. Conclusion](#Conclusions)
## [Credits](#credits)

### <a class="anchor" id="import"></a> Import Necessary Libraries


```python
import pandas as pd
import glob
import os

import bs4
import time
from datetime import datetime
from bs4 import BeautifulSoup  
import pandas as pd
import scipy as sc
import numpy as np
import requests

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import unidecode 
import re
import time 
import stopwords 
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk import word_tokenize
from wordcloud import WordCloud
import gensim
from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS

import langid
from langdetect import DetectorFactory
from langdetect import detect
from langdetect import detect_langs

import numpy as np
from scipy.stats import randint
import seaborn as sns # used for plot interactive graph. 
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import plot_confusion_matrix

import warnings
warnings.filterwarnings('ignore')
```

# <a class="anchor" id="scrapingthedata"></a> <p style="color:blue"> Part 1. Scraping the data </p> 

### Introduction
>
<font size="3"> In this section we will get data from a list of sources from all over the world.
    We are going to scraping from 9 websites a LOT of data!  </font>
>
Our sources are:
>
>- ABC: www.abc.net.au
>
>- Articlebiz: www.articlebiz.com
>
>- Ezinearticles: www.ezinearticles.com
>
>- Hubpage: www.hubpages.com
>
>- Huffpost: www.huffpost.com
>
>- Indiatoday: www.indiatoday.in
>
>- Inshorts: www.inshorts.com
>
>- Medium: www.medium.com
>
>- Moneycontrol: www.moneycontrol.com
>
>
We will use Selenium to scrape a lot of data from the websites, and we will use a wide variety of articles.
>
Then, we will get the data from each website and insert the raw data from the websites into a formatted CSV file

<font size="3"> The dataframes include the above fields:

- Topic - Topic of the article
- Category - Category of the article (tech, business, sport, entertainment, politics, etc.)
- Content - Whole content of the article

### Html into beautiful soup object <a class="anchor" id="Htmlintobeautifulsoupobject"></a>


```python
def get_article_html(url):
    html = requests.get(url)
    soup = BeautifulSoup(html.content,"html.parser")
    time.sleep(10) 
    return soup
```

### Inserting the raw data from the websites into a formatted CSV file <a class="anchor" id="InsertingtherawdatafromthewebsitesintoaformattedCSVfile"></a>


```python
def insert_data_to_csv(topic, category, content):
    # Create dataframe
    data = pd.DataFrame({
    "topic": topic,
    "category": category,
    "content": content
    })
    data.to_csv(f'Scraping-{category}-{datetime.now().strftime("%d-%m-%Y")}.csv', mode='a')
```

### Scraping the data from each website <a class="anchor" id="Scrapingthedatafromeachwebsite"></a>

### 1. ABC

### The story behind the website
>
ABC - Australian Broadcasting Corporation is the Australian national broadcaster.
>
The ABC, has a lot of articles and breaking news from all over the world.
>
In this year, the ABC will celebrate 90 years of news.
>
The categories from the website are:
>
>- Politics
>
>- Analysis-And-Opinion
>
>- Arts-Culture
>
>- Environment
>
>- House-And-Home
>
>- Travel-And-Tourism-Lifestyle-And-Leisure
>
>- Shopping-Mall
>
>- Markets
>
>- Society
>
>- Business
>
>- World
>
>- Health
>
>- Sport
>
>- Science
>

### Example of article from ABC

![image.png](attachment:778fcadf-8c9f-4303-b95c-6fd9650179b4.png)


```python
def get_data_abc_web(soup):   
    all_articles_part1= soup.find_all('div',attrs={'class':'yqrQw'})
    all_articles_part2 = soup.find_all('div',attrs={'class':'_2I1aj'})
    
    # Declaration
    articles_topic = []
    articles_content = []
   
    for article in all_articles_part1:
        topic = article.find('a').get_text()
        articles_topic.append(topic)

        url_of_article = article.find('a')['href']
        soup = get_article_html("https://www.abc.net.au/" + url_of_article)

        content = soup.find_all("p")
        full_content = ""
        if content != None:
            for paragraph in content:
                full_content = full_content + " " + paragraph.get_text().strip()
        articles_content.append(full_content) 
        
    for article in all_articles_part2:
        if article.find('a') != None:
            topic = article.find('a').get_text()
        
            if article.find('a')['href'] != None:
                url_of_article = article.find('a')['href']
                soup = get_article_html("https://www.abc.net.au/" + url_of_article)

                content = soup.find_all("p")
                full_content = ""
                if content != None:
                    for paragraph in content:
                        full_content = full_content + " " + paragraph.get_text().strip()
                articles_topic.append(topic)
                articles_content.append(full_content) 
      
    return articles_topic, articles_content
```


```python
def abc_scraping_with_selenium_from_url(url):
        PATH = ".\webdriver.Chrome"
        driver = webdriver.Chrome(PATH)
        driver.get(url)
        driver.maximize_window()
        time.sleep(5)
        close_cookies= "//button[@class='_1KwgR _1uCkA _2vzFN _20SSK eCXal _2VqCY pigdr']"
        driver.find_element(by=By.XPATH, value=close_cookies).click()
        time.sleep(5)
        for i in range (1, 20):
            next_xpath = "//button[@class='_1KwgR _2-5J1 eCXal']"
            driver.find_element(by=By.XPATH, value=next_xpath).click() 
            time.sleep(3)
        html = driver.page_source
        driver.close()
        soup = BeautifulSoup(html,"html.parser")
        topic, content = get_data_abc_web(soup)
        insert_data_to_csv(topic, category, content)
    
def abc_scraping_with_selenium():
    categories = ["politics", "analysis-and-opinion", "arts-culture", "environment", "house-and-home",
              "travel-and-tourism-lifestyle-and-leisure", "shopping-mall", "markets", "society"
              "business", "world", "health", "sport", "science",]

    print("Start scraping from abc!")
    for category in categories:
        if category in ("environment", "house-and-home", "travel-and-tourism-lifestyle-and-leisure", "shopping-mall",  "markets", "society"):
            abc_scraping_with_selenium_from_url(f"https://www.abc.net.au/news/topic/{category}")
        else:
            abc_scraping_with_selenium_from_url(f"https://www.abc.net.au/news/{category}")

    print("Finish scraping from abc.net.au!")

```

### 2. Articlebiz

### The story behind the website
>
Articlebiz is a website which includes opinion articles from different and diverse categories
>
The categories from the website are:
>
>- Politics
>
>- Business
>
>- Pets
>
>- Environment
>
>- News-Society
>
>- Arts-Entertainment
>
>- Travel-Leisure
>
>- Social-Issues
>
>- Autos-Trucks
>
>- Health-Fitness
>
>- Shopping
>
>- Computers-Technology
>
>- Finance
>
>- Home
>
>- Sports-Recreations

### Example of article from Articlebiz

![image.png](attachment:7439931b-ec54-4850-a5cd-eac58b6bcef0.png)


```python
def get_data_articlebiz_web(soup):
    
    articles = soup.find_all('div',{"class": lambda L: L and L.startswith('mb-4 card')})
    
    # Declaration
    articles_topic = []
    articles_content = []

    for article in articles:
        # Get article's topic
        topic = article.find("a").get_text()
        articles_topic.append(topic)

        # Get article's content
       
        url_of_article = article.find('a')['href']

        soup = get_article_html(url_of_article)

        content = soup.find('div',attrs={'class':'clearfix'}).find_all("p")
        full_content = ""
        if content != None:
            for paragraph in content:
                full_content = full_content + " " + paragraph.get_text().strip()
        articles_content.append(full_content) 
        
    return articles_topic, articles_content
```


```python
def aritclebiz_scraping_with_selenium():
    categories = ["politics", "business", "pets", "news-society", "arts-entertainment", "travel-leisure", "social-issues", "autos-trucks", "health-fitness",
                  "shopping", "computers-technology", "finance", "home", "sports-recreations"]

    print("Start scraping from articlebiz!")

    for category in categories:
        PATH = ".\webdriver.Chrome"
        driver = webdriver.Chrome(PATH)
        driver.get(f"https://www.articlebiz.com/category/{category}")
        driver.maximize_window()
        time.sleep(3)
        html = driver.page_source
        soup = BeautifulSoup(html,"html.parser")
        topic, content = get_data_articlebiz_web(soup)
        insert_data_to_csv(topic, category, content)
        next_button = driver.find_element(by=By.LINK_TEXT, value=('Next')).click() 
        for i in range (1, 28):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(10)
            html = driver.page_source
            soup = BeautifulSoup(html,"html.parser")
            topic, content = get_data_articlebiz_web(soup)
            insert_data_to_csv(topic, category, content)
            next_button = driver.find_element(by=By.LINK_TEXT, value=('Next')).click() 
        driver.close()

    print("Finish scraping from articlebiz.com!")


```

### 3. Ezinearticles

### The story behind the website
>
EzineArticles is an online content publishing platform with user-friendly article submission & analytic tools.
>
Every user-generated submission is human curated for quality control.
>
The categories from the website are:
>
>- Business
>
>- Arts-And-Entertainment
>
>- Computers-And-Technology
>
>- Travel-And-Leisure
>
>- Finance
>
>- Health-And-Fitness
>
>- Investing
>
>- News-And-Society
>
>- Communications
>
>- Food-And-Drink
>
>- Home-and-Family
>
>- Pets
>

### Example of article from Ezinearticles 

![image.png](attachment:1c152b99-e70c-4f5a-8ab0-00fbac3d240c.png)


```python
def get_data_ezinearticles_web(soup):
    
    articles=soup.find('div',attrs={'class':'category-list'}).find_all('div',attrs={'class':'article'})
    
    # Declaration
    articles_topic = []
    articles_content = []

    for article in articles:
        # Get article's topic
        topic = article.find("a").get_text()
        articles_topic.append(topic)
        
        # Get article's content - blocking us with lots of requests so we take the article's summary
        articles_content.append(article.find('div',attrs={'class':'article-summary'}).get_text()) 
   
    return articles_topic, articles_content
```


```python
def ezinearticles_scraping_with_selenium():
    categories = ["Business","Arts-and-Entertainment", "Computers-and-Technology", "Travel-and-Leisure", "Finance",
                 "Health-and-Fitness", "Investing", "News-and-Society", "Communications", "Food-and-Drink",
                  "Home-and-Family", "Pets"]

    print("Start scraping from ezinearticles!")
    for category in categories:
        PATH = ".\webdriver.Chrome"
        driver = webdriver.Chrome(PATH)
        driver.get(f"https://www.ezinearticles.com/?cat={category}")
        driver.maximize_window()
        time.sleep(3)
        html = driver.page_source
        soup = BeautifulSoup(html,"html.parser")
        topic, content = get_data_ezinearticles_web(soup)
        insert_data_to_csv(topic, category, content)
        next_button = driver.find_element(by=By.LINK_TEXT, value=('Next »')).click() 
        for i in range (1, 29):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(10)
            html = driver.page_source
            soup = BeautifulSoup(html,"html.parser")
            topic, content = get_data_ezinearticles_web(soup)
            insert_data_to_csv(topic, category, content)
            next_button = driver.find_element(by=By.LINK_TEXT, value=('Next »')).click() 
    driver.close()

    print("Finish scraping from ezinearticles.com!")
```

### 4. Hubpages

### The story behind the website
>
HubPages is an American online publishing platform developed by Paul Edmondson and was launched in 2006. 
>
The categories from the website are:
>
>- Politics
>
>- Sports
>
>- Technology
>
>- Business
>
>- Art
>
>- Entertainment
>
>- Health
>
>- Travel
>
>- Religion-Philosophy
>
>- Money
>
>- Community
>
>- Home
>
>- Food
>
>- Style
>
>- Education
>
>- Literature
> 
>- Autos
>
>- Games-Hobbies
>
>- Relationships
>
>- Holidays
>

### Example of article from Hubpages 

![image.png](attachment:2a328b6f-2572-4ae8-9fb0-7c754a814bea.png)


```python
def get_data_hubpages_web(soup):
    all_articles = soup.find_all('section',attrs={'class':'m-card-group'})
    # Declaration
    articles_topic = []
    articles_content = []
        
    for group_of_article in all_articles:
        
        articles = group_of_article.find_all('div',attrs={'class':'l-grid--item'})
        for article in articles:
            
            if article.find('h2') != None:
                topic = article.find('h2').get_text()
                articles_topic.append(topic)
                
                url_of_article = article.find('a')['href']
                soup = get_article_html("https://discover.hubpages.com/" + url_of_article)

                content = soup.find('div',attrs={'class':'m-detail--body'}).find_all("p")
                full_content = ""
                if content != None:
                    for paragraph in content:
                        full_content = full_content + " " + paragraph.get_text().strip()
                articles_content.append(full_content) 
      
    print("Finish get data!")     
    return articles_topic, articles_content
```


```python
def hubpages_scraping_with_selenium():
    categories = ["politics", "sports", "technology", "business", "art", "entertainment", "health", "travel", "religion-philosophy",
                  "money", "community", "home", "food", "style", "education", "literature", "autos", "games-hobbies",
                  "relationships", "holidays"]

    print("Start scraping from hubpages!")

    for category in categories:
        PATH = ".\webdriver.Chrome"
        driver = webdriver.Chrome(PATH)
        driver.get(f"https://discover.hubpages.com/{category}")
        time.sleep(2)
        driver.maximize_window() 
        for i in range(1, 20):   
            relative_xpath= "//*[@id='main-content']/section[3]/phoenix-hub/div/phoenix-footer-loader/button"
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            element = WebDriverWait(driver,10).until(EC.presence_of_element_located((By.XPATH,relative_xpath)))
            driver.execute_script("arguments[0].click()",element)
            time.sleep(3)
            html = driver.page_source
        driver.close()       
        soup = BeautifulSoup(html,"html.parser")
        topic, content = get_data_hubpages_web(soup)
        insert_data_to_csv(topic, category, content)

    print("Finish scraping from hubpages.com!")
```

### 5. Huffpost

### The story behind the website
>
HuffPost is an American news aggregator and blog, with localized and international editions. 
>
The site offers news, satire, blogs, and original content
>
The categories from the website are:
>
>- Health
>
>- Opinion
>
>- Business
>
>- Enviorment
>
>- Politics
>
>- Social-Justice
>
>- Coronavirus
>
>- Entertainment
>

### Example of article from Huffpost

![image.png](attachment:2a1ff35f-52e4-4e6e-8bc7-dfa4920e815a.png)


```python
def get_data_huffpost_web(soup):   
    all_articles = soup.find_all('div',attrs={'class':'zone__content'})
    
    # Declaration
    articles_topic = []
    articles_content = []

    for group_of_article in all_articles:
        
        articles = group_of_article.find_all('div',attrs={'class':'card__headlines'})
        for article in articles:
            topic = article.find('h3').get_text()
            articles_topic.append(topic)
                
            url_of_article = article.find('a')['href']
            soup = get_article_html(url_of_article)
            
            content = soup.find_all("p")
            full_content = ""
            if content != None:
                for paragraph in content:
                    full_content = full_content + " " + paragraph.get_text().strip()
            articles_content.append(full_content) 
      
    return articles_topic, articles_content
```


```python
def huffpost_scraping_with_selenium_from_url(url):
    PATH = ".\webdriver.Chrome"
    driver = webdriver.Chrome(PATH)
    driver.get(url)
    driver.maximize_window()
    time.sleep(5)
    html = driver.page_source
    soup = BeautifulSoup(html,"html.parser")
    topic, content = get_data_huffpost_web(soup)
    insert_data_to_csv(topic, category, content)
    next_button = driver.find_element(by=By.LINK_TEXT, value=('Next')).click()
    for i in range (1, 30):
        time.sleep(5)
        html = driver.page_source
        soup = BeautifulSoup(html,"html.parser")
        topic, content = get_data_huffpost_web(soup)
        insert_data_to_csv(topic, category, content)
        next_button = driver.find_element(by=By.LINK_TEXT, value=('Next')).click()
    driver.close()
    
def huffpost_scraping_with_selenium():
    categories = ["health", "opinion", "business", "environment", "politics", "social-justice", "coronavirus", "entertainment"]
    
    print("Start scraping from huffpost!")    
    for category in categories:
        if category in ("health", "opinion"):
            huffpost_scraping_with_selenium_from_url(f"https://www.huffpost.com/section/{category}")
        if category == "business":
            huffpost_scraping_with_selenium_from_url(f"https://www.huffpost.com/impact/{category}")
        if category == "environment":
            huffpost_scraping_with_selenium_from_url(f"https://www.huffpost.com/impact/green")
        if category == "politics":
            huffpost_scraping_with_selenium_from_url(f"https://www.huffpost.com/news/{category}")
        if category == "social-justice":
            huffpost_scraping_with_selenium_from_url(f"https://www.huffpost.com/impact/topic/{category}")
        if category == "coronavirus":
            huffpost_scraping_with_selenium_from_url(f"https://www.huffpost.com/news/topic/{category}")
        if category == "entertainment":
            huffpost_scraping_with_selenium_from_url(f"https://www.huffpost.com/{category}")     

    print("Finish scraping from huffpost.com!")
```

### 6. Indiatoday

### The story behind the website
>
India Today is a weekly Indian English-language news magazine.
>
It is the most widely circulated magazine in India, with a readership of close to 8 million.
>
The categories from the website are:
>
>- Business
>
>- World
>
>- Technology
>
>- Science
>
>- Health
>

### Example of article from Indiatoday

![image.png](attachment:9a79930e-60f4-4263-81de-7e16c5f139b6.png)


```python
def get_data_indiatoday_web(soup):
    
    articles = soup.find_all("div",attrs={"class":"catagory-listing"})
    
    # Declaration
    articles_topic = []
    articles_content = []
    for article in articles:
        # Get article's topic
        if article.find("p") != None and article.find("h2") != None:
            topic = article.find("h2").get_text()
            articles_topic.append(topic)

            # Get article's content - there are lots of videos description so we take the article's summary
            articles_content.append(article.find("p").get_text()) 
        
    return articles_topic, articles_content
```


```python
def indiatoday_scraping_with_selenium_from_url(url):
    PATH = ".\webdriver.Chrome"
    driver = webdriver.Chrome(PATH)
    driver.get(url)
    driver.maximize_window()
    time.sleep(10)
    html = driver.page_source
    soup = BeautifulSoup(html,"html.parser")
    topic, content = get_data_indiatoday_web(soup)
    insert_data_to_csv(topic, category, content)
    next_button = driver.find_element(by=By.LINK_TEXT, value=('Next')).click()
    for i in range (1, 15):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(10)
        html = driver.page_source
        soup = BeautifulSoup(html,"html.parser")
        topic, content = get_data_indiatoday_web(soup)
        insert_data_to_csv(topic, category, content)
        next_button = driver.find_element(by=By.LINK_TEXT, value=('Next')).click()
    driver.close()

def indiatoday_scraping_with_selenium():     
    categories = ["business", "world", "technology", "science", "health"]
    print("Start scraping from indiatoday!")

    for category in categories:
        if category == "technology":
            indiatoday_scraping_with_selenium_from_url(f"https://www.indiatoday.in/{category}/news")
        else:
            indiatoday_scraping_with_selenium_from_url(f"https://www.indiatoday.in/{category}")

    print("Finish scraping from indiatoday.in!")
     
```

### 7. Inshorts

### The story behind the website
>
Inshorts is an aggregator app that summarizes news articles in 60 words and covers a wide-range of topics, including tech and business.
>
The categories from the website are:
>
>- Sports
>
>- World
>
>- Politics
>
>- Technology
>
>- Startup
>
>- Entertainment
>
>- Science
>

### Example of article from Ishorts

![image.png](attachment:5cc9ae93-77ba-426c-8663-6826e748d9a2.png)


```python
def get_data_inshorts_web(soup):
    
    articles = soup.find_all("div",attrs={"class":"news-card z-depth-1"})
    
    # Declaration
    articles_topic = []
    articles_content = []

    for article in articles:
        # Get article's topic
        topic = article.find("span",attrs={"itemprop":"headline"}).get_text()
        articles_topic.append(topic)
        
        # Get article's content
        content = article.find('div',attrs={'itemprop':'articleBody'}).get_text()
        articles_content.append(content) 
        
    return articles_topic, articles_content
```


```python
def inshorts_scraping_with_selenium():
    categories = ["sports", "world", "politics", "technology", "startup", "entertainment", "science"]
    print("Start scraping from inshorts!")
    for category in categories:
        PATH = ".\webdriver.Chrome"
        driver = webdriver.Chrome(PATH)
        driver.get(f"https://www.inshorts.com/en/read/{category}")
        driver.maximize_window()  
        for i in range (1, 20):
            time.sleep(5)
            button_xpath = "//div[@class='clickable unselectable load-more z-depth-1 hoverable']"  
            driver.find_element(by=By.XPATH, value=button_xpath).click()                    
            time.sleep(5)
        html = driver.page_source
        driver.close()
        soup = BeautifulSoup(html,"html.parser")
        topic, content = get_data_inshorts_web(soup)
        insert_data_to_csv(topic, category, content)
    print("Finish scraping from inshorts.com!")
```

### 8. Medium

### The story behind the website
>
Medium is an open platform where over 100 million readers come to find insightful and dynamic thinking.
>
There, expert and undiscovered voices alike dive into the heart of any topic and bring new ideas to the surface
>
Medium is where those ideas take shape, take off, and spark powerful conversations.
>
The categories from the website are:
>
>- Technology
>
>- Health
>
>- Business
>
>- Money
>
>- Entrepreneurship
>
>- Marketing
>
>- Life
>

### Example of article from Medium

![image.png](attachment:16538efc-22b4-427f-9b4a-68a5396e40a9.png)


```python
def get_data_medium_web(soup):
    articles=soup.find_all("article")
    
    # Declaration
    articles_topic = []
    articles_content = []
    
    for article in articles:
        # Get article's topic
        topic = article.find("h2").get_text()
        articles_topic.append(topic)

        # Get article's content
        content = article.find('a')['href']
        url_of_article = "https://medium.com/" + article.find('a')['href']
        soup = get_medium_html(url_of_article)
        content = soup.find("main").find_all("p")
        
        full_content = ""
        if content != None:
            for paragraph in content:
                full_content = full_content + " " + paragraph.get_text().strip()
        articles_content.append(full_content)
      
    return articles_topic, articles_content
```


```python
def medium_scraping_with_selenium():
    categories = ["technology", "health", "business", "money", "entrepreneurship", "marketing", "life"]
    print("Start scraping from Medium.com!")
    PATH = ".\webdriver.Chrome"
    driver = webdriver.Chrome(PATH)    
    driver.get(f"https://medium.com/tag/{category}")
    driver.maximize_window()
    for i in range (1, 15):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)
    html = driver.page_source
    driver.close()
    soup_obj = get_soup_objects(html)
    topics, contents = get_data_medium_web(soup_obj)
    insert_data_to_csv(topics, contents)
    print("Finish scraping from Medium.com!")
```

### 9. Moneycontrol

### The story behind the website
>
moneycontrol is India's No 1 Financial and Business portal. 
>
With in-depth market coverage, analysis, expert opinions and a gamut of financial tools, moneycontrol.com has been the premier destination for consumers and market watchers.
>
The categories from the website are:
>
>- Markets
>
>- Stocks
>
>- Companies
>
>- Trends
>
>- Business
>
>- Economy
>

### Example of article from Moneycontrol

![image.png](attachment:47ca37e2-c3b1-4179-bb99-b331e63b6749.png)


```python
def get_data_moneycontrol_web(soup):
    all_articles = soup.find('ul',attrs={'id':'cagetory'}).find_all('li',attrs={'class':'clearfix'})
    
    # Declaration
    articles_topic = []
    articles_content = []
    
    for article in all_articles:
            topic = article.find('h2').find('a').get_text()
            
            url_of_article = article.find('h2').find('a')['href']
            soup = get_article_html(url_of_article)
            
            if soup.find('div',attrs={'class':'content_wrapper arti-flow'}) != None:
                content = soup.find('div',attrs={'class':'content_wrapper arti-flow'}).find_all("p")
                full_content = ""
                if content != None:
                    for paragraph in content:
                        full_content = full_content + " " + paragraph.get_text().strip()
                        
                articles_topic.append(topic)
                articles_content.append(full_content)            

    return articles_topic, articles_content
```


```python
def money_control_scraping_with_selenium_from_url(url):
    PATH = ".\webdriver.Chrome"
    driver = webdriver.Chrome(PATH)   
    driver.get(url)
    time.sleep(30)
    driver.maximize_window()
    time.sleep(30)
    html = driver.page_source
    soup = BeautifulSoup(html,"html.parser")
    topic, content = get_data_moneycontrol_web(soup)
    insert_data_to_csv(topic, category, content)
    next_button = driver.find_element(by=By.LINK_TEXT, value=('»')).click()
    for i in range (1, 28): 
        time.sleep(30)
        html = driver.page_source
        soup = BeautifulSoup(html,"html.parser")
        topic, content = get_data_moneycontrol_web(soup)
        insert_data_to_csv(topic, category, content)
        next_button = driver.find_element(by=By.LINK_TEXT, value=('»')).click()
    driver.close()
        
def money_control_scraping_with_selenium():
    categories = ["markets", "stocks", "companies", "trends", "business", "economy"]

    print("Start scraping from moneycontrol!")

    for category in categories:
        if category in ("economy", "markets", "stocks", "companies"):
            money_control_scraping_with_selenium_from_url(f"https://www.moneycontrol.com/news/business/{category}")
        else:
            money_control_scraping_with_selenium_from_url(f"https://www.moneycontrol.com/news/{category}")

    print("Finish scraping from moneycontrol.com!")
```

### Now we are ready to scrape articles from all these websites!


```python
print("start scraping from all the websites!")
# abc_scraping_with_selenium()
# aritcle_biz_scraping_with_selenium()
# exine_article_scraping_with_selenium()
# hubpages_scraping_with_selenium()
# huffpost_scraping_with_selenium()
# indiatoday_scraping_with_selenium()
# inshorts_scraping_with_selenium()
# medium_scraping_with_selenium()
# money_control_scraping_with_selenium()
print("Finish Scraping")
```

    start scraping from all the websites!
    Finish Scraping
    

### Merge all csv files into single DataFrame


```python
def get_data_from_csv(url):
    df = pd.read_csv(url)
    return df
```


```python
def insert_data_to_csv(topic, category, content):
    # Create dataframe
    data = pd.DataFrame({
    "topic": topic,
    "category": category,
    "content": content
    })
    data.to_csv(f'merge.csv', mode='a')
```


```python
def merge_csv_files():
    # setting the path for joining multiple files, for example: "example*"
    files = os.path.join("C:\\Users\\Ori\\Desktop\\data", "scraping*.csv")
    
    # list of merged files returned
    files = glob.glob(files)

    # joining files with concat and read_csv
    df = pd.concat(map(pd.read_csv, files), ignore_index=True)
    insert_data_to_csv(df["topic"], df["category"], df["content"])
```


```python
print("Start merging csv")
merge_csv_files()
print("Merged to 'merge.csv'!")
```

    Start merging csv
    Merged to 'merge.csv'!
    

### The DataFrame is ready to use!


```python
df = get_data_from_csv("merge.csv")
print("Shape of the dataframe (rows, columns): " , df.shape)
df
```

    Shape of the dataframe (rows, columns):  (42141, 4)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>topic</th>
      <th>category</th>
      <th>content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>analysis: America is standing on the precipice...</td>
      <td>analysis-and-opinion</td>
      <td>The US Supreme Court's monumental decisions ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>analysis: America is standing on the precipice...</td>
      <td>analysis-and-opinion</td>
      <td>The US Supreme Court's monumental decisions ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>analysis: Why it's time for Samoa, rugby leagu...</td>
      <td>analysis-and-opinion</td>
      <td>Sport As Samoa looks to begin again, it's tim...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>analysis: How going without a ruck is helping ...</td>
      <td>analysis-and-opinion</td>
      <td>Sport Inside the game: How Port Adelaide's co...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>analysis: The science of silliness: How psycho...</td>
      <td>analysis-and-opinion</td>
      <td>How Socceroos goalkeeper Andrew Redmayne use...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>42136</th>
      <td>42136</td>
      <td>10 Steps to Building a Data Collection System ...</td>
      <td>technology</td>
      <td>268 Followers Published in DataDrivenInvestor...</td>
    </tr>
    <tr>
      <th>42137</th>
      <td>42137</td>
      <td>When Should I Talk to My Child About Periods?</td>
      <td>technology</td>
      <td>1.99K Followers 23 hours ago Follow up questi...</td>
    </tr>
    <tr>
      <th>42138</th>
      <td>42138</td>
      <td>If I Did Things Differently</td>
      <td>technology</td>
      <td>23 Followers 22 hours ago Gaming has been wit...</td>
    </tr>
    <tr>
      <th>42139</th>
      <td>42139</td>
      <td>Get Paid Directly by Fans</td>
      <td>technology</td>
      <td>1.99K Followers Pinned Do at least 7 immediat...</td>
    </tr>
    <tr>
      <th>42140</th>
      <td>42140</td>
      <td>On Clarity vs Certainty: Strategy for Navigati...</td>
      <td>technology</td>
      <td>64 Followers Pinned On Clarity vs Certainty: ...</td>
    </tr>
  </tbody>
</table>
<p>42141 rows × 4 columns</p>
</div>



# <a class="anchor" id="Part2.Initialcleaningdata"></a> <p style="color:blue"> Part 2. Initial cleaning data </p>

### 1. Removing rows and columns with Null/NaN values. <a class="anchor" id="RemovingrowsandcolumnswithNullNaNvalues"></a>


```python
print("Before removing null objects: " , df.shape)
df.dropna(inplace=True)
print("After removing null objects: " ,  df.shape)
```

    Before removing null objects:  (42141, 4)
    After removing null objects:  (41632, 4)
    

### 2. Dropping duplicate rows <a class="anchor" id="Droppingduplicaterows"></a>


```python
print("Before dropping duplicates" ,  df.shape)
df.drop_duplicates('content', inplace = True)
print("After dropping duplicates" ,  df.shape)
```

    Before dropping duplicates (41632, 4)
    After dropping duplicates (37951, 4)
    

### 3. Dropping short data <a class="anchor" id="Droppingshortdata"></a>


```python
def drop_rows_with_short_content(df):

    # split values by whitespace and drop data lt 3 words 
    df = df[df["content"].str.split().str.len() > 3]
    
    return df
```


```python
df = drop_rows_with_short_content(df)
```

### Shape of the DataFrame after initial cleaning


```python
df.shape
```




    (37947, 4)



# <a class="anchor" id="eda"></a> <p style="color:blue"> Part 3. EDA & Visualiztion </p>

### Exploratory Data Analysis

Let's check the distribution of the different categories across the dataset.

How many categories and how many articles in each category are there in the dataset?

### <a class="anchor" id="articlesPerCategory1"></a> How many articles per category?


```python
print('Number of Categories: ',df.groupby('category').ngroups)
print(df['category'].value_counts())
```

    Number of Categories:  56
    politics                                    1978
    business                                    1933
    technology                                  1707
    sports                                      1554
    entertainment                               1534
    health                                      1525
    money                                       1113
    style                                       1000
    education                                   1000
    literature                                   999
    holidays                                     999
    travel                                       999
    games-hobbies                                999
    food                                         933
    relationships                                898
    science                                      860
    world                                        711
    autos                                        707
    Communications                               600
    News-and-Society                             600
    Investing                                    600
    Home-and-Family                              600
    Health-and-Fitness                           600
    Food-and-Drink                               600
    Finance                                      600
    religion-philosophy                          600
    Business                                     600
    pets                                         600
    news-society                                 600
    Computers-and-Technology                     599
    Travel-and-Leisure                           599
    Arts-and-Entertainment                       599
    Pets                                         597
    social-issues                                560
    autos-trucks                                 560
    arts-entertainment                           560
    finance                                      560
    travel-leisure                               559
    shopping                                     558
    health-fitness                               557
    computers-technology                         556
    startup                                      514
    house-and-home                               458
    environment                                  441
    markets                                      417
    travel-and-tourism-lifestyle-and-leisure     387
    arts-culture                                 197
    sport                                        147
    marketing                                    117
    entrepreneurship                             113
    society                                      100
    community                                     53
    art                                           50
    analysis-and-opinion                          21
    shopping-mall                                 12
    life                                           7
    Name: category, dtype: int64
    


```python
df.category.value_counts().plot(kind='bar', color='#348ABD',figsize=(20,15))
```




    <AxesSubplot:>




    
![png](output_80_1.png)
    



```python
import plotly.express as px

x = df['category'].value_counts().index.values.astype('str')
y = df['category'].value_counts().values
pct = [("%.2f"%(v*200))+"%"for v in (y/len(df))]

print("Number of articles in each category: ")

trace1 = px.bar(x=x, y=y, text=pct)
layout = dict(title= 'Number of articles in each category',
              yaxis = dict(title='Count'),
              xaxis = dict(title='category'))


fig=dict(data=[trace1], layout=layout)
trace1.show()
```

    Number of articles in each category: 
    


<div>                            <div id="b087c374-6b2e-44b9-aa46-996078b33f3c" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("b087c374-6b2e-44b9-aa46-996078b33f3c")) {                    Plotly.newPlot(                        "b087c374-6b2e-44b9-aa46-996078b33f3c",                        [{"alignmentgroup":"True","hovertemplate":"x=%{x}<br>y=%{y}<br>text=%{text}<extra></extra>","legendgroup":"","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"","offsetgroup":"","orientation":"v","showlegend":false,"text":["10.43%","10.19%","9.00%","8.19%","8.08%","8.04%","5.87%","5.27%","5.27%","5.27%","5.27%","5.27%","5.27%","4.92%","4.73%","4.53%","3.75%","3.73%","3.16%","3.16%","3.16%","3.16%","3.16%","3.16%","3.16%","3.16%","3.16%","3.16%","3.16%","3.16%","3.16%","3.16%","3.15%","2.95%","2.95%","2.95%","2.95%","2.95%","2.94%","2.94%","2.93%","2.71%","2.41%","2.32%","2.20%","2.04%","1.04%","0.77%","0.62%","0.60%","0.53%","0.28%","0.26%","0.11%","0.06%","0.04%"],"textposition":"auto","x":["politics","business","technology","sports","entertainment","health","money","style","education","literature","holidays","travel","games-hobbies","food","relationships","science","world","autos","Communications","News-and-Society","Investing","Home-and-Family","Health-and-Fitness","Food-and-Drink","Finance","religion-philosophy","Business","pets","news-society","Computers-and-Technology","Travel-and-Leisure","Arts-and-Entertainment","Pets","social-issues","autos-trucks","arts-entertainment","finance","travel-leisure","shopping","health-fitness","computers-technology","startup","house-and-home","environment","markets","travel-and-tourism-lifestyle-and-leisure","arts-culture","sport","marketing","entrepreneurship","society","community","art","analysis-and-opinion","shopping-mall","life"],"xaxis":"x","y":[1978,1933,1707,1554,1534,1525,1113,1000,1000,999,999,999,999,933,898,860,711,707,600,600,600,600,600,600,600,600,600,600,600,599,599,599,597,560,560,560,560,559,558,557,556,514,458,441,417,387,197,147,117,113,100,53,50,21,12,7],"yaxis":"y","type":"bar"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"x"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"y"}},"legend":{"tracegroupgap":0},"margin":{"t":60},"barmode":"relative"},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('b087c374-6b2e-44b9-aa46-996078b33f3c');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


### <a class="anchor" id="distribution1"></a> The distribution of the content' lengths


```python
lists = []
min_len = 100000
max_len = 0

for content in df['content']:
    words_count_in_article = len(str(content).split())
    if words_count_in_article < min_len:
        min_len = words_count_in_article
    if words_count_in_article > max_len:
        max_len = words_count_in_article
    lists.append(words_count_in_article)

print("Minimum content length in dataframe: ", min_len)
print("Maximum content length in dataframe: ", max_len)

plt.plot(lists)
plt.xlabel('Index of article')
plt.ylabel('Count of words')
plt.show()

mean = np.mean(lists)
print("The mean is: %.2f" % mean)

std = np.std(lists)
print("The std is: %.2f" % std)
```

    Minimum content length in dataframe:  4
    Maximum content length in dataframe:  91209
    


    
![png](output_83_1.png)
    


    The mean is: 711.08
    The std is: 1124.51
    

### <a class="anchor" id="commonwords1"></a> Common words in dataset

#### What are the most common terms in our dataset? Let's find out!

#### <a class="anchor" id="barplot1"></a> Barplot


```python
from nltk import FreqDist

def freq_words(x, terms = 20):
    all_words = ' '.join([content for content in x])
    all_words = all_words.split()

    freq_dist = FreqDist(all_words)
    words_df = pd.DataFrame({'Word':list(freq_dist.keys()), 'Count':list(freq_dist.values())})

    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="Count", n = terms) 
    plt.figure(figsize=(20,7))
    ax = sns.barplot(data=d, x= "Word", y = "Count")
    plt.show()
```


```python
freq_words(df.content)
```


    
![png](output_87_0.png)
    


Most common terms in topic and content for 3 categories:


```python
i = 1
for category in set(df["category"]):
    if i <= 3:
        try:
            print(category + ":")
            print("most frequent terms in content: ")
            freq_words(df[df["category"] == category].content)

            print("most frequent terms in topic: ")
            freq_words(df[df["category"] == category].topic)
            print("--------------------------------------------------------------------------------------------------------------------")
        except:
            print(category)
    i += 1
```

    autos-trucks:
    most frequent terms in content: 
    


    
![png](output_89_1.png)
    


    most frequent terms in topic: 
    


    
![png](output_89_3.png)
    


    --------------------------------------------------------------------------------------------------------------------
    Travel-and-Leisure:
    most frequent terms in content: 
    


    
![png](output_89_5.png)
    


    most frequent terms in topic: 
    


    
![png](output_89_7.png)
    


    --------------------------------------------------------------------------------------------------------------------
    Home-and-Family:
    most frequent terms in content: 
    


    
![png](output_89_9.png)
    


    most frequent terms in topic: 
    


    
![png](output_89_11.png)
    


    --------------------------------------------------------------------------------------------------------------------
    

### <a class="anchor" id="wordcloud1"></a> WordCloud

Visual representation of text data in the form of tags, which are typically single words whose importance is visualized by way of their size and color.

#### What are the most common terms in our dataset? Let's find out!


```python
def wordCloud_content(terms = 30):
    all_words = ' '.join([text for text in df['content']])
    all_words = all_words.split()
    
    freq_dist = nltk.FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(freq_dist.keys()), 'count':list(freq_dist.values())})
    
    fig = plt.figure(figsize=(21,16))
    ax1 = fig.add_subplot(2,1,1)
    wordcloud = WordCloud(width=1000, height=300, background_color='black', 
                          max_words=1628, relative_scaling=1,
                          normalize_plurals=False).generate_from_frequencies(freq_dist)
    
    ax1.imshow(wordcloud, interpolation="bilinear")
    ax1.axis('off')
    
    # select top 20 most frequent word
    ax2 = fig.add_subplot(2,1,2)
    d = words_df.nlargest(columns="count", n = terms) 
    ax2 = sns.barplot(data=d, palette = sns.color_palette('BuGn_r'), x= "count", y = "word")
    ax2.set(ylabel= 'Word')
    plt.show()
    
def wordCloud_category_content_topic(category, x, terms = 30):
    if x == 1: # content
        all_words = ' '.join([text for text in df[df['category'] == category].content])
    else: # topic
        all_words = ' '.join([text for text in df[df['category'] == category].topic])
    all_words = all_words.split()
    
    freq_dist = nltk.FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(freq_dist.keys()), 'count':list(freq_dist.values())})
    
    fig = plt.figure(figsize=(21,16))
    ax1 = fig.add_subplot(2,1,1)
    wordcloud = WordCloud(width=1000, height=300, background_color='black', 
                          max_words=1628, relative_scaling=1,
                          normalize_plurals=False).generate_from_frequencies(freq_dist)
    
    ax1.imshow(wordcloud, interpolation="bilinear")
    ax1.axis('off')
    
    # select top 30 most frequent word
    ax2 = fig.add_subplot(2,1,2)
    d = words_df.nlargest(columns="count", n = terms) 
    ax2 = sns.barplot(data=d, palette = sns.color_palette('BuGn_r'), x= "count", y = "word")
    ax2.set(ylabel= 'Word')
    plt.show()
```


```python
wordCloud_content()
```


    
![png](output_92_0.png)
    


Most common terms in topic and content for 3 categories:


```python
i = 1
for category in set(df["category"]):
    if i <=3:
        try:
            print(category + " WordCloud:")
            print("Content:")
            wordCloud_category_content_topic(category, 1)

            print("Topic:")
            wordCloud_category_content_topic(category, 0)
            print("--------------------------------------------------------------------------------------------------")
        except:
            print(category)
    i+=1
```

    autos-trucks WordCloud:
    Content:
    


    
![png](output_94_1.png)
    


    Topic:
    


    
![png](output_94_3.png)
    


    --------------------------------------------------------------------------------------------------
    Travel-and-Leisure WordCloud:
    Content:
    


    
![png](output_94_5.png)
    


    Topic:
    


    
![png](output_94_7.png)
    


    --------------------------------------------------------------------------------------------------
    Home-and-Family WordCloud:
    Content:
    


    
![png](output_94_9.png)
    


    Topic:
    


    
![png](output_94_11.png)
    


    --------------------------------------------------------------------------------------------------
    

## <a class="anchor" id="conclusioneda"></a> Conclusion of EDA

1. <font size="3"> The most common terms in our dataset are words that calles StopWords. If we remove these words, we can focus on the important words instead.
2. <font size="3"> There are categories with some articles. In order to train our model to get the best results, we need to have a sufficient amount of articles per category.
    Therfore, after removing articles with not enough words, some of categories had less than 1,500 words. We have decided to eliminate those categories to get the best results.

# <a class="anchor" id="cleaningaftereda"></a> <p style="color:blue">Part 4. Cleaning the data after EDA </p>

- Lowercase the data
- Remove links
- Remove punctuation, stop words
- Lemmatization the words
- Union categories with same meaning
- Remove categories with some articles - less than 1500

## <a class="anchor" id="capitalization"></a> Capitalization
Sentences can contain a mixture of uppercase and lower case letters. Multiple sentences make up a text document. To reduce the problem space, the most common approach is to reduce everything to lower case.

## <a class="anchor" id="stopwords"></a> Stop words
Stopwords are those words that do not provide any useful information to decide in which category a text should be classified. This may be either because they don't have any meaning (prepositions, conjunctions, etc.) or because they are too frequent in the classification context.

## <a class="anchor" id="noiseremovel"></a> Noise Removal
Text documents generally contains characters like punctuations or special characters and they are not necessary for text mining or classification purposes. Although punctuation is critical to understand the meaning of the sentence, but it can affect the classification algorithms negatively.

## <a class="anchor" id="lemmatization"></a> Lemmatization
Process of grouping together the different inflected forms of a word so they can be analyzed as a single item.


```python
def cleaning_preprocessing_data_from_csv(data):  
    cleaned_data = []
    
    for text in data:

        # Replacing all the occurrences of \n,\\n,\t,\\ with a space.
        data_to_clean = text.replace('\\n', ' ').replace('\n', ' ').replace('\t',' ').replace('\\', ' ').replace('. com', '.com')

        # Removing all the occurrences of links that starts with https
        data_to_clean = re.sub(r'http\S+', '', data_to_clean)

        # Remove all the occurrences of text that ends with .com
        data_to_clean = re.sub(r"\ [A-Za-z]*\.com", " ", data_to_clean)

        # Remove all whitespaces
        pattern = re.compile(r'\s+') 
        data_to_clean = re.sub(pattern, ' ', data_to_clean)
        data_to_clean = data_to_clean.replace('?', ' ? ').replace(')', ') ')

        # Remove accented characters from text using unidecode.
        # Unidecode() - It takes unicode data & tries to represent it to ASCII characters. 
        remove_character = unidecode.unidecode(data_to_clean)

        # Convert text to lower case
        data_to_clean = remove_character.lower()

        # Pattern matching for all case alphabets
        Pattern_alpha = re.compile(r"([A-Za-z])\1{1,}", re.DOTALL)
        # Limiting all the  repeatation to two characters.
        data_to_clean = Pattern_alpha.sub(r"\1\1", data_to_clean) 

        # Pattern matching for all the punctuations that can occur
        Pattern_Punct = re.compile(r'(\'[.,/#!"$<>@[]^&%^&*?;:{}=_`~()+-])\1{1,}')
        # Limiting punctuations in previously formatted string to only one.
        data_to_clean = Pattern_Punct.sub(r'\1', data_to_clean)

        # The below statement is replacing repeatation of spaces that occur more than two times with that of one occurrence.
        data_to_clean = re.sub(' {2,}',' ', data_to_clean)

        # The formatted text after removing not necessary punctuations.
        data_to_clean = re.sub(r"[^a-zA-Z]+", ' ', data_to_clean) 
        
        # Text without stopwords
        remove_stop_words = repr(data_to_clean)
        stoplist = stopwords.words('english') 

        # Append words to Medium.com
        stoplist.extend(['ago', 'followers', 'pinned', 'read', 'min', 'published', 'days', 'hours', 'the', 'was', 'has' , 'us'])
            
        No_StopWords = [word for word in word_tokenize(remove_stop_words) if word.lower() not in stoplist ]

        # Convert list of tokens_without_stopwords to String type.
        words_string = No_StopWords[0]
        words_string = ' '.join(No_StopWords[1:]) 
        
        # Remove more stop words  
        data_to_clean = remove_stopwords(words_string) 
        
         # Lemmatization is the process of grouping together the different inflected forms of a word so they can be analyzed as a single item
        wordnet = WordNetLemmatizer()
        data_to_clean =  " ".join([wordnet.lemmatize(word) for word in word_tokenize(data_to_clean)])

        # Split the "'" from the edges
        cleaned_data.append(data_to_clean[:len(data_to_clean)-1])
        
    return cleaned_data
```


```python
def lower_category_and_union_categories_with_same_meaning():
    category_list = []
    for category in df["category"]:
        lower_category = category.lower()
        if lower_category == "sports":
            lower_category = "sport"
        if lower_category == "health-and-fitness":
            lower_category = "health-fitness"
        if lower_category == "arts-and-entertainment":
            lower_category = "arts-entertainment"
        if lower_category == "computers-and-technology" or lower_category == "computers-technology":
            lower_category = "computers"
        if lower_category == "travel-and-leisure" or lower_category == "travel-leisure" or lower_category == "travel-and-tourism-lifestyle-and-leisure":
            lower_category = "travel"
        if lower_category == "news-and-society" or lower_category == "news-society":
            lower_category = "society"
        if lower_category == "food-and-drink":
            lower_category = "food"
        category_list.append(lower_category)

    return category_list
```


```python
print("Before cleaning data the shape of data frame is : " + str(df.shape))
df
```

    Before cleaning data the shape of data frame is : (37947, 4)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>topic</th>
      <th>category</th>
      <th>content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>analysis: America is standing on the precipice...</td>
      <td>analysis-and-opinion</td>
      <td>The US Supreme Court's monumental decisions ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>analysis: Why it's time for Samoa, rugby leagu...</td>
      <td>analysis-and-opinion</td>
      <td>Sport As Samoa looks to begin again, it's tim...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>analysis: How going without a ruck is helping ...</td>
      <td>analysis-and-opinion</td>
      <td>Sport Inside the game: How Port Adelaide's co...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>analysis: The science of silliness: How psycho...</td>
      <td>analysis-and-opinion</td>
      <td>How Socceroos goalkeeper Andrew Redmayne use...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>analysis: The RBA boss has a message, but you ...</td>
      <td>analysis-and-opinion</td>
      <td>The RBA boss wants you to follow his path aw...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>42136</th>
      <td>42136</td>
      <td>10 Steps to Building a Data Collection System ...</td>
      <td>technology</td>
      <td>268 Followers Published in DataDrivenInvestor...</td>
    </tr>
    <tr>
      <th>42137</th>
      <td>42137</td>
      <td>When Should I Talk to My Child About Periods?</td>
      <td>technology</td>
      <td>1.99K Followers 23 hours ago Follow up questi...</td>
    </tr>
    <tr>
      <th>42138</th>
      <td>42138</td>
      <td>If I Did Things Differently</td>
      <td>technology</td>
      <td>23 Followers 22 hours ago Gaming has been wit...</td>
    </tr>
    <tr>
      <th>42139</th>
      <td>42139</td>
      <td>Get Paid Directly by Fans</td>
      <td>technology</td>
      <td>1.99K Followers Pinned Do at least 7 immediat...</td>
    </tr>
    <tr>
      <th>42140</th>
      <td>42140</td>
      <td>On Clarity vs Certainty: Strategy for Navigati...</td>
      <td>technology</td>
      <td>64 Followers Pinned On Clarity vs Certainty: ...</td>
    </tr>
  </tbody>
</table>
<p>37947 rows × 4 columns</p>
</div>




```python
title_list = cleaning_preprocessing_data_from_csv(df["topic"])
print("Finish cleaning on topic")

content_list = cleaning_preprocessing_data_from_csv(df["content"])
print("Finish cleaning on content")

category_list = lower_category_and_union_categories_with_same_meaning()
```

    Finish cleaning on topic
    Finish cleaning on content
    

### Keep a version of the 'cleaned' DataFrame


```python
def insert_df_to_csv(topic, category, content, csv_file_name):
    cleaned_df = pd.DataFrame({
    "topic": topic,
    "category": category,
    "content": content
    })
    cleaned_df.to_csv(csv_file_name)
    return cleaned_df
```


```python
cleaned_df = insert_df_to_csv(title_list, category_list, content_list, "cleaned-scraping-data.csv")
print("After cleaning data the shape of data frame is : " + str(cleaned_df.shape))

cleaned_df
```

    After cleaning data the shape of data frame is : (37947, 3)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>topic</th>
      <th>category</th>
      <th>content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>america standing precipice monumental change f...</td>
      <td>analysis-and-opinion</td>
      <td>supreme court monumental decision loom america...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>time samoa rugby league sleeping giant awaken ...</td>
      <td>analysis-and-opinion</td>
      <td>sport samoa look begin time rugby league sleep...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>going ruck helping port ball impact rest league</td>
      <td>analysis-and-opinion</td>
      <td>sport inside game port adelaide change game ru...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>science silliness psychology socceroos world c...</td>
      <td>analysis-and-opinion</td>
      <td>socceroos goalkeeper andrew redmayne psycholog...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>rba bos message like job</td>
      <td>analysis-and-opinion</td>
      <td>rba bos want follow path away recession mean r...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>37942</th>
      <td>step building data collection office</td>
      <td>technology</td>
      <td>datadriveninvestor passive income easy heard t...</td>
    </tr>
    <tr>
      <th>37943</th>
      <td>talk child period</td>
      <td>technology</td>
      <td>k follow question good news talk time thing pa...</td>
    </tr>
    <tr>
      <th>37944</th>
      <td>thing differently</td>
      <td>technology</td>
      <td>gaming life key difference gaming kid versus a...</td>
    </tr>
    <tr>
      <th>37945</th>
      <td>paid directly fan</td>
      <td>technology</td>
      <td>k immediately looking good money making hack m...</td>
    </tr>
    <tr>
      <th>37946</th>
      <td>clarity v certainty strategy navigating vuca w...</td>
      <td>technology</td>
      <td>clarity v certainty strategy navigating vuca w...</td>
    </tr>
  </tbody>
</table>
<p>37947 rows × 3 columns</p>
</div>



### <a class="anchor" id="removecategories"></a> Remove categories with some articles


```python
# remove categories with less than 1500 total articles
for my_category in set(cleaned_df['category']):
    if(cleaned_df['category'].value_counts()[my_category] < 1500):
        cleaned_df.drop(cleaned_df.index[(cleaned_df["category"] == my_category)], axis=0, inplace=True)
```


```python
print(cleaned_df['category'].value_counts())
df = cleaned_df
```

    travel           2544
    business         2533
    politics         1978
    technology       1707
    sport            1701
    entertainment    1534
    food             1533
    health           1525
    Name: category, dtype: int64
    

# Finished cleaning the data!

## <a class="anchor" id="visualization"></a> Visualization of the 'cleaned' DataFrame

Let's check the distribution of the different categories across the dataset.

How many categories and how many articles in each category are there in the dataset?

### <a class="anchor" id="articlesPerCategory2"></a> How many articles per category?


```python
print('Number of Categories: ',df.groupby('category').ngroups)
print(df['category'].value_counts())
```

    Number of Categories:  8
    travel           2544
    business         2533
    politics         1978
    technology       1707
    sport            1701
    entertainment    1534
    food             1533
    health           1525
    Name: category, dtype: int64
    


```python
df.category.value_counts().plot(kind='bar', color='#348ABD',figsize=(10,7))
```




    <AxesSubplot:>




    
![png](output_111_1.png)
    



```python
import plotly.express as px

x = df['category'].value_counts().index.values.astype('str')
y = df['category'].value_counts().values
pct = [("%.2f"%(v*200))+"%"for v in (y/len(df))]

print("Number of articles in each category: ")

trace1 = px.bar(x=x, y=y, text=pct)
layout = dict(title= 'Number of articles in each category',
              yaxis = dict(title='Count'),
              xaxis = dict(title='category'))

fig = dict(data=[trace1], layout=layout)
trace1.show()
```

    Number of articles in each category: 
    


<div>                            <div id="343559c3-732c-453c-a3d6-9fc2f9f2e627" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("343559c3-732c-453c-a3d6-9fc2f9f2e627")) {                    Plotly.newPlot(                        "343559c3-732c-453c-a3d6-9fc2f9f2e627",                        [{"alignmentgroup":"True","hovertemplate":"x=%{x}<br>y=%{y}<br>text=%{text}<extra></extra>","legendgroup":"","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"","offsetgroup":"","orientation":"v","showlegend":false,"text":["33.80%","33.65%","26.28%","22.68%","22.60%","20.38%","20.37%","20.26%"],"textposition":"auto","x":["travel","business","politics","technology","sport","entertainment","food","health"],"xaxis":"x","y":[2544,2533,1978,1707,1701,1534,1533,1525],"yaxis":"y","type":"bar"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"x"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"y"}},"legend":{"tracegroupgap":0},"margin":{"t":60},"barmode":"relative"},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('343559c3-732c-453c-a3d6-9fc2f9f2e627');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
count = []
labels = []

fig = plt.figure(figsize = (8,8))

colors = ["skyblue"]
business = df[df['category'] == 'business']
travel = df[df['category'] == 'travel']
politics = df[df['category'] == 'politics']
technology = df[df['category'] == 'technology']
sport = df[df['category'] == 'sport']
entertainment = df[df['category'] == 'entertainment']
food = df[df['category'] == 'food']
health = df[df['category'] == 'health']

count = [business['category'].count(), travel['category'].count(), politics['category'].count(),
         technology['category'].count(), sport['category'].count(),entertainment['category'].count(),
         food['category'].count(), health['category'].count()]

pie = plt.pie(count, labels = ['business', 'travel', 'politics', 'technology', 'sport','entertainment', 'food','health'],
              autopct = "%1.1f%%",
              shadow = True,
              colors = colors,
              startangle = 45,
              explode = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
             )
```


    
![png](output_113_0.png)
    


### <a class="anchor" id="distribution2"></a> The distribution of the content' lengths


```python
lists = []
min_len = 100000
max_len = 0

for content in df['content']:
    words_count_in_article = len(str(content).split())
    if words_count_in_article < min_len:
        min_len = words_count_in_article
    if words_count_in_article > max_len:
        max_len = words_count_in_article
    lists.append(words_count_in_article)

print("Minimum content length in dataframe: ", min_len)
print("Maximum content length in dataframe: ", max_len)

plt.plot(lists)
plt.xlabel('Index of article')
plt.ylabel('Count of words')
plt.show()

mean = np.mean(lists)
print("The mean is: %.2f" % mean)

std = np.std(lists)
print("The std is: %.2f" % std)
```

    Minimum content length in dataframe:  2
    Maximum content length in dataframe:  14509
    


    
![png](output_115_1.png)
    


    The mean is: 339.39
    The std is: 416.63
    

### <a class="anchor" id="commonwords2"></a> Common words in dataset

#### What are the most common terms in our dataset? Let's find out!
#### <a class="anchor" id="barplot2"></a> Barplot


```python
freq_words(df.content)
```


    
![png](output_117_0.png)
    


Most common terms in topic and content for each category:


```python
for category in set(df["category"]):
    try:
        print(category + ":")
        print("most frequent terms in content: ")
        freq_words(df[df["category"] == category].content)

        print("most frequent terms in topic: ")
        freq_words(df[df["category"] == category].topic)
        print("--------------------------------------------------------------------------------------------------------------------")
    except:
        print(category)
```

    business:
    most frequent terms in content: 
    


    
![png](output_119_1.png)
    


    most frequent terms in topic: 
    


    
![png](output_119_3.png)
    


    --------------------------------------------------------------------------------------------------------------------
    technology:
    most frequent terms in content: 
    


    
![png](output_119_5.png)
    


    most frequent terms in topic: 
    


    
![png](output_119_7.png)
    


    --------------------------------------------------------------------------------------------------------------------
    sport:
    most frequent terms in content: 
    


    
![png](output_119_9.png)
    


    most frequent terms in topic: 
    


    
![png](output_119_11.png)
    


    --------------------------------------------------------------------------------------------------------------------
    health:
    most frequent terms in content: 
    


    
![png](output_119_13.png)
    


    most frequent terms in topic: 
    


    
![png](output_119_15.png)
    


    --------------------------------------------------------------------------------------------------------------------
    travel:
    most frequent terms in content: 
    


    
![png](output_119_17.png)
    


    most frequent terms in topic: 
    


    
![png](output_119_19.png)
    


    --------------------------------------------------------------------------------------------------------------------
    politics:
    most frequent terms in content: 
    


    
![png](output_119_21.png)
    


    most frequent terms in topic: 
    


    
![png](output_119_23.png)
    


    --------------------------------------------------------------------------------------------------------------------
    food:
    most frequent terms in content: 
    


    
![png](output_119_25.png)
    


    most frequent terms in topic: 
    


    
![png](output_119_27.png)
    


    --------------------------------------------------------------------------------------------------------------------
    entertainment:
    most frequent terms in content: 
    


    
![png](output_119_29.png)
    


    most frequent terms in topic: 
    


    
![png](output_119_31.png)
    


    --------------------------------------------------------------------------------------------------------------------
    

### <a class="anchor" id="wordcloud2"></a> WordCloud

#### What are the most common terms in our dataset? Let's find out!


```python
wordCloud_content()
```


    
![png](output_121_0.png)
    


Most common terms in topic and content for each category:


```python
for category in set(df["category"]):
    try:
        print(category + " WordCloud:")
        print("Content:")
        wordCloud_category_content_topic(category, 1)

        print("Topic:")
        wordCloud_category_content_topic(category, 0)
        print("--------------------------------------------------------------------------------------------------")
    except:
        print(category)
```

    business WordCloud:
    Content:
    


    
![png](output_123_1.png)
    


    Topic:
    


    
![png](output_123_3.png)
    


    --------------------------------------------------------------------------------------------------
    technology WordCloud:
    Content:
    


    
![png](output_123_5.png)
    


    Topic:
    


    
![png](output_123_7.png)
    


    --------------------------------------------------------------------------------------------------
    sport WordCloud:
    Content:
    


    
![png](output_123_9.png)
    


    Topic:
    


    
![png](output_123_11.png)
    


    --------------------------------------------------------------------------------------------------
    health WordCloud:
    Content:
    


    
![png](output_123_13.png)
    


    Topic:
    


    
![png](output_123_15.png)
    


    --------------------------------------------------------------------------------------------------
    travel WordCloud:
    Content:
    


    
![png](output_123_17.png)
    


    Topic:
    


    
![png](output_123_19.png)
    


    --------------------------------------------------------------------------------------------------
    politics WordCloud:
    Content:
    


    
![png](output_123_21.png)
    


    Topic:
    


    
![png](output_123_23.png)
    


    --------------------------------------------------------------------------------------------------
    food WordCloud:
    Content:
    


    
![png](output_123_25.png)
    


    Topic:
    


    
![png](output_123_27.png)
    


    --------------------------------------------------------------------------------------------------
    entertainment WordCloud:
    Content:
    


    
![png](output_123_29.png)
    


    Topic:
    


    
![png](output_123_31.png)
    


    --------------------------------------------------------------------------------------------------
    

# <a class="anchor" id="advancheddataanalysis"></a> <p style="color:blue">Part 5. Bag of Words and Machine Learning</p>

## <a class="anchor" id="bagofwords"></a> Create and Fit Bag of Words Model

### How many times words appear in each document?

The Bag of Words (BoW) model is a form of text representation in numbers. Like the term itself, we can represent a sentence as a bag of words vector (a string of numbers).

### Term Frequency (TF) + Inverse Document Frequency (IDF)
Term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.

Words with a higher score are more important, and those with a lower score are less important


```python
tfidf_vec = TfidfVectorizer(ngram_range=(1, 1), max_features=50000, use_idf=True)
features = tfidf_vec.fit_transform(df.content).toarray().astype(float)
labels = df.category
```


```python
features.shape
```




    (15055, 50000)



## <a class="anchor" id="trainandtest"></a> Train Test and Split the Dataset

We now have the data in a format we want: feature vectors.
The original data was divided into features (X) and target (y).

We need to split a dataset into train (70%) and test (30%) sets to evaluate how well our machine learning model performs. 

Thus, the algorithms would be trained on one set of data and tested out on a completely different set of data (not seen before by the algorithm).


```python
print("total of articles before splitting the dataset: " + str(len(df)))
```

    total of articles before splitting the dataset: 15055
    


```python
X = df['content'] # the articles content
y = df['category'] # what we will predict
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index,
                                                                               test_size = 0.3, random_state = 1)

print("Length of train set: " + str(len(X_train)))
print("Length of test set: " + str(len(X_test)))
```

    Length of train set: 10538
    Length of test set: 4517
    

## <a class="anchor" id="mlmodels"></a> Machine Learning Models:

We work with 4 main machine learning models:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Gaussian Naive Bayes
4. Linear SVC

We will train these models with the train set and then we will test the models with the test set.
In the end, you will get the accurency score for each.


```python
performance_of_model = []

def run_model_test(model_name, est_c, est_pnlty):
    
    function_of_model = ''
    
    if model_name == 'Logistic Regression':
        function_of_model = LogisticRegression()
        
    elif model_name == 'Multinomial Naive Bayes':
        function_of_model = MultinomialNB(alpha = 1.0, fit_prior = True)
        
    elif model_name == 'Gaussian Naive Bayes':
        function_of_model = GaussianNB()
        
    elif model_name == 'Linear SVC':
        function_of_model = LinearSVC()
    
    oneVsRest = OneVsRestClassifier(function_of_model)
    
    #tf_idf
    oneVsRest.fit(X_train, y_train)
    y_pred = oneVsRest.predict(X_test)
    
    # Performance metrics
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    
    # Get precision, recall, f1 scores
    precision, recall, f1score, support = score(y_test, y_pred, average = 'micro')

    print(metrics.classification_report(y_test, y_pred))

    print(f'Test Accuracy Score of Basic {model_name}: % {accuracy}')
    print(f'Precision : {precision}')
    print(f'Recall    : {recall}')
    print(f'F1-score   : {f1score}')
    print("")
    
    # Add performance parameters to list
    performance_of_model.append(dict([
        ('Model', model_name),
        ('Test Accuracy', round(accuracy, 2)),
        ('Precision', round(precision, 2)),
        ('Recall', round(recall, 2)),
        ('F1', round(f1score, 2))
    ]))
```

### <a class="anchor" id="Mnaivebayes"></a> Multinomial Naive Bayes


```python
run_model_test('Multinomial Naive Bayes', est_c=None, est_pnlty=None)
```

                   precision    recall  f1-score   support
    
         business       0.60      0.90      0.72       764
    entertainment       0.98      0.84      0.91       467
             food       0.92      0.78      0.84       411
           health       0.82      0.68      0.74       459
         politics       0.85      0.82      0.84       604
            sport       0.97      0.84      0.90       510
       technology       0.92      0.51      0.65       494
           travel       0.80      0.93      0.86       808
    
         accuracy                           0.81      4517
        macro avg       0.86      0.79      0.81      4517
     weighted avg       0.84      0.81      0.81      4517
    
    Test Accuracy Score of Basic Multinomial Naive Bayes: % 80.54
    Precision : 0.8054018153641798
    Recall    : 0.8054018153641798
    F1-score   : 0.8054018153641799
    
    

### <a class="anchor" id="logisticregression"></a> Logistic Regression 


```python
run_model_test('Logistic Regression', est_c=None, est_pnlty=None)
```

                   precision    recall  f1-score   support
    
         business       0.77      0.82      0.79       764
    entertainment       0.95      0.91      0.93       467
             food       0.86      0.89      0.87       411
           health       0.80      0.76      0.78       459
         politics       0.86      0.87      0.86       604
            sport       0.95      0.89      0.92       510
       technology       0.82      0.78      0.80       494
           travel       0.88      0.92      0.90       808
    
         accuracy                           0.86      4517
        macro avg       0.86      0.85      0.86      4517
     weighted avg       0.86      0.86      0.86      4517
    
    Test Accuracy Score of Basic Logistic Regression: % 85.72
    Precision : 0.857206110250166
    Recall    : 0.857206110250166
    F1-score   : 0.857206110250166
    
    

### <a class="anchor" id="Gnaivebayes"></a> Gaussian Naive Bayes


```python
run_model_test('Gaussian Naive Bayes', est_c=None, est_pnlty=None)
```

                   precision    recall  f1-score   support
    
         business       0.76      0.23      0.35       764
    entertainment       0.89      0.56      0.69       467
             food       0.93      0.29      0.44       411
           health       0.60      0.22      0.32       459
         politics       0.72      0.49      0.58       604
            sport       0.91      0.70      0.79       510
       technology       0.59      0.35      0.44       494
           travel       0.29      0.94      0.44       808
    
         accuracy                           0.50      4517
        macro avg       0.71      0.47      0.51      4517
     weighted avg       0.68      0.50      0.50      4517
    
    Test Accuracy Score of Basic Gaussian Naive Bayes: % 49.52
    Precision : 0.49524020367500554
    Recall    : 0.49524020367500554
    F1-score   : 0.49524020367500554
    
    

### <a class="anchor" id="svc"></a> SVC


```python
run_model_test('Linear SVC', est_c=None, est_pnlty=None)
```

                   precision    recall  f1-score   support
    
         business       0.79      0.79      0.79       764
    entertainment       0.94      0.93      0.93       467
             food       0.86      0.89      0.88       411
           health       0.80      0.79      0.79       459
         politics       0.87      0.87      0.87       604
            sport       0.95      0.90      0.93       510
       technology       0.79      0.81      0.80       494
           travel       0.90      0.91      0.90       808
    
         accuracy                           0.86      4517
        macro avg       0.86      0.86      0.86      4517
     weighted avg       0.86      0.86      0.86      4517
    
    Test Accuracy Score of Basic Linear SVC: % 86.07
    Precision : 0.8607482842594643
    Recall    : 0.8607482842594643
    F1-score   : 0.8607482842594643
    
    

### <a class="anchor" id="results"></a> Results:


```python
model_performance = pd.DataFrame(data = performance_of_model)
model_performance = model_performance[['Model', 'Test Accuracy', 'Precision', 'Recall', 'F1']]
model_performance
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Test Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Multinomial Naive Bayes</td>
      <td>80.54</td>
      <td>0.81</td>
      <td>0.81</td>
      <td>0.81</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Logistic Regression</td>
      <td>85.72</td>
      <td>0.86</td>
      <td>0.86</td>
      <td>0.86</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gaussian Naive Bayes</td>
      <td>49.52</td>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.50</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Linear SVC</td>
      <td>86.07</td>
      <td>0.86</td>
      <td>0.86</td>
      <td>0.86</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(18, 6))
sns.boxplot(x = 'Model', y = 'Test Accuracy', 
            data = model_performance, 
            color = 'lightblue', 
            showmeans = True)
plt.title("Accuracy \n", size = 16);
```


    
![png](output_142_0.png)
    


## <a class="anchor" id="bestmodel"></a> Best model!


```python
max_accuracy = model_performance["Test Accuracy"].max()
model_index = model_performance.index[model_performance["Test Accuracy"] == max_accuracy].values

best_model_name = model_performance['Model'][model_index].values
print("The best accuracy of model is", max_accuracy, "from:", best_model_name)
```

    The best accuracy of model is 86.07 from: ['Linear SVC']
    

## <a class="anchor" id="worstmodel"></a> Worst model


```python
min_accuracy = model_performance["Test Accuracy"].min()
worst_model_index = model_performance.index[model_performance["Test Accuracy"] == min_accuracy].values

worst_model_name = model_performance['Model'][worst_model_index].values
print("The best accuracy of model is", min_accuracy, "from:", worst_model_name)
```

    The best accuracy of model is 49.52 from: ['Gaussian Naive Bayes']
    

## <a class="anchor" id="confusionmatrix"></a> Confusion Matrix 
Classification problems to assess where errors in the model were made.
### The Best Classifier


```python
def create_confusion_matrix(model):
    fig, axes = plt.subplots(figsize=(11,11))
    plot_confusion_matrix(model, X_test, y_test, cmap='PuBu',ax=axes)
    axes.title.set_text("Confusion Matrix")
    plt.tight_layout()  
    plt.show()
```


```python
function_of_model = ''

if best_model_name == 'Logistic Regression':
    function_of_model = LogisticRegression()

elif best_model_name == 'Multinomial Naive Bayes':
    function_of_model = MultinomialNB(alpha = 1.0, fit_prior = True)

elif best_model_name == 'Gaussian Naive Bayes':
    function_of_model = GaussianNB()

elif best_model_name == 'Linear SVC':
    function_of_model = LinearSVC()

best_classifier = function_of_model.fit(X_train, y_train)
```


```python
print("Confusion Martix for: ", best_model_name)
create_confusion_matrix(best_classifier)
```

    Confusion Martix for:  ['Linear SVC']
    


    
![png](output_150_1.png)
    


### The Worst Classifier


```python
function_of_model = ''

if worst_model_name == 'Logistic Regression':
    function_of_model = LogisticRegression()

elif worst_model_name == 'Multinomial Naive Bayes':
    function_of_model = MultinomialNB(alpha = 1.0, fit_prior = True)

elif worst_model_name == 'Gaussian Naive Bayes':
    function_of_model = GaussianNB()

elif worst_model_name == 'Linear SVC':
    function_of_model = LinearSVC()


worst_classifier = function_of_model.fit(X_train, y_train)
worst_classifier
```




    GaussianNB()




```python
print("Confusion Martix for: ", worst_model_name)
create_confusion_matrix(worst_classifier)
```

    Confusion Martix for:  ['Gaussian Naive Bayes']
    


    
![png](output_153_1.png)
    


# <a class="anchor" id="woweffect"></a> Insert your data and we will tell you it category!
You can insert text to 'test.txt' 
We will use the best model to predict your category!


```python
with open('test.txt') as f:
    text_to_predict = f.read()
    print("Your data is: ", text_to_predict)
```

    Your data is:  government
    


```python
data = []

data.append(text_to_predict)
predict_new_data = tfidf_vec.transform(data)
result = best_classifier.predict(predict_new_data)

print("Your data is about: ")
print(result)
```

    Your data is about: 
    ['politics']
    

# <a class="anchor" id="Conclusions"></a> <p style="color:blue">Part 6. Conclusion</p>

<font size="3"> News article classification is very difficult process and we had to over the whole data and to do data cleaning very carefully. The scraping was very challenging because each website have differnet layout.
   
<font size="3"> Finally after doing scraping from many websites, data cleaning, data preprocessing, train_test_split model, creating a bag of words and machine learning models, we got an impressive accuracy score with Linear SVC model - 86%,  so it gives us the best accuracy among all machine learning models!

### <a class="anchor" id="credits"></a> Credits:

- https://pandas.pydata.org/docs/index.html
- https://monkeylearn.com/text-classification/
- https://www.abc.net.au
- https://www.articlebiz.com
- https://www.ezinearticles.com
- https://www.hubpages.com
- https://www.huffpost.com
- https://www.indiatoday.in
- https://www.inshorts.com
- https://www.medium.com
- https://www.moneycontrol.com
- https://www.crummy.com/software/BeautifulSoup/bs4/doc/
- https://medium.com/analytics-vidhya/bbc-news-text-classification-a1b2a61af903
- https://www.analyticsvidhya.com
- https://towardsdatascience.com
- https://www.selenium.dev/selenium/docs/api/py/api.html
- https://www.programiz.com/python-programming/csv#:~:text=To%20write%20to%20a%20CSV,data%20into%20a%20delimited%20string.
- https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html
- https://github.com/
- https://stackoverflow.com/
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- https://machinelearningmastery.com/gentle-introduction-bag-words-model/
- https://monkeylearn.com/blog/text-classification-machine-learning/#:~:text=Text%20classification%20is%20a%20machine,and%20more%20accurately%20than%20humans.
- https://www.projectpro.io/article/machine-learning-nlp-text-classification-algorithms-and-models/523


```python

```
