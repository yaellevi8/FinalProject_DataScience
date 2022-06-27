# Import


```python
import pandas as pd
import glob
import os

import unidecode 
import re
import time 
import stopwords 
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk import word_tokenize
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

#For ML models part
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB

from sklearn.metrics import plot_confusion_matrix

import warnings
warnings.filterwarnings('ignore')
```

# <p style="color:blue"> Part 1. Scraping the data </p>

### Merge all data into single DataFrame


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
print("Start merge")
merge_csv_files()
print("Merged!")
```

    Start merge
    Merged!
    

### The DataFrame is ready to use!


```python
df = get_data_from_csv("merge.csv")
print(str(df.shape))
df
```

    (42141, 4)
    




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



# <p style="color:blue"> Part 2. Initial cleaning data </p>

### 1. Removing rows and columns with Null/NaN values. 


```python
# 
print("Before removing null objects" , str(df.shape))
df.dropna(inplace=True)
print("After removing null objects" ,  str(df.shape))
```

    Before removing null objects (42141, 4)
    After removing null objects (41632, 4)
    

### 2. Dropping duplicate rows


```python
print("Before dropping duplicates" ,  str(df.shape))
df.drop_duplicates('content', inplace = True)
print("After dropping duplicates" ,  str(df.shape))
```

    Before dropping duplicates (41632, 4)
    After dropping duplicates (37951, 4)
    

### 3. Dropping short data



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



# <p style="color:blue"> Part 3. EDA & Visualiztion </p>

### Exploratory Data Analysis

Let's check the distribution of the different categories across the dataset.

How many categories and how many articles in each category are there in the dataset?

### How many articles per category?


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




    
![png](output_22_1.png)
    



```python
# https://plotly.com/python/bar-charts/
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
    


<div>                            <div id="12f6d5c3-4638-4885-90a2-0957b5179ed1" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("12f6d5c3-4638-4885-90a2-0957b5179ed1")) {                    Plotly.newPlot(                        "12f6d5c3-4638-4885-90a2-0957b5179ed1",                        [{"alignmentgroup":"True","hovertemplate":"x=%{x}<br>y=%{y}<br>text=%{text}<extra></extra>","legendgroup":"","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"","offsetgroup":"","orientation":"v","showlegend":false,"text":["10.43%","10.19%","9.00%","8.19%","8.08%","8.04%","5.87%","5.27%","5.27%","5.27%","5.27%","5.27%","5.27%","4.92%","4.73%","4.53%","3.75%","3.73%","3.16%","3.16%","3.16%","3.16%","3.16%","3.16%","3.16%","3.16%","3.16%","3.16%","3.16%","3.16%","3.16%","3.16%","3.15%","2.95%","2.95%","2.95%","2.95%","2.95%","2.94%","2.94%","2.93%","2.71%","2.41%","2.32%","2.20%","2.04%","1.04%","0.77%","0.62%","0.60%","0.53%","0.28%","0.26%","0.11%","0.06%","0.04%"],"textposition":"auto","x":["politics","business","technology","sports","entertainment","health","money","style","education","literature","holidays","travel","games-hobbies","food","relationships","science","world","autos","Communications","News-and-Society","Investing","Home-and-Family","Health-and-Fitness","Food-and-Drink","Finance","religion-philosophy","Business","pets","news-society","Computers-and-Technology","Travel-and-Leisure","Arts-and-Entertainment","Pets","social-issues","autos-trucks","arts-entertainment","finance","travel-leisure","shopping","health-fitness","computers-technology","startup","house-and-home","environment","markets","travel-and-tourism-lifestyle-and-leisure","arts-culture","sport","marketing","entrepreneurship","society","community","art","analysis-and-opinion","shopping-mall","life"],"xaxis":"x","y":[1978,1933,1707,1554,1534,1525,1113,1000,1000,999,999,999,999,933,898,860,711,707,600,600,600,600,600,600,600,600,600,600,600,599,599,599,597,560,560,560,560,559,558,557,556,514,458,441,417,387,197,147,117,113,100,53,50,21,12,7],"yaxis":"y","type":"bar"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"x"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"y"}},"legend":{"tracegroupgap":0},"margin":{"t":60},"barmode":"relative"},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('12f6d5c3-4638-4885-90a2-0957b5179ed1');
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


Let's now check the disctribution of the content' lengths.


```python
df.content.map(len).hist(figsize=(15, 8))
```




    <AxesSubplot:>




    
![png](output_25_1.png)
    


### Common words in dataset

#### What are the most common terms in our dataset? Let's find out!


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


    
![png](output_28_0.png)
    


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

    pets:
    most frequent terms in content: 
    


    
![png](output_30_1.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_3.png)
    


    --------------------------------------------------------------------------------------------------------------------
    arts-entertainment:
    most frequent terms in content: 
    


    
![png](output_30_5.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_7.png)
    


    --------------------------------------------------------------------------------------------------------------------
    society:
    most frequent terms in content: 
    


    
![png](output_30_9.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_11.png)
    


    --------------------------------------------------------------------------------------------------------------------
    Home-and-Family:
    most frequent terms in content: 
    


    
![png](output_30_13.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_15.png)
    


    --------------------------------------------------------------------------------------------------------------------
    News-and-Society:
    most frequent terms in content: 
    


    
![png](output_30_17.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_19.png)
    


    --------------------------------------------------------------------------------------------------------------------
    Investing:
    most frequent terms in content: 
    


    
![png](output_30_21.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_23.png)
    


    --------------------------------------------------------------------------------------------------------------------
    Health-and-Fitness:
    most frequent terms in content: 
    


    
![png](output_30_25.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_27.png)
    


    --------------------------------------------------------------------------------------------------------------------
    community:
    most frequent terms in content: 
    


    
![png](output_30_29.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_31.png)
    


    --------------------------------------------------------------------------------------------------------------------
    marketing:
    most frequent terms in content: 
    


    
![png](output_30_33.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_35.png)
    


    --------------------------------------------------------------------------------------------------------------------
    sport:
    most frequent terms in content: 
    


    
![png](output_30_37.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_39.png)
    


    --------------------------------------------------------------------------------------------------------------------
    autos:
    most frequent terms in content: 
    


    
![png](output_30_41.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_43.png)
    


    --------------------------------------------------------------------------------------------------------------------
    style:
    most frequent terms in content: 
    


    
![png](output_30_45.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_47.png)
    


    --------------------------------------------------------------------------------------------------------------------
    life:
    most frequent terms in content: 
    


    
![png](output_30_49.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_51.png)
    


    --------------------------------------------------------------------------------------------------------------------
    science:
    most frequent terms in content: 
    


    
![png](output_30_53.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_55.png)
    


    --------------------------------------------------------------------------------------------------------------------
    business:
    most frequent terms in content: 
    


    
![png](output_30_57.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_59.png)
    


    --------------------------------------------------------------------------------------------------------------------
    Arts-and-Entertainment:
    most frequent terms in content: 
    


    
![png](output_30_61.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_63.png)
    


    --------------------------------------------------------------------------------------------------------------------
    art:
    most frequent terms in content: 
    


    
![png](output_30_65.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_67.png)
    


    --------------------------------------------------------------------------------------------------------------------
    Food-and-Drink:
    most frequent terms in content: 
    


    
![png](output_30_69.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_71.png)
    


    --------------------------------------------------------------------------------------------------------------------
    money:
    most frequent terms in content: 
    


    
![png](output_30_73.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_75.png)
    


    --------------------------------------------------------------------------------------------------------------------
    games-hobbies:
    most frequent terms in content: 
    


    
![png](output_30_77.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_79.png)
    


    --------------------------------------------------------------------------------------------------------------------
    analysis-and-opinion:
    most frequent terms in content: 
    


    
![png](output_30_81.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_83.png)
    


    --------------------------------------------------------------------------------------------------------------------
    Business:
    most frequent terms in content: 
    


    
![png](output_30_85.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_87.png)
    


    --------------------------------------------------------------------------------------------------------------------
    finance:
    most frequent terms in content: 
    


    
![png](output_30_89.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_91.png)
    


    --------------------------------------------------------------------------------------------------------------------
    shopping:
    most frequent terms in content: 
    


    
![png](output_30_93.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_95.png)
    


    --------------------------------------------------------------------------------------------------------------------
    literature:
    most frequent terms in content: 
    


    
![png](output_30_97.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_99.png)
    


    --------------------------------------------------------------------------------------------------------------------
    health-fitness:
    most frequent terms in content: 
    


    
![png](output_30_101.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_103.png)
    


    --------------------------------------------------------------------------------------------------------------------
    education:
    most frequent terms in content: 
    


    
![png](output_30_105.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_107.png)
    


    --------------------------------------------------------------------------------------------------------------------
    news-society:
    most frequent terms in content: 
    


    
![png](output_30_109.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_111.png)
    


    --------------------------------------------------------------------------------------------------------------------
    Pets:
    most frequent terms in content: 
    


    
![png](output_30_113.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_115.png)
    


    --------------------------------------------------------------------------------------------------------------------
    sports:
    most frequent terms in content: 
    


    
![png](output_30_117.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_119.png)
    


    --------------------------------------------------------------------------------------------------------------------
    Communications:
    most frequent terms in content: 
    


    
![png](output_30_121.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_123.png)
    


    --------------------------------------------------------------------------------------------------------------------
    entrepreneurship:
    most frequent terms in content: 
    


    
![png](output_30_125.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_127.png)
    


    --------------------------------------------------------------------------------------------------------------------
    Travel-and-Leisure:
    most frequent terms in content: 
    


    
![png](output_30_129.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_131.png)
    


    --------------------------------------------------------------------------------------------------------------------
    entertainment:
    most frequent terms in content: 
    


    
![png](output_30_133.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_135.png)
    


    --------------------------------------------------------------------------------------------------------------------
    startup:
    most frequent terms in content: 
    


    
![png](output_30_137.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_139.png)
    


    --------------------------------------------------------------------------------------------------------------------
    world:
    most frequent terms in content: 
    


    
![png](output_30_141.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_143.png)
    


    --------------------------------------------------------------------------------------------------------------------
    autos-trucks:
    most frequent terms in content: 
    


    
![png](output_30_145.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_147.png)
    


    --------------------------------------------------------------------------------------------------------------------
    arts-culture:
    most frequent terms in content: 
    


    
![png](output_30_149.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_151.png)
    


    --------------------------------------------------------------------------------------------------------------------
    technology:
    most frequent terms in content: 
    


    
![png](output_30_153.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_155.png)
    


    --------------------------------------------------------------------------------------------------------------------
    Finance:
    most frequent terms in content: 
    


    
![png](output_30_157.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_159.png)
    


    --------------------------------------------------------------------------------------------------------------------
    shopping-mall:
    most frequent terms in content: 
    


    
![png](output_30_161.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_163.png)
    


    --------------------------------------------------------------------------------------------------------------------
    house-and-home:
    most frequent terms in content: 
    


    
![png](output_30_165.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_167.png)
    


    --------------------------------------------------------------------------------------------------------------------
    computers-technology:
    most frequent terms in content: 
    


    
![png](output_30_169.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_171.png)
    


    --------------------------------------------------------------------------------------------------------------------
    health:
    most frequent terms in content: 
    


    
![png](output_30_173.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_175.png)
    


    --------------------------------------------------------------------------------------------------------------------
    travel-leisure:
    most frequent terms in content: 
    


    
![png](output_30_177.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_179.png)
    


    --------------------------------------------------------------------------------------------------------------------
    travel-and-tourism-lifestyle-and-leisure:
    most frequent terms in content: 
    


    
![png](output_30_181.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_183.png)
    


    --------------------------------------------------------------------------------------------------------------------
    environment:
    most frequent terms in content: 
    


    
![png](output_30_185.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_187.png)
    


    --------------------------------------------------------------------------------------------------------------------
    politics:
    most frequent terms in content: 
    


    
![png](output_30_189.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_191.png)
    


    --------------------------------------------------------------------------------------------------------------------
    holidays:
    most frequent terms in content: 
    


    
![png](output_30_193.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_195.png)
    


    --------------------------------------------------------------------------------------------------------------------
    markets:
    most frequent terms in content: 
    


    
![png](output_30_197.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_199.png)
    


    --------------------------------------------------------------------------------------------------------------------
    food:
    most frequent terms in content: 
    


    
![png](output_30_201.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_203.png)
    


    --------------------------------------------------------------------------------------------------------------------
    travel:
    most frequent terms in content: 
    


    
![png](output_30_205.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_207.png)
    


    --------------------------------------------------------------------------------------------------------------------
    social-issues:
    most frequent terms in content: 
    


    
![png](output_30_209.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_211.png)
    


    --------------------------------------------------------------------------------------------------------------------
    Computers-and-Technology:
    most frequent terms in content: 
    


    
![png](output_30_213.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_215.png)
    


    --------------------------------------------------------------------------------------------------------------------
    religion-philosophy:
    most frequent terms in content: 
    


    
![png](output_30_217.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_219.png)
    


    --------------------------------------------------------------------------------------------------------------------
    relationships:
    most frequent terms in content: 
    


    
![png](output_30_221.png)
    


    most frequent terms in topic: 
    


    
![png](output_30_223.png)
    


    --------------------------------------------------------------------------------------------------------------------
    

### WordCloud

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


    
![png](output_33_0.png)
    


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

    pets WordCloud:
    Content:
    


    
![png](output_35_1.png)
    


    Topic:
    


    
![png](output_35_3.png)
    


    --------------------------------------------------------------------------------------------------
    arts-entertainment WordCloud:
    Content:
    


    
![png](output_35_5.png)
    


    Topic:
    


    
![png](output_35_7.png)
    


    --------------------------------------------------------------------------------------------------
    society WordCloud:
    Content:
    


    
![png](output_35_9.png)
    


    Topic:
    


    
![png](output_35_11.png)
    


    --------------------------------------------------------------------------------------------------
    Home-and-Family WordCloud:
    Content:
    


    
![png](output_35_13.png)
    


    Topic:
    


    
![png](output_35_15.png)
    


    --------------------------------------------------------------------------------------------------
    News-and-Society WordCloud:
    Content:
    


    
![png](output_35_17.png)
    


    Topic:
    


    
![png](output_35_19.png)
    


    --------------------------------------------------------------------------------------------------
    Investing WordCloud:
    Content:
    


    
![png](output_35_21.png)
    


    Topic:
    


    
![png](output_35_23.png)
    


    --------------------------------------------------------------------------------------------------
    Health-and-Fitness WordCloud:
    Content:
    


    
![png](output_35_25.png)
    


    Topic:
    


    
![png](output_35_27.png)
    


    --------------------------------------------------------------------------------------------------
    community WordCloud:
    Content:
    


    
![png](output_35_29.png)
    


    Topic:
    


    
![png](output_35_31.png)
    


    --------------------------------------------------------------------------------------------------
    marketing WordCloud:
    Content:
    


    
![png](output_35_33.png)
    


    Topic:
    


    
![png](output_35_35.png)
    


    --------------------------------------------------------------------------------------------------
    sport WordCloud:
    Content:
    


    
![png](output_35_37.png)
    


    Topic:
    


    
![png](output_35_39.png)
    


    --------------------------------------------------------------------------------------------------
    autos WordCloud:
    Content:
    


    
![png](output_35_41.png)
    


    Topic:
    


    
![png](output_35_43.png)
    


    --------------------------------------------------------------------------------------------------
    style WordCloud:
    Content:
    


    
![png](output_35_45.png)
    


    Topic:
    


    
![png](output_35_47.png)
    


    --------------------------------------------------------------------------------------------------
    life WordCloud:
    Content:
    


    
![png](output_35_49.png)
    


    Topic:
    


    
![png](output_35_51.png)
    


    --------------------------------------------------------------------------------------------------
    science WordCloud:
    Content:
    


    
![png](output_35_53.png)
    


    Topic:
    


    
![png](output_35_55.png)
    


    --------------------------------------------------------------------------------------------------
    business WordCloud:
    Content:
    


    
![png](output_35_57.png)
    


    Topic:
    


    
![png](output_35_59.png)
    


    --------------------------------------------------------------------------------------------------
    Arts-and-Entertainment WordCloud:
    Content:
    


    
![png](output_35_61.png)
    


    Topic:
    


    
![png](output_35_63.png)
    


    --------------------------------------------------------------------------------------------------
    art WordCloud:
    Content:
    


    
![png](output_35_65.png)
    


    Topic:
    


    
![png](output_35_67.png)
    


    --------------------------------------------------------------------------------------------------
    Food-and-Drink WordCloud:
    Content:
    


    
![png](output_35_69.png)
    


    Topic:
    


    
![png](output_35_71.png)
    


    --------------------------------------------------------------------------------------------------
    money WordCloud:
    Content:
    


    
![png](output_35_73.png)
    


    Topic:
    


    
![png](output_35_75.png)
    


    --------------------------------------------------------------------------------------------------
    games-hobbies WordCloud:
    Content:
    


    
![png](output_35_77.png)
    


    Topic:
    


    
![png](output_35_79.png)
    


    --------------------------------------------------------------------------------------------------
    analysis-and-opinion WordCloud:
    Content:
    


    
![png](output_35_81.png)
    


    Topic:
    


    
![png](output_35_83.png)
    


    --------------------------------------------------------------------------------------------------
    Business WordCloud:
    Content:
    


    
![png](output_35_85.png)
    


    Topic:
    


    
![png](output_35_87.png)
    


    --------------------------------------------------------------------------------------------------
    finance WordCloud:
    Content:
    


    
![png](output_35_89.png)
    


    Topic:
    


    
![png](output_35_91.png)
    


    --------------------------------------------------------------------------------------------------
    shopping WordCloud:
    Content:
    


    
![png](output_35_93.png)
    


    Topic:
    


    
![png](output_35_95.png)
    


    --------------------------------------------------------------------------------------------------
    literature WordCloud:
    Content:
    


    
![png](output_35_97.png)
    


    Topic:
    


    
![png](output_35_99.png)
    


    --------------------------------------------------------------------------------------------------
    health-fitness WordCloud:
    Content:
    


    
![png](output_35_101.png)
    


    Topic:
    


    
![png](output_35_103.png)
    


    --------------------------------------------------------------------------------------------------
    education WordCloud:
    Content:
    


    
![png](output_35_105.png)
    


    Topic:
    


    
![png](output_35_107.png)
    


    --------------------------------------------------------------------------------------------------
    news-society WordCloud:
    Content:
    


    
![png](output_35_109.png)
    


    Topic:
    


    
![png](output_35_111.png)
    


    --------------------------------------------------------------------------------------------------
    Pets WordCloud:
    Content:
    


    
![png](output_35_113.png)
    


    Topic:
    


    
![png](output_35_115.png)
    


    --------------------------------------------------------------------------------------------------
    sports WordCloud:
    Content:
    


    
![png](output_35_117.png)
    


    Topic:
    


    
![png](output_35_119.png)
    


    --------------------------------------------------------------------------------------------------
    Communications WordCloud:
    Content:
    


    
![png](output_35_121.png)
    


    Topic:
    


    
![png](output_35_123.png)
    


    --------------------------------------------------------------------------------------------------
    entrepreneurship WordCloud:
    Content:
    


    
![png](output_35_125.png)
    


    Topic:
    


    
![png](output_35_127.png)
    


    --------------------------------------------------------------------------------------------------
    Travel-and-Leisure WordCloud:
    Content:
    


    
![png](output_35_129.png)
    


    Topic:
    


    
![png](output_35_131.png)
    


    --------------------------------------------------------------------------------------------------
    entertainment WordCloud:
    Content:
    


    
![png](output_35_133.png)
    


    Topic:
    


    
![png](output_35_135.png)
    


    --------------------------------------------------------------------------------------------------
    startup WordCloud:
    Content:
    


    
![png](output_35_137.png)
    


    Topic:
    


    
![png](output_35_139.png)
    


    --------------------------------------------------------------------------------------------------
    world WordCloud:
    Content:
    


    
![png](output_35_141.png)
    


    Topic:
    


    
![png](output_35_143.png)
    


    --------------------------------------------------------------------------------------------------
    autos-trucks WordCloud:
    Content:
    


    
![png](output_35_145.png)
    


    Topic:
    


    
![png](output_35_147.png)
    


    --------------------------------------------------------------------------------------------------
    arts-culture WordCloud:
    Content:
    


    
![png](output_35_149.png)
    


    Topic:
    


    
![png](output_35_151.png)
    


    --------------------------------------------------------------------------------------------------
    technology WordCloud:
    Content:
    


    
![png](output_35_153.png)
    


    Topic:
    


    
![png](output_35_155.png)
    


    --------------------------------------------------------------------------------------------------
    Finance WordCloud:
    Content:
    


    
![png](output_35_157.png)
    


    Topic:
    


    
![png](output_35_159.png)
    


    --------------------------------------------------------------------------------------------------
    shopping-mall WordCloud:
    Content:
    


    
![png](output_35_161.png)
    


    Topic:
    


    
![png](output_35_163.png)
    


    --------------------------------------------------------------------------------------------------
    house-and-home WordCloud:
    Content:
    


    
![png](output_35_165.png)
    


    Topic:
    


    
![png](output_35_167.png)
    


    --------------------------------------------------------------------------------------------------
    computers-technology WordCloud:
    Content:
    


    
![png](output_35_169.png)
    


    Topic:
    


    
![png](output_35_171.png)
    


    --------------------------------------------------------------------------------------------------
    health WordCloud:
    Content:
    


    
![png](output_35_173.png)
    


    Topic:
    


    
![png](output_35_175.png)
    


    --------------------------------------------------------------------------------------------------
    travel-leisure WordCloud:
    Content:
    


    
![png](output_35_177.png)
    


    Topic:
    


    
![png](output_35_179.png)
    


    --------------------------------------------------------------------------------------------------
    travel-and-tourism-lifestyle-and-leisure WordCloud:
    Content:
    


    
![png](output_35_181.png)
    


    Topic:
    


    
![png](output_35_183.png)
    


    --------------------------------------------------------------------------------------------------
    environment WordCloud:
    Content:
    


    
![png](output_35_185.png)
    


    Topic:
    


    
![png](output_35_187.png)
    


    --------------------------------------------------------------------------------------------------
    politics WordCloud:
    Content:
    


    
![png](output_35_189.png)
    


    Topic:
    


    
![png](output_35_191.png)
    


    --------------------------------------------------------------------------------------------------
    holidays WordCloud:
    Content:
    


    
![png](output_35_193.png)
    


    Topic:
    


    
![png](output_35_195.png)
    


    --------------------------------------------------------------------------------------------------
    markets WordCloud:
    Content:
    


    
![png](output_35_197.png)
    


    Topic:
    


    
![png](output_35_199.png)
    


    --------------------------------------------------------------------------------------------------
    food WordCloud:
    Content:
    


    
![png](output_35_201.png)
    


    Topic:
    


    
![png](output_35_203.png)
    


    --------------------------------------------------------------------------------------------------
    travel WordCloud:
    Content:
    


    
![png](output_35_205.png)
    


    Topic:
    


    
![png](output_35_207.png)
    


    --------------------------------------------------------------------------------------------------
    social-issues WordCloud:
    Content:
    


    
![png](output_35_209.png)
    


    Topic:
    


    
![png](output_35_211.png)
    


    --------------------------------------------------------------------------------------------------
    Computers-and-Technology WordCloud:
    Content:
    


    
![png](output_35_213.png)
    


    Topic:
    


    
![png](output_35_215.png)
    


    --------------------------------------------------------------------------------------------------
    religion-philosophy WordCloud:
    Content:
    


    
![png](output_35_217.png)
    


    Topic:
    


    
![png](output_35_219.png)
    


    --------------------------------------------------------------------------------------------------
    relationships WordCloud:
    Content:
    


    
![png](output_35_221.png)
    


    Topic:
    


    
![png](output_35_223.png)
    


    --------------------------------------------------------------------------------------------------
    

## Conclusion of EDA

- a lot of stopwords
- remove categories with some srticles

#  <p style="color:blue">Part 4. Cleaning the data after EDA </p>

- Lowercase the data
- Remove links
- remove punctuation, stop words
- Lemmatization the words
- Union categories with same meaning
- Remove categories with some articles - less than 1500

# Stop words
Text and document classification over social media, such as Twitter, Facebook, and so on is usually affected by the noisy nature (abbreviations, irregular forms) of the text corpuses.


# Capitalization
Sentences can contain a mixture of uppercase and lower case letters. Multiple sentences make up a text document. To reduce the problem space, the most common approach is to reduce everything to lower case. This brings all words in a document in same space, but it often changes the meaning of some words, such as "US" to "us" where first one represents the United States of America and second one is a pronoun. To solve this, slang and abbreviation converters can be applied.

# Noise Removal
Another issue of text cleaning as a pre-processing step is noise removal. Text documents generally contains characters like punctuations or special characters and they are not necessary for text mining or classification purposes. Although punctuation is critical to understand the meaning of the sentence, but it can affect the classification algorithms negatively.

# Lemmatization
process of grouping together the different inflected forms of a word so they can be analyzed as a single item.


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
def remove_non_english_articles_and_remove_duplicated_rows(df):
    # dropping ALL duplicate values (keep only one)
    
#     print("DETECT TOPICS:")
#     for topic in df["topic"]:
#         DetectorFactory.seed = 0
#         if detect(topic) != "en":
#             print(topic)
#             print("Found different language: " + detect(topic))
#             df.drop(df.index[(df["topic"] == topic)], axis=0, inplace=True)
    
#     print("DETECT CONTENTS:")
#     for content in df["content"]:
#         DetectorFactory.seed = 0
#         if detect(content) != "en":
#             print("Found different language: " + detect(content))
#             df.drop(df.index[(df["content"] == content)], axis=0, inplace=True) #axis 0 for rows
    
#     print("DETECT CATEGORIES:")
#     for category in df["category"]:   
#         DetectorFactory.seed = 0
#         if detect(topic) != "en":
#             print(category)
#             print("Found different language: " + detect(topic))
#             df.drop(df.index[(df["category"] == category)], axis=0, inplace=True)
            

    return df
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
df = drop_rows_with_short_content(df)
print("After drop_rows_with_short_content the shape of data frame is : " + str(df.shape))

df = remove_non_english_articles_and_remove_duplicated_rows(df)
print("After remove_non_english_articles_and_remove_duplicated_rows the shape of data frame is : " + str(df.shape))

title_list = cleaning_preprocessing_data_from_csv(df["topic"])
print("Finish cleaning on topic")

content_list = cleaning_preprocessing_data_from_csv(df["content"])
print("Finish cleaning on content")

category_list = lower_category_and_union_categories_with_same_meaning()
```

    After drop_rows_with_short_content the shape of data frame is : (37947, 4)
    After remove_non_english_articles_and_remove_duplicated_rows the shape of data frame is : (37947, 4)
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



### Remove categories with some articles


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

## Visualiztion of the 'cleaned' DataFrame

Let's check the distribution of the different categories across the dataset.

How many categories and how many articles in each category are there in the dataset?

### How many articles per category?


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




    
![png](output_53_1.png)
    



```python
# https://plotly.com/python/bar-charts/
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
    


<div>                            <div id="ad5a4a1c-832c-4e43-b13e-ad96c169547d" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("ad5a4a1c-832c-4e43-b13e-ad96c169547d")) {                    Plotly.newPlot(                        "ad5a4a1c-832c-4e43-b13e-ad96c169547d",                        [{"alignmentgroup":"True","hovertemplate":"x=%{x}<br>y=%{y}<br>text=%{text}<extra></extra>","legendgroup":"","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"","offsetgroup":"","orientation":"v","showlegend":false,"text":["33.80%","33.65%","26.28%","22.68%","22.60%","20.38%","20.37%","20.26%"],"textposition":"auto","x":["travel","business","politics","technology","sport","entertainment","food","health"],"xaxis":"x","y":[2544,2533,1978,1707,1701,1534,1533,1525],"yaxis":"y","type":"bar"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"x"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"y"}},"legend":{"tracegroupgap":0},"margin":{"t":60},"barmode":"relative"},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('ad5a4a1c-832c-4e43-b13e-ad96c169547d');
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
# # https://www.analyticsvidhya.com/blog/2021/12/text-classification-of-news-articles/

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


    
![png](output_55_0.png)
    


Let's now check the disctribution of the content' lengths.




```python
df.content.map(len).hist(figsize=(20, 8))
```




    <AxesSubplot:>




    
![png](output_57_1.png)
    


### Common words in dataset

#### What are the most common terms in our dataset? Let's find out!


```python
freq_words(df.content)
```


    
![png](output_59_0.png)
    


Most common terms in topic and content for each category:


```python
# for category in set(df["category"]):
#     try:
#         print(category + ":")
#         print("most frequent terms in content: ")
#         freq_words(df[df["category"] == category].content)

#         print("most frequent terms in topic: ")
#         freq_words(df[df["category"] == category].topic)
#         print("--------------------------------------------------------------------------------------------------------------------")
#     except:
#         print(category)
```

### WordCloud

#### What are the most common terms in our dataset? Let's find out!


```python
wordCloud_content()
```


    
![png](output_63_0.png)
    


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

    entertainment WordCloud:
    Content:
    


    
![png](output_65_1.png)
    


    Topic:
    


    
![png](output_65_3.png)
    


    --------------------------------------------------------------------------------------------------
    politics WordCloud:
    Content:
    


    
![png](output_65_5.png)
    


    Topic:
    


    
![png](output_65_7.png)
    


    --------------------------------------------------------------------------------------------------
    sport WordCloud:
    Content:
    


    
![png](output_65_9.png)
    


    Topic:
    


    
![png](output_65_11.png)
    


    --------------------------------------------------------------------------------------------------
    food WordCloud:
    Content:
    


    
![png](output_65_13.png)
    


    Topic:
    


    
![png](output_65_15.png)
    


    --------------------------------------------------------------------------------------------------
    technology WordCloud:
    Content:
    


    
![png](output_65_17.png)
    


    Topic:
    


    
![png](output_65_19.png)
    


    --------------------------------------------------------------------------------------------------
    travel WordCloud:
    Content:
    


    
![png](output_65_21.png)
    


    Topic:
    


    
![png](output_65_23.png)
    


    --------------------------------------------------------------------------------------------------
    business WordCloud:
    Content:
    


    
![png](output_65_25.png)
    


    Topic:
    


    
![png](output_65_27.png)
    


    --------------------------------------------------------------------------------------------------
    health WordCloud:
    Content:
    


    
![png](output_65_29.png)
    


    Topic:
    


    
![png](output_65_31.png)
    


    --------------------------------------------------------------------------------------------------
    

### Tokenization

#### Text into Words
Tokenization is the process of breaking down a stream of text into words.
The main goal of this step is to extract individual words in a sentence.

Along with text classifcation, in text mining, it is necessary to incorporate
a parser in the pipeline which performs the tokenization of the documents


```python
# def tokenizer(text):
#     tokens = [word_tokenize(sent) for sent in sent_tokenize(text)]
#     tokens = list(reduce(lambda x,y: x+y, tokens))
#     tokens = list(filter(lambda token: token not in (stop_words + list(punctuation)) , tokens))
#     return tokens
```

PyLDAvis allows us to interpret the topics in a topic model like below:

Let’s use pyLDAvis to visualize the topics:

- Each bubble represents a topic. The larger the bubble, the higher percentage of the number of tweets in the corpus is about that topic.
- Blue bars represent the overall frequency of each word in the corpus. If no topic is selected, the blue bars of the most frequently used words will be displayed.
- Red bars give the estimated number of times a given term was generated by a given topic. As you can see from the image below, there are about 22,000 of the word ‘go’, and this term is used about 10,000 times within topic 1. The word with the longest red bar is the word that is used the most by the tweets belonging to that topic.


For more detailes about this model - https://neptune.ai/blog/pyldavis-topic-modelling-exploration-tool-that-every-nlp-data-scientist-should-know


```python
# # https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

# def get_data_from_csv(url):
#     return pd.read_csv(url)

# # Gensim
# import gensim
# import gensim.corpora as corpora
# from gensim.utils import simple_preprocess
# from gensim.models import CoherenceModel

# # Plotting tools
# import pyLDAvis
# import pyLDAvis.gensim_models  # don't skip this
# import matplotlib.pyplot as plt

# def tokenizer(sentences):
#     for sentence in sentences:
#         yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

# tokens = list(tokenizer(df["content"]))


# # Create Dictionary
# id2word = corpora.Dictionary(tokens)

# # Term Document Frequency
# corpus = [id2word.doc2bow(text) for text in tokens]

# # View
# print(corpus[:1])

# # Human readable format of corpus (term-frequency)
# [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

# lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
#                                            id2word=id2word,
#                                            num_topics=20, 
#                                            random_state=100,
#                                            update_every=1,
#                                            chunksize=100,
#                                            passes=10,
#                                            alpha='auto',
#                                            per_word_topics=True)

# # You can see the keywords for each topic and the weightage(importance) of each keyword using lda_model.
# # print_topics() as shown next.
# print(lda_model.print_topics())
# doc_lda = lda_model[corpus]

# # Visualize the topics
# pyLDAvis.enable_notebook()
# vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
# vis
```

# <p style="color:blue">Part 5. Advanced Data Analysis</p>

## Create and Fit Bag of Words Model

https://www.analyticsvidhya.com/blog/2020/02/quick-introduction-bag-of-words-bow-tf-idf/
https://www.analyticsvidhya.com/blog/2021/12/text-classification-of-news-articles/

### How many times words appear in each document?

The Bag of Words (BoW) model is a form of text representation in numbers. Like the term itself, we can represent a sentence as a bag of words vector (a string of numbers).

### Term Frequency (TF) + Inverse Document Frequency (IDF)
Term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.

Words with a higher score are more important, and those with a lower score are less important


```python
# Defining a TF-IDF Vectorizer
tfidf_vec = TfidfVectorizer(ngram_range=(1, 1), max_features=50000, use_idf=True)
```


```python
features = tfidf_vec.fit_transform(df.content).toarray().astype(float)
labels = df.category

features.shape
```




    (15055, 50000)



## Train Test and Split the Dataset

We need to split a dataset into train and test sets to evaluate how well our machine learning model performs. The train set is used to fit the model, the statistics of the train set are known. The second set is called the test data set, this set is solely used for predictions.


```python
print("total of articles: " + str(len(df)))
X = df['content'] # data
y = df['category'] # what we will predict
X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(features, 
                                                                               labels, 
                                                                               df.index, test_size=0.3, 
                                                                               random_state=1)

print("Length of train set: " + str(len(X_train)))
print("Length of test set: " + str(len(X_test)))
```

    total of articles: 15055
    Length of train set: 10538
    Length of test set: 4517
    

## Machine Learning Models:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Gaussian Naive Bayes
4. Linear SVC


```python
perform_list = [ ]

def run_model(model_name, est_c, est_pnlty):
    
    mdl=''
    if model_name == 'Logistic Regression':
        mdl = LogisticRegression()
    elif model_name == 'Multinomial Naive Bayes':
        mdl = MultinomialNB(alpha=1.0,fit_prior=True)
    elif model_name == 'Gaussian Naive Bayes':
        mdl = GaussianNB()
    elif model_name == 'Linear SVC':
        mdl = LinearSVC()
    
    oneVsRest = OneVsRestClassifier(mdl)
    
    #tf_idf
    oneVsRest.fit(X_train, y_train)
    y_pred = oneVsRest.predict(X_test)
    
    print(metrics.classification_report(y_test, y_pred))

    # Performance metrics
    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    
    # Get precision, recall, f1 scores
    precision, recall, f1score, support = score(y_test, y_pred, average='micro')

    print(f'Test Accuracy Score of Basic {model_name}: % {accuracy}')
    print(f'Precision : {precision}')
    print(f'Recall    : {recall}')
    print(f'F1-score   : {f1score}')
    print("")
    
    # Add performance parameters to list
    perform_list.append(dict([
        ('Model', model_name),
        ('Test Accuracy', round(accuracy, 2)),
        ('Precision', round(precision, 2)),
        ('Recall', round(recall, 2)),
        ('F1', round(f1score, 2))
         ]))
```

### Multinomial Naive Bayes


```python
run_model('Multinomial Naive Bayes', est_c=None, est_pnlty=None)
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
    
    

### Logistic Regression 


```python
run_model('Logistic Regression', est_c=None, est_pnlty=None)
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
    
    

### Gaussian Naive Bayes


```python
run_model('Gaussian Naive Bayes', est_c=None, est_pnlty=None)
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
    
    

### SVC


```python
run_model('Linear SVC', est_c=None, est_pnlty=None)
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
    
    

#### Results:


```python
model_performance = pd.DataFrame(data=perform_list)
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
plt.figure(figsize=(18,6))
sns.boxplot(x='Model', y='Test Accuracy', 
            data=model_performance, 
            color='lightblue', 
            showmeans=True)
plt.title("Accuracy \n", size=16);
```


    
![png](output_87_0.png)
    


## Best model to perform accuracy score


```python
max_accuracy = model_performance["Test Accuracy"].max()
model_index = model_performance.index[model_performance["Test Accuracy"] == max_accuracy].values

model_name = model_performance['Model'][model_index].values
print("The best accuracy of model is", max_accuracy, "from:", model_name)
```

    The best accuracy of model is 86.07 from: ['Linear SVC']
    

## Worst model to perform accuracy score


```python
min_accuracy = model_performance["Test Accuracy"].min()
worst_model_index = model_performance.index[model_performance["Test Accuracy"] == min_accuracy].values

worst_model_name = model_performance['Model'][worst_model_index].values
print("The best accuracy of model is", min_accuracy, "from:", worst_model_name)
```

    The best accuracy of model is 49.52 from: ['Gaussian Naive Bayes']
    

# Insert your data and we will tell you it category!
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
mdl=''
if model_name == 'Logistic Regression':
    mdl = LogisticRegression()
elif model_name == 'Multinomial Naive Bayes':
    mdl = MultinomialNB(alpha=1.0,fit_prior=True)
elif model_name == 'Gaussian Naive Bayes':
    mdl = GaussianNB()
elif model_name == 'Linear SVC':
    mdl = LinearSVC()

best_classifier = mdl.fit(X_train, y_train)
best_classifier

data.append(text_to_predict)
predict_new_data = tfidf_vec.transform(data)
result = best_classifier.predict(predict_new_data)

print("Your data is about: ")
print(result)
```

    Your data is about: 
    ['politics']
    

### Confusion Matrix 
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
create_confusion_matrix(best_classifier)
```


    
![png](output_97_0.png)
    


### The Worst Classifier


```python
mdl=''
if model_name == 'Logistic Regression':
    mdl = LogisticRegression()
elif model_name == 'Multinomial Naive Bayes':
    mdl = MultinomialNB(alpha=1.0,fit_prior=True)
elif model_name == 'Gaussian Naive Bayes':
    mdl = GaussianNB()
elif model_name == 'Linear SVC':
    mdl = LinearSVC()


worst_classifier = mdl.fit(X_train, y_train)
worst_classifier
```




    LinearSVC()




```python
create_confusion_matrix(worst_classifier)
```


    
![png](output_100_0.png)
    


# <p style="color:blue">Conclusion</p>


Linear SVC will bring us the most accurate results!

### Credits:

https://pandas.pydata.org/docs/index.html
https://monkeylearn.com/text-classification/


```python

```
