#!/usr/bin/env python
# coding: utf-8

# ## Getting the dataset

# In[1]:


get_ipython().system(' kaggle datasets download -d kazanova/sentiment140')


# In[3]:


# specifying the path of kaggle.json file
get_ipython().system('mkdir -p ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle/')
get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')


# In[5]:


# extracting the compresssed dataset
from zipfile import ZipFile
dataset = '~/sentiment140.zip'

with ZipFile(dataset, 'r') as zip:
    zip.extractall()
    print('the dataset is extracted')


# ## Importing the dependencies

# In[13]:


import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[14]:


import nltk
nltk.download('stopwords')


# In[15]:


#printing the stopwords in english
print(stopwords.words('english')) #stopwords are essentially the words which don't add significant meaning to your analysis, as sentiment can not be gazed from it alone.


# ## Data Processing

# In[16]:


# Loading the data from csv file to pandas dataframe
twitter_data = pd.read_csv('/Users/abdullahraashid/Documents/PythonML/TwitterSentimentAnalysis/training.1600000.processed.noemoticon.csv', encoding = 'ISO-8859-1')


# In[17]:


# checking rows and columns
twitter_data.shape # 1.6 million tweets are rows, the 6 fields are the columnds, target, ids, date and all


# In[19]:


# print the first five rows
twitter_data.head() # Now it is accidentally assuming that the first tweet is the column names


# In[20]:


# naming the cloumns and reading the dataset again
column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
twitter_data = pd.read_csv('/Users/abdullahraashid/Documents/PythonML/TwitterSentimentAnalysis/training.1600000.processed.noemoticon.csv', names = column_names, encoding = 'ISO-8859-1')
twitter_data.head()


# In[21]:


# Number of missing values in the dataset
twitter_data.isnull().sum()


# In[22]:


# checking the districution of target column, how many positive, neutral, and negative tweets are present
twitter_data['target'].value_counts()


# **Convert the target '4' to '1'**

# In[25]:


twitter_data.replace({'target':{4:1}},inplace=True)


# In[26]:


twitter_data['target'].value_counts()

0 = negative
1 = positive
# **Stemming**: process of reducing a word to its Root word.
# Example: actor, actress, acting = act

# In[27]:


port_stem = PorterStemmer() # loads the instance


# In[28]:


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ', content) #removes everything except letters since sentiment analysis
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split() #split everything and add it to a list, so if a tweet has 10 words those 10 individual words will be added to the list 
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')] # if the word is not in stopwords, it will be stemmed
    stemmed_content = ' '.join(stemmed_content) # join everything 
    return stemmed_content


# In[29]:


twitter_data['stemmed_text'] = twitter_data['text'].apply(stemming) # 50 minutes to complete the stemming


# In[30]:


twitter_data.head()


# In[32]:


print(twitter_data['stemmed_text'])


# In[33]:


print(twitter_data['target'])


# In[34]:


# seperating the data(stemmed_text) and label(target)
X = twitter_data['stemmed_text'].values
y = twitter_data['target'].values


# In[35]:


print(X)


# **Splitting the data to training and testing data**

# In[37]:


# creating 4 arrays
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,stratify=y, random_state = 5) 


# In[38]:


print(X.shape, X_train.shape, X_test.shape)


# In[39]:


## Converting the textual data to numerical data: feature extraction, vectorizer

vectorizer = TfidfVectorizer()

X_train=vectorizer.fit_transform(X_train) # assigns value to words that have been repeated and ML model tries to correspond that word to either zero or 1
X_test=vectorizer.transform(X_test) # so you don't fit the test data as you need it to test your model


# In[40]:


print(X_train) #all the words get some importance value


# In[41]:


print(X_test)


# ## Training the Logistic Regression model

# In[43]:


model = LogisticRegression(max_iter=1000)


# In[44]:


model.fit(X_train,y_train) # maps to the desired targets


# ## Model Evaluation

# **Accuracy Score**

# In[45]:


# Accuracy score on the training data
X_train_pred = model.predict(X_train)
training_data_accuracy=accuracy_score(y_train,X_train_pred)


# In[46]:


print('accuracy score on the training data = ', training_data_accuracy) # 0.78068828125, 78% accuracy


# In[47]:


# Accuracy score on the training data
X_test_pred = model.predict(X_test)
training_data_accuracy=accuracy_score(y_test,X_test_pred)
print('accuracy score on the test data = ', training_data_accuracy) # 0.772940625, 77%, so model is performing well, and no overfitting has taken place


# Model accuracy = 77.8%

# ## Saving the trained model

# In[50]:


import pickle


# In[51]:


filename = 'twitter_sentiment.sav'
pickle.dump(model,open(filename,'wb')) # dumps/writes the model in the filename 


# ## Using the saved model for future predictions

# In[54]:


# loading the saved model
loaded_model = pickle.load(open('~/twitter_sentiment.sav','rb'))


# In[56]:


X_new = X_test[45]
print(y_test[45])

prediction = loaded_model.predict(X_new)
print(prediction)

if (prediction[0] ==0):
    print('Negative Tweet')
else:
    print('Positive Tweet')

