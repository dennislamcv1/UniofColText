#!/usr/bin/env python
# coding: utf-8

# # MSDS Marketing Text Analytics, Assignment 1: Segmenting Amazon product reviews by sentiment

# ## ⚡️ Make a Copy
# 
# Save a copy of this notebook in your Google Drive before continuing. Be sure to edit your own copy, not the original notebook.

# Topic models are only as good as the data you put into them. If we were Nike, and we worked a team that was designed to increase the quality of our products, it might make sense to only look at review that expressed negative sentiment.
# 
# For this assignment, you will use a pretrained sentiment analysis model to implement a function capable of filtering a list of texts for negative sentiment. This function can be used in turn to extract a set of negative reviews.
# 
# For information about how to calculate text sentiment, see the [TextBlob documentation](https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis)

# ## Imports

# In[2]:


import gzip
import itertools
import json
from textblob import TextBlob


# ## Implement get_negative_texts

# In[3]:


def get_negative_texts(texts):
    """Implement this function which should take a list of texts
    and returns a list of the texts that are determined to be
    of negative sentiment.

    See the TextBlob documentation for how to evaluate sentiment. For our
    purposes here, negative sentiment is a sentiment with polarity < 0.0.
    """
    negative_texts = []
    for text in texts:
        blob = TextBlob(text)
        if blob.sentiment.polarity < 0.0:
            negative_texts.append(text)
    return negative_texts


# In[4]:


#~~ /autograde # do not delete this code cell


# ---
# ### ⚠️  **Caution:** No arbitrary code above this line
# 
# The only code written above should be the implementation of your graded
# function. For experimentation and testing, only add code below.
# 
# ---

# ## **⚡️ Important:** Beware of sentiment fuzziness
# 
# Like most machine-learning approaches, TextBlob's sentiment analysis is probabilistic -- results will sometimes not match your expectations. Keep that in mind if you edit the texts below. Your assignment will be tested against actual results from TextBlob's sentiment polarities, so be sure to use that specific approach for this assignment. You are free to experiment with other approaches in your peer-reviewed unit project.

# In[5]:


texts = [ # you may edit this list for further exploration and testing
    "We all love apple pie",
    "We hate aparagus",
    "Rainbows are beautiful",
    "Landfills are ugly"
]
get_negative_texts(texts)


# ## **💡 Tip:** Python generators for advanced implementation
# 
# When processing large amounts of data, it can be problematic to simply pass around lists of things, which can eat up system resources like memory.
# 
# 🐍 Python's solution to this problem is the concept of a generator. A generator "yields" its elements one-by-one, rather than returning them all as a single data structure. For example, consider the following simple function:
# 
# ```
# def get_list():
#     return [1, 2, 3]
# ```
# 
# The generator version of this function would look like this:
# 
# ```
# def get_list():
#     for x in [1, 2, 3]:
#         yield x
# ```
# 
# The result can be iterated as expected:
# 
# ```
# for i in get_list():
#     print(i)
# ```
# 
# For purposes of this assignment and for the Unit 2 project, you may use either a list-based solution, or a generator as you see fit. For more information about generators, see the [Python wiki documentation on generators](https://wiki.python.org/moin/Generators) or this [generator tutorial from RealPython](https://realpython.com/introduction-to-python-generators/)
# 
# 

# ## Applying segmentation to Amazon reviews

# The code below demonstrates segmentation of reviews by negative sentiment by first aggregating the Nike reviews, then calling get_negative_texts on the list of review texts. If you have implemented your function correctly, and have uploaded the necessary data files to your Drive account, this code should just work.

# ## Mount Google Drive

# In[6]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[7]:


asins = []

# To run this code, you will need to download the metadata file from the course
# assets and upload it to your Google Drive. See the notes about that file
# regarding how it was processed from the original file into json-l format.

with gzip.open("meta_Clothing_Shoes_and_Jewelry.jsonl.gz") as products:
    for product in products:
        data = json.loads(product)
        categories = [c.lower() for c in
                      list(itertools.chain(*data.get("categories", [])))]
        if "nike" in categories:
            asins.append(data["asin"])


# In[8]:


asins[:3]


# In[9]:


len(asins)


# In[ ]:


# To run this code, you will need to download the reviews file from the course
# assets and upload it to your Google Drive. Unlike the metadata above, this
# file was originally provided as json-l, and is json-l despite the .json
# file name.

all_texts = []
with gzip.open("reviews_Clothing_Shoes_and_Jewelry.json.gz") as reviews:
    for review in reviews:
        data = json.loads(review)
        if data["asin"] in asins:
            text = data["reviewText"]
            all_texts.append(text)


# In[ ]:


for i, text in enumerate(all_texts[:5]):
    print(i, text[:80])


# In[ ]:


negative_texts = get_negative_texts(all_texts)


# In[ ]:


for i, text in enumerate(negative_texts[:5]):
    print(i, text)


# In[ ]:


len(negative_texts)


# In[ ]:


print(negative_texts)


# ## Moving forward
# 
# 

# ### About segmentation
# 
# Between the Sales Rank segmentation demo notebook, and this sentiment segmentation assigment, you should have a good sense at this point of the idea of segmentation, and how you might approach it in this data set. You will make use of this knowledge in the unit project -- your peers will be checking to see that you segmented your product data in an interesting way as part of the analysis for the project.

# ### Moving on to topic modeling
# 
# Before tackling the project, be sure to do the topic modeling assignment which is the next and final component in preparation for the unit project.

# In[ ]:




