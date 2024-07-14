#!/usr/bin/env python
# coding: utf-8

# # MSDS Marketing Text Analytics, Unit 2, Assignment 2: Build a topic model

# ## ⚡️ Make a Copy
# 
# Save a copy of this notebook in your Google Drive before continuing. Be sure to edit your own copy, not the original notebook.

# In this assignment, you will implement a topic model preprocessor which can then be applied to the task of topic-modeling Amazon text reviews. Please review the course lectures and documentation up to this point before continuing. Be sure also to be familiar with the [documentation for TMToolkit](https://tmtoolkit.readthedocs.io/en/latest/topic_modeling.html)
# 
# Be sure to make a copy into your own Drive account before editing this notebook.
# 
# You will implement a preprocessing function to prepare your corpus for topic modeling. It is recommended that you use a small test corpus (an example is provided below) for development, rather than starting with the full review set.
# 
# ---
# 
# ## ⚠️ Important Note
# 
# This notebook has been updated to reflect **significant changes to the tmtoolkit API**. You will find some differences from other course materials, including the lectures. There is no longer a TMPreproc object in tmtoolkit. Rather, preprocessing functions have been moved into the tmtoolkit.corpus module. All of the preprocessing functions you will need for this assignment are imported for you below.
# 
# ---

# ## Dependency installs
# 
# **Important:** You will likely see a message to restart the runtime after the installations are complete, and should do so.

# ### Remove some libraries in Colab that cause conflicts

# In[1]:


# !pip uninstall -y numba
# !pip uninstall -y tensorflow


# ### Install LDA and tmtoolkit

# In[2]:


# !pip install lda
# !pip install "tmtoolkit[recommended]"


# ## Imports
# 

# In[3]:


# This is just here for documentation purposes: tmtoolkit no longer has
# a TMPreproc. Don't use it for this assignment! Instead use the functions
# that are imported below.

# Do not do this:
#from tmtoolkit.preprocess import TMPreproc


# In[4]:


from tmtoolkit.corpus import Corpus, lemmatize, to_lowercase, remove_chars, filter_clean_tokens
from tmtoolkit.corpus import corpus_num_tokens, corpus_tokens_flattened
from tmtoolkit.corpus import dtm
from tmtoolkit.corpus import vocabulary
from tmtoolkit.topicmod.model_io import print_ldamodel_topic_words
from tmtoolkit.topicmod.tm_lda import compute_models_parallel


# ### Import punctuation from the string module
# 
# You will use this in your `build_corpus` function to remove punctuation from the corpus.

# In[5]:


from string import punctuation


# ## ⚒️ Implement a pre-processor
# 
# Here you will implement a function called `build_corpus` which returns a Corpus object to be used for topic modeling.
# 
# The build_corpus function will take a list of texts and return a pre-processed Corpus object. Preprocessing should include the following actions on the corpus using the appropriate functions imported from the corpus module.
# 
#  - lemmatize the texts
#  - convert tokens to lowercase
#  - remove punctuation
#  - clean tokens to remove numbers and any tokens shorter than 3 characters
# 
# The first part of the function to create the corpus object is done for you. Your job is to call the specific preprocessing functions on the corpus and to return corpus object.
# 

# ---
# 
# ### 💡 Note
# 
# Loading a corpus as a list of strings is not the only way to use tmtoolkit. Given, for example, a large corpus that might not fit in memory, the current approach would not work well. See the tmtoolkit docs on [working with text corpora](https://tmtoolkit.readthedocs.io/en/latest/text_corpora.html) for more info.
# 
# ---

# In[6]:


def build_corpus(texts, lang="en"):
    """Corpus builder which returns a Corpus object processed on texts as language
    specified by lang (defaults to "en"):

    Should perform all of the following pre-processing functions:

     - Lemmatize the tokens
     - Convert tokens to lowercase
     - Remove punctuation
     - Remove numbers
     - Remove tokens shorter than 2 characters
    """
    # Here, we just use the index of the text as the label for the corpus item
    corpus = Corpus({ i:r for i, r in enumerate(texts) }, language=lang)

    # Lemmatize the tokens
    lemmatize(corpus)
    
    # Convert tokens to lowercase
    to_lowercase(corpus)

    # Remove punctuation (using correct parameter)
    remove_chars(corpus, chars=punctuation)

    # Clean tokens to remove numbers and any tokens shorter than 3 characters
    filter_clean_tokens(corpus, remove_punct=True, remove_stopwords=False, remove_empty=True, remove_shorter_than=3)

    return corpus


# In[7]:


#~~ /autograde # do not delete this cell


# ---
# ### ⚠️  **Caution:** No arbitrary code above this line
# 
# The only code written above should be the implementation of your graded function. For experimentation and testing, only add code below.
# ___

# ## Function development
# 
# Use this section of code to verify your function implementation. You may change the test_corpus as needed to verify your implementation. The grader will be checking that your function returns a Corpus object that meets all of the following critera:
# 
#  - tokens are lemmatized
#  - tokens are converted to lowercase
#  - special characters are removed from tokens
#  - tokens shorter than 3 characters and numerics are removed

# In[ ]:


import spacy
spacy.load("en_core_web_sm")

example_docs = [ # Feel free to edit this corpus for further testing
                # to be sure that your functions meet specifications.
    "The 3 cats sat on the mats!",
    "1 fish 2 fish Red fish Blue fish",
    "She sells $ea$shells"
]
example_corpus = build_corpus(example_docs)
corpus_tokens_flattened(example_corpus)


# In[ ]:


dtms = {
    "test_corpus": dtm(example_corpus)
}
lda_params = {
    'n_topics': 2,
    'eta': .01,
    'n_iter': 10,
    'random_state': 1234,  # to make results reproducible
    'alpha': 1/16
}

models = compute_models_parallel(dtms, constant_parameters=lda_params)


# In[ ]:


model = models["test_corpus"][0][1]
print_ldamodel_topic_words(model.topic_word_, vocabulary(example_corpus), top_n=5)


# ### Assignment submission
# 
# After completing the `build_corpus` implementation, download your notebook as a .py file (File > Download > Download .py) and submit the downloaded file for grading.

# ## Topic modeling Amazon Reviews
# 
# Once you have completed the assignment above, you will be well prepared to start your final project for this unit. The project will include loading Amazon reviews into a corpus for topic modeling. The code below demonstrates topic modeling the reviews for a given brand. Note that the final project will require additional segmentation of the data, which is not done for you in the example here.

# In[ ]:


import gzip
import itertools
import json

asins = []

# To run this code, you will need to download the metadata file from the course
# assets and upload it to your Google Drive. See the notes about that file
# regarding how it was processed from the original file into json-l format.

META_FILE = "meta_Clothing_Shoes_and_Jewelry.jsonl.gz"

with gzip.open(META_FILE) as products:
    for product in products:
        data = json.loads(product)
        categories = [c.lower() for c in
                      list(itertools.chain(*data.get("categories", [])))]
        if "nike" in categories:
            asins.append(data["asin"])


# ### Inspect the first fews ASINs

# In[ ]:


asins[:3]


# ### Check the length, i.e. the number of resulting ASINs

# In[ ]:


len(asins)


# ### Compile a selection of review texts
# 
# 🔥 **Note:** This code as-is uses a reduced version of the reviews file. You may alternatively use the full version of the reviews file but expect it to take a **long time** to process this code.
# 
# As with the meta file above, whichever file you use here should be uploaded to the root of your Google Drive.

# In[ ]:


REDUCED_REVIEWS_FILE = "drive/MyDrive/reviews_Clothing_Shoes_and_Jewelry_5.json.gz"
FULL_REVIEWS_FILE = "drive/MyDrive/reviews_Clothing_Shoes_and_Jewelry.json.gz"

reviews = []
with gzip.open(REDUCED_REVIEWS_FILE) as f:
    for review in f:
        data = json.loads(review)
        if data["asin"] in asins: # This is where we check to see if it is a Nike ASIN
            text = data["reviewText"]
            reviews.append(text)


# ### Inspect a few of the reviews

# In[ ]:


for i, review in enumerate(reviews[:5]):
    print(i, review[:80])


# ### Build a Corpus object from the review texts

# In[ ]:


reviews_corpus = build_corpus(reviews)


# In[ ]:


dtms = {
    "reviews_corpus": dtm(reviews_corpus)
}
lda_params = {
    'n_topics': 10,
    'eta': .01,
    'n_iter': 10,
    'random_state': 1234,  # to make results reproducible
    'alpha': 1/16
}

models = compute_models_parallel(dtms, constant_parameters=lda_params)


# ### Print the topics

# In[ ]:


model = models["reviews_corpus"][0][1]
print_ldamodel_topic_words(model.topic_word_, vocabulary(reviews_corpus), top_n=5)


# ## 💾 Save your topic model and review texts for use in Lab 2

# Once you have completed the above assignment, run the following code to save your topic model and your review texts to your Google Drive. You will load this model and use it for document classification in Lab 2.

# In[ ]:


import pickle
from tmtoolkit.topicmod.model_io import save_ldamodel_to_pickle

with open("drive/MyDrive/MSDS_HW2_model.p", "wb") as modelfile:
    save_ldamodel_to_pickle(modelfile, model, vocabulary(reviews_corpus), reviews_corpus.doc_labels, dtm=dtm(reviews_corpus))


# In[ ]:


with open("drive/MyDrive/MSDS_HW2_corpus.p", "wb") as reviewsfile:
    pickle.dump(reviews, reviewsfile)

