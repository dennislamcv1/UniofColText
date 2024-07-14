#!/usr/bin/env python
# coding: utf-8

# # MSDS Network Analysis, Homework #2: Tweet tokenization and part-of-speech tagging

# ## ⚡️ Make a Copy
# 
# Save a copy of this notebook in your Google Drive before continuing. Be sure to edit your own copy, not the original notebook.

# ### 📝 Completing the assignment

# There is are two function definitions below for you to complete: `unique_tokens`, and `token_counts`. Write the code to meet the function specifications, and submit the .py version of this notebook to the grader.
# 
# > Indented block
# 
# 

# > **⚠️  Don't code outside the lines.** Keep your function implementation code inside the function blocks. Be sure not to write any code above the `/autorade` delimiter other than the specified function code. Any experimentation or testing code should go below the `/autograde` indicator, and will be ignored by the grader.

# ## 📁 Getting the data file

# The code below makes use of a Twitter dataset harvested from the Twitter API and saved in a Gzipped JSON-L file. This is the same file used in the previous assignment.
# 
# You should have previously downloaded the multi-brand tweet file and uploaded it to your Drive. You will use that file again here.

# ## Mount Google Drive

# In[1]:


# from google.colab import drive
# drive.mount('/content/drive')


# ## Setup

# ### Imports

# In[2]:


import gzip
import json
import nltk


# ### Downloads

# The NLTK is a corpus-linguistics oriented toolkit, and including all of its resources in a standard install would be too heavy. Instead, you download what you need for the problem at hand. The utilities you will be using in this assignment require the availability of the punkt and averaged_perceptron_tagger packages.

# In[3]:


nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")


# ## Data Prep

# When using natural language processing (NLP) techniques with any specialized textual dataset, such as social media texts, one should consider some of the specifics of text usage in that data and how those specifics might affect an analysis.

# ### Tokenizing Tweet texts

# For Tweets, some of the specifics that come immediately to mind are:
# 
#  * "At" mentions: @username
#  * Hashtags: #hashtag
#  * URLs: https://...
# 
#  Let's take a look at how a standard tokenizer might parse these things:

# In[4]:


example_tweet = "hope I get a new pair of these @Nike shoes!!!! #nikelife https://www.nike.com/launch/t/womens-air-force-1-reveal-pastel-reveal"


# In[5]:


nltk.tokenize.word_tokenize(example_tweet)


# Notice how our three specific language usages of concern are treated here by a tokenizer that is unaware of the concepts of at-mentions, hashtags, and links. This might not be ideal for our analysis, and we might something that does have an awareness of these things. To this end, NLTK provides us with a TweetTokenizer:

# In[6]:


nltk.TweetTokenizer().tokenize(example_tweet)


# 🎉 That's better!

# Below, you will make use of the TweetTokenizer to implement a function that collects the unique tokens in an entire dataset of tweets. We'll write the function in a way that you could pass in an alternative tokenizer if needed.

# ## Implement unique_tokens

# Recall from Homework #1 the approach that we took to processing a dataset of Tweets that could be either dictionaries or JSON strings. This approach enabled us to have a function that could take either a list of tweet objects or a file handle for a JSON-L file of tweets.

# ### The unique_tokens function

# Here you will implement a similar function. This function, instead of returning filtered tweets, will return the set of unique tokens found in the texts of the tweets.
# 
# The function should default to using the NLTK TweetTokenizer to tokenize the texts, but alternative callables could be passed into the function. This is setup for you by the keyword argument `tokenizer`. Note that this is set to the callable tokenize method, not to the tokenizer object.
# 
# Recall from the [Twitter object model documentation](https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/object-model/tweet)
# 
# Note that the text field of a Tweet can be called either `text` or `full_text` and you should implement the function to handle either one.
# 
# > **⚠️  Important Note:** The Tweets in the provided branded data file were harvested with the ["extended Tweet option in v1.1](https://developer.twitter.com/en/docs/twitter-api/premium/data-dictionary/overview) of the Twitter API, and so have a `full_text` field rather than a `text` field. You should implement the function to first check for a `full_text` field and fallback to `text` field if full_text does not exist.

# In[7]:


from nltk.tokenize import TweetTokenizer

def unique_tokens(tweets, tokenizer=nltk.TweetTokenizer().tokenize):
    """Tokenizes the texts of an iterable of Tweet objects.
    Accepts an iterable of either tweet dictionary objects
    or tweet objects encoded as JSON text.

    For backwards compatibility in the Twitter API, the function
    should support Tweets with either a `full_text` field or
    a `text` field, defaulting to `full_text` if it exists.

    Returns the set of unique tokens.
    """
    
    # TODO: Implement this function to return the set of
    # unique tokens parsed from the tweet texts by the
    # NLTK TweetTokenizer
    unique_tokens_set = set()

    for tweet in tweets:
        if isinstance(tweet, bytes):
            tweet = tweet.decode('utf-8')
        if isinstance(tweet, str):
            tweet = json.loads(tweet)
        
        text = tweet.get('full_text') or tweet.get('text')
        if text:
            tokens = tokenizer(text)
            unique_tokens_set.update(tokens)
    
    return unique_tokens_set


# Before continuing onto the next function, try some of the unique_tokens testing code after the autograde cutoff below and be sure the function works as expected.

# ## Token counting and parts of speech

# You'll note that the count of unique tokens in our multi-brand Tweet data file is over 150 thousand.
# 
# In a large analysis project with a lot of data, it can be useful to think of ways to reduce the data being analyized. In natural language processing (NLP) jobs such as the analysis of free-form social media text, we might want to consider taking part-of-speech into account to determine the value a term might bring to the analysis.
# 
# It can also be useful to count up token instances to get a sense of word usage in the dataset. Here you will implement a function that does both of these things, i.e., you will count up tokens that match a filtering set of part-of-speech tags.

# > 💡 The NLTK Perceptron Tagger uses the part-of-speech tags defined by the Penn Treebank project, which are documented [here](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html). If you are curious about the inner workings of the PerceptronTagger algorithm, take a look at [this article](https://explosion.ai/blog/part-of-speech-pos-tagger-in-python) by its creator, Matthew Honnibal.

# In[8]:


import json
import nltk
from nltk.tokenize import TweetTokenizer
from collections import Counter


def token_counts(tweets,
                 tagger=nltk.tag.PerceptronTagger().tag,
                 tokenizer=nltk.TweetTokenizer().tokenize,
                 parts_of_speech=None):
    """
    Tokenizes the texts of an iterable of Tweet objects and counts the
    tokens that match the provided parts of speech.

    Args:
    - tweets: An iterable of either tweet dictionary objects or tweet objects encoded as JSON text.
    - tagger: A callable for tagging tokens with their parts of speech. Defaults to the NLTK PerceptronTagger.
    - tokenizer: A callable for tokenizing tweet texts. Defaults to the NLTK TweetTokenizer.
    - parts_of_speech: A set of parts of speech tags to filter tokens by. If None, all tokens are counted.

    Returns:
    - A Counter object with the counts of the filtered tokens.
    """
    token_counter = Counter()

    for tweet in tweets:
        if isinstance(tweet, bytes):
            tweet = tweet.decode('utf-8')
        if isinstance(tweet, str):
            tweet = json.loads(tweet)
        
        text = tweet.get('full_text') or tweet.get('text')
        if text:
            tokens = tokenizer(text)
            tagged_tokens = tagger(tokens)
            
            for token, pos in tagged_tokens:
                if parts_of_speech is None or pos in parts_of_speech:
                    token_counter[token] += 1
    
    return token_counter


# After implementing the functions above and testing them out in the space below, submit your code to the grader:
# 
#  * Download the .py file (File > Download > Download .py)
#  * Upload the file to the Coursera grader for assessment.

# In[9]:


#~~ /autograde # do not delete this code cell


# ---
# ### ⚠️  **Caution:** No arbitrary code above this line
# 
# The only code written above should be the implementation of your graded
# function. For experimentation and testing, only add code below.
# 
# ---

# ## Testing out unique_tokens

# ### A simple test of unique_tokens

# Here is a simple test of your unique_tokens function. For a bigger test, see the file loading test below, but keep in mind that the Tweets in the file only have `full_text`, not `text`.

# In[10]:


test_tweets = [ # Minimal tweet constructs for testing purposes.
                # You may want to add your own for further diligence.
    { "text": "The sky is blue."},
    { "full_text": "The grass is green."},
]
unique_tokens(test_tweets)


# ### A bigger test of unique_tokens

# In[11]:


# Note: these tweets have a `full_text` field not a `text` field.
# Your function should support both.
with gzip.open("nikelululemonadidas_tweets.jsonl.gz") as f:
    tokens = unique_tokens(f)


# ### The length of tokens is the count of unique tokens in the Tweet texts

# In[12]:


len(tokens)


# If your function is correct, you should see a count of over 150k here. That's a lot of unique tokens. Let's take a look at a few:

# ### Inspect some of the tokens

# In[13]:


list(tokens)[:50]


# So, it looks like a good number of these are http links, but even if half of our tokens were links, we still have a lot of unique tokens here. In a project with a lot of data, it may be prudent to think of ways to cull the data that is appropriate for the analysis.
# 
# Below we will consider the idea of targeting salient terms by identifying the parts of speech for the tokens.

# The NLP processing for determining parts of speech can be a bit resource intensive. So in the test code below, there is a truncated version of tweet loading that will load only 100 tweets.
# 
# The full-file version of the test code is commented out so you do not accedentally run it. Uncomment it to check your numbers against those mentioned below. It will take a few minutes to process the whole file.

# **Reminder:** See the [Penn Treebank POS list](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html) for possible part-of-speech tags.

# ### A truncated test that only loads 100 tweets

# In[14]:


tweets = []
with gzip.open("nikelululemonadidas_tweets.jsonl.gz") as f:
    for i, line in enumerate(f):
        if i >= 100:
            break
        tweets.append(json.loads(line))
counts = token_counts(tweets, parts_of_speech=["VB"])


# In[15]:


counts


# ### A full test that will take some time to run

# This will take a few minutes to run. Uncomment the code to execute it.

# In[ ]:


#with gzip.open("drive/MyDrive/nikelululemonadidas_tweets.jsonl.gz") as f:
#    counts = token_counts(f, parts_of_speech=["VB"])


# There should be something over 6500 VB matches ..

# In[ ]:


len(counts.items())


# .. with something close to 133000 as the sum of all counts:

# In[ ]:


sum(counts.values())


# Inpect a few items

# In[ ]:


dict(list(counts.items())[:30])

