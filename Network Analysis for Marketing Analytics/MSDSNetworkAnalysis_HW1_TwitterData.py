#!/usr/bin/env python
# coding: utf-8

# # MSDS Network Analysis, Homework 1: Twitter data wrangling

# ## ⚡️ Make a Copy
# 
# Save a copy of this notebook in your Google Drive before continuing. Be sure to edit your own copy, not the original notebook.

# ## 🏁 We are working toward a goal: the final project

# Please take a moment to review the requirements for the upcoming final project for this course. The homework assignments and coding labs are designed to step you toward the goal of preparing to complete your final project.

# ## Working with Twitter data

# Social media data tends to have a lot of metadata. When doing an analysis, you can take advantage of the richness of this data toward the end of slicing it up to meet the needs for your project.

# In this assignment, you will practice working with Twitter API data by implementing a function for matching Tweets to a given set of metadata parameters. We'll then take a look at how you can use that function for filtering Tweets to a specific subset.
# 
# > 💡 Before continuing, take a moment to get familiar with the structure of the [Tweet object model](https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/object-model/tweet)

# ### 📝 Completing the assignment

# > **⚠️  Don't code outside the lines.** Keep your function implementation code inside the function blocks. Be sure not to write any code above the `/autorade` delimiter other than the specified function code. Any experimentation or testing code should go below the `/autograde` indicator, and will be ignored by the grader.

# ## Defining the match criteria

# Take a look at the Twitter documentation referenced above, and determine what fields you will need to complete this assignment. In this assignment, you will be implementing a function called `match_tweet` which determines if a Tweet matches the set of matching parameters:
# 
#  * **country_codes:** If provided and non-empty, is the Tweet's country code in this list of provided country codes?
#  * **start_date:** If provided, is the date of Tweet's `created_at` date greater than or equal to this date?
#  * **end_date:** If provided, is the date of the Tweet's `created_at` date less than or equal to this date?
# 
# 
# Let's consider further details about matching before jumping into the implementation.

# ## 🌎 Matching by country code

# The following criteria should be implemented in your function for matching to the country code:
# 
#  * If the country_code parameter is `None` or an empty list the function should match any Tweet regardless of country code.
#  * Country codes should be matched in a case-insensitive way. E.g. "US" is equivalent to "us". However, "US" and "USA" are not considered to be matches -- the case-insensitive match should be exact.
#  * If this parameter is provided and the Tweet's country code is not in the list of provided country codes, the function should return False regardless of the other match parameters.

# ## 📅 Matching by date

# The `start_date` and `end_date` parameters are used to determine if the Tweet was created within a specific date range. One of these parameters may be provide to match an open-ended range, or neither of them to match all Tweets.
# 
# Use the following criteria when matching by date:
# 
#  * The "date" of a Tweet is the date component of the `created_at` timestamp of the Tweet (see below for info about working with dates)
#  * start_date and end_date are both inclusive matches. E.g. a Tweet created on 2021-11-01 will match either a start_date or an end_date of 2021-11-01
#  * `None` for either date parameter means "no limit" in that direction. By this criterion, `None` for both parameters means: "match all Tweets"
#  * If date parameters are provided and the Tweet does not match, the function should return False regardless of other matching parameters.

# ### 🐍 Working with Tweet timestamps as dates

# The following code snippet can be used to parse a Tweet's created_at time into a datetime object:
# 
# ```
#     dt = datetime.strptime(tweet["created_at"], "%a %b %d %H:%M:%S +0000 %Y")
# ```
# 
# You can then get just a date object from that by calling the `.date()` method on the datetime object:
# 
# ```
#     dt = dt.date()
# ```

# ## ⚡️ Getting started

# You should now be ready to go. To complete the assignment:
# 
#  1. Complete the implementation of the `match_tweet` function.
# 
#     The function definition line is created for you below. Your job is to complete the function so that it works to specification.
# 
# 2. Write any exploratory and testing code only below the `/autograde` note.
# 
# 3. Download the completed notebook as a .py file:
# 
#     File > Download > Download .py
# 
#     ⚠️ The .ipynb file will not work with the grader. Be sure to download the .py file
# 
# 4. Submit the file to the Coursera grader for assessment.

# ## Imports

# In[1]:


import datetime
import gzip
import json


# ## Implement match_tweet

# In[2]:


def match_tweet(tweet, country_codes=None, start_date=None, end_date=None):
    """Return the boolean value of whether this tweet object matches
    the specified parameters.

    Supports the following match criteria:

      * country_codes (a list of case-insensitive country codes)
      * start_date (earliest date of matching tweets)
      * end_date (latest date of matching tweets)

    `None` values for any parameters, as well as an empty country_code list are
    interpreted as "match all" for the respective parameter.
    """
    def str_to_date(date_str):
        if isinstance(date_str, datetime.date):
            return date_str
        return datetime.datetime.strptime(date_str, '%Y-%m-%d').date() if date_str else None


    # Extract the date from the Tweet's created_at timestamp
    tweet_date = datetime.datetime.strptime(tweet["created_at"], "%a %b %d %H:%M:%S +0000 %Y").date()
    
    # Check country code
    if country_codes:
        tweet_country_code = tweet.get('place', {}).get('country_code', '').lower()
        country_codes_lower = [code.lower() for code in country_codes]
        if tweet_country_code not in country_codes_lower:
            return False

    # Check start date
    start_date_obj = str_to_date(start_date)
    if start_date_obj and tweet_date < start_date_obj:
        return False

    # Check end date
    end_date_obj = str_to_date(end_date)
    if end_date_obj and tweet_date > end_date_obj:
        return False

    return True


# In[3]:


#~~ /autograde # do not delete this code cell


# ---
# ### ⚠️  **Caution:** No arbitrary code above this line
# 
# The only code written above should be the implementation of your graded
# function. For experimentation and testing, only add code below.
# 
# ---

# ## Testing things out

# A test Tweet with minimal data, and a start and end date are created below for you to test with. You will almost certainly want to create some of your own values for further testing.

# In[4]:


test_tweet =  {
    "created_at": "Fri Nov 11 08:25:03 +0000 2021",
    "place": { "country_code": "US"}
}


# In[5]:


start = datetime.date.fromisoformat("2021-11-01")
end = datetime.date.fromisoformat("2021-11-30")


# In[6]:


match_tweet(test_tweet, country_codes=["US"], start_date=start, end_date=end)


# ## Making use of the Tweet matcher: Filtering a set of Tweets

# The code below shows how you might make use of the matching function for filtering a list or a data file of Tweets.

# The filter_tweets function uses your match_tweet implementation to filter an iterable of tweets using the provided matching parameters.
# 
# `filter_tweets` works with an iterable of either JSON strings, or dict objects, so you can use it to parse tweets out of a JSON-L file, but also use it to re-filter the Tweets it returns. We'll see how that works below.

# In[ ]:


def filter_tweets(tweets, country_codes=None, start_date=None, end_date=None):
    """Returns a list of tweets filtered by provided filter criteria.

    Currently only supports country_codes filtering. See match_tweet
    for details about how match based filtering works.

    This function accepts an iterable of tweet objects which may either
    be JSON strings, or previously parsed tweet dictionaries.

    Yields an iterable of dictionaries.
    """
    for tweet in tweets:
        if isinstance(tweet, (bytes, str)):
            tweet = json.loads(tweet)
        if match_tweet(tweet,
                       country_codes=country_codes,
                       start_date=start_date,
                       end_date=end_date):
            yield tweet


# > 🐍 **About yield:** "Yielding" instead of "returning" from a function allows a Python function to iteratively turn out results as it processes them, rather than collecting all the results and returning them at once. This is called a **generator** and is handy particularly when processing large amounts of data. While we convert all the results to lists below for simple testing purposes, in a real application you would probably take advantage of the generator, and process the filtered Tweets as they are yielded by the function.

# ## 📁 Getting the data file

# The code below makes use of a Twitter dataset harvested from the Twitter API and saved in a Gzipped JSON-L file. This is the same file used in the labs and final project for the course.
# 
# > **💡  JSON-L.** JSON-L is an unofficial yet common format of one JSON document per line in the file. It is typically read by parsing each line of the file as a JSON string.
# 
# ---
# 
# Before continuing, you will need to:
# 
#  * Download the file from the course resources
#  * Upload it to the root of your Google Drive account

# ## Mount Google Drive

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# ## Testing out the filter

# In[ ]:


with gzip.open('drive/MyDrive/nikelululemonadidas_tweets.jsonl.gz') as f:
    US_tweets = list(filter_tweets(f, country_codes=["us"]))


# ## Check the count
# 
# If you have implemented your matching function correctly, the count of US tweets should be **5713**:

# In[ ]:


len(US_tweets)


# ## Refiltering
# 
# Now, make another pass on the tweets, this time narrowing the selection down by date. You could have passed in all of the filter parameters above, but sometimes iterative filtering like this is useful during a data exploration or analysis.

# In[ ]:


nov_usa_tweets = list(filter_tweets(US_tweets, start_date=start, end_date=end))


# ### Check the count
# 
# There should be a total of **1713** tweets in the data set that are US Tweets posted in Nov 2021.

# In[ ]:


len(nov_usa_tweets)


# In[ ]:




