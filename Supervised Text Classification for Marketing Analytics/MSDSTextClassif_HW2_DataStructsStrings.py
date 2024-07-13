#!/usr/bin/env python
# coding: utf-8

# # MSDS Text Classification. Assignment #2: Data Structures and Strings

# ## ⚡️ Make a Copy
# 
# Save a copy of this notebook in your Google Drive before continuing. Be sure to edit your own copy, not the original notebook.

# ## Overview

# ### 🐍 Continuing Python skills assessment

# The goal here is to further assess your Python readiness for the course.
# 
# > ⚠️  As mentioned in Assignment #1: If the work below is difficult for you, then you will want to be absolutely certain to go through the Python review materials before moving forward.

# ### ☑️ Skill checks

# We continue your Python skills assessment by evaluating the following abilities:
# 
#  * Write basic Python functions that meet specified requirements
#  * Work with and manipulate Python data structures, particularly lists and dictionaries
#  * Use Python control-flow constructs, particularly for-loops and if-statements
#  * Manipulate Python strings
# 
# 

# ### 📝 Completing the assignment

# There are two function definitions below for you to complete. You will need to write the code to meet the function specifications, and submit the .py version of this notebook to the grader.

# > **⚠️  Don't code outside the lines.** Keep your function implementation code inside the function blocks. Be sure not to write any code above the `/autorade` delimiter other than the specified function code. Any experimentation or testing code should go below the `/autograde` indicator, and will be ignored by the grader.

# ## 🔨 Do the work
# 
# To complete the assignment:
# 
#  1. Complete the implementations of the `unique_values` and `summary` functions.
# 
#     The function definition lines are created for you below. Your job is to complete the functions so that they work to specification.
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

# ### Complete the `unique_values` function

# In[1]:


def unique_values(data, key):
    """Where `data` is a list of dictionaries which may or may
    not contain the key specified by `key`.

    Returns a list of the distinct item values for items keyed by
    `key`. Skips any data entries that do not contain that key.

    E.g.: Consider following data list:

    data =[
        { "product_type": "shoes", "color": "red" },
        { "product_type": "hats", "color": "blue" },
        { "product_type": "shoes", "size": "large" },
        { "product_type": "shoes", "color": "red" }
    ]

    The response for the call unique_values(data, "color") would be:

    [ "red", "blue" ]

    Noting that "red" is only included 1 time, and that none of "shoes",
    "hats", or "large" are values for which the key matches "color".
    """
    return list({item[key] for item in data if key in item})
    # TODO: Implement this function to return a list of the unique items
    #       in data keyed by key


# ## 📆 A note about date formatting in Python
# 
# The standard library approach to parsing and formatting dates and times in Python is described in the [strftime() and strptime() Behavior](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior) section of the Python docs.
# 
# ### Parsing the date
# 
# The [datetime.date.fromisoformat](https://docs.python.org/3/library/datetime.html#datetime.date.fromisoformat) function will make quick work of parsing the dates in the News Category dataset, which are in the ISO format of YYYY-MM-DD
# 
# ### Formatting the data
# 
# Call the strftime(FORMAT) method of a parsed date object, with some specified FORMAT string to get a date string in the format you want. See the strftime formatting options (documentation linked above) for information about constructing a format string.

# ### Complete the `summary` function

# In[2]:


import datetime


def summary(data):
    """Returns a summary line of the data in a readable format.

    The string returned is of the format:
    DATE. CATEGORY. HEADLINE

    Where:

        DATE: Abbreviated MONTH, YEAR format. E.g.: Jan 2022
        CATEGORY: Title-cased format. E.g. CRIME becomes Crime
        HEADLINE: The headline, truncated to 50 characters.

    Example:

    {
        'authors': 'Melissa Jeltsen',
        'category': 'CRIME',
        'date': '2018-05-26',
        'headline': 'There Were 2 Mass Shootings In Texas Last Week, But Only 1 On TV',
        'link': 'https://www.huffingtonpost.com/entry/texas-amanda-painter-mass-shooting_us_5b081ab4e4b0802d69caad89',
        'short_description': 'She left her husband. He killed their children. Just another day in America.'
    }

    Returns:
        "May 2018. Crime. There Were 2 Mass Shootings In Texas Last Week, Bu"

    NOTE: some data items may not exist in the data, in which case they should be
    treated as an empty string. A missing date should not result in any date
    parsing errors.
    """
    category = data.get("category", "").title()
    article_date = data.get("date", "")
    headline = data.get("headline", "")
    
    try:
        date = datetime.datetime.strptime(article_date, '%Y-%m-%d').strftime('%b, %Y')
    except ValueError:
        date = ""

    truncated_headline = headline[:50]
    
    return f"{date}. {category}. {truncated_headline}"

    # TODO: Use the assigned variables above to complete this function and
    #       return a properly formatted summary string.


# In[3]:


#~~ /autograde # do not delete this code cell


# ---
# ### ⚠️  **Caution:** No arbitrary code above this line
# 
# The only code written above should be the implementation of your graded
# function. For experimentation and testing, only add code below.
# 
# ---

# In[4]:


data = {
    'authors': 'Melissa Jeltsen',
    'category': 'CRIME',
    'date': '2018-05-26',
    'headline': 'There Were 2 Mass Shootings In Texas Last Week, But Only 1 On TV',
    'link': 'https://www.huffingtonpost.com/entry/texas-amanda-painter-mass-shooting_us_5b081ab4e4b0802d69caad89',
    'short_description': 'She left her husband. He killed their children. Just another day in America.'
}


# > 💡  Here we test `unique_values` on a list with just one item. Consider pasting in your `read_json` function from Assignment #1 in order to load and extract unique values from a larger set of data. Alternatively, simply create your own list of data items to test.

# In[5]:


unique_values([data], "category") # Note that we must pass in a list


# In[6]:


summary(data)


# In[ ]:




