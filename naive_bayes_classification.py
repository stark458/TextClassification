# %%
# author: ujwol dahal


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import re

# %%
data = pd.read_csv('spam-ham-data.csv', index_col = None)
data.head()
data.shape

# %%
#our dataframe holds  columns one for message and next for the  category. .
#the next step will be data cleaning.
def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    df = df[col_list]
    return df
data = swap_columns(data,'Category','Message')
# %%
data['Category'].value_counts(normalize=True)

# %%
#removing punctuations:
import string
def remove_punctuations(text):
    for punctuation in string. punctuation:
        text = text. replace(punctuation, '')
    return text
data['Message'] = data['Message'].apply(remove_punctuations)

#lowercasing all the Messages and labels.
data= data.applymap(lambda s:s.lower() if type(s) == str else s)
data['Message'] = data['Message'].str.replace(
   '\W', ' ')


# %%
# creating vocabulary for the data:
data_bag = data['Message'].str.split()

vocabulary = []
for sentence in data_bag:
    for word in sentence:
        vocabulary.append(word)
vocabulary = list(set(vocabulary))
total_words = len(vocabulary) 


# %%
#splitting dataframe into test set and train set:

#shuffling the dataset to avoid bias or variance
data = data.sample(frac=1)
training_data = data.iloc[:round((0.75)*len(data))].reset_index(drop=True)
test_data = data.iloc[round((0.75)*len(data)):].reset_index(drop=True)

#seeing the proportion of spam and ham messages in our test set.
test_data['Category'].value_counts(normalize=True)

#Considering train_data: here indicies of messages are random and count is 4179 = 75% * 5572
training_data


# %%
word_counts = {unique_word: [0] * len(training_data['Message']) for unique_word in vocabulary}

for index, msg in enumerate(training_data['Message']):
   for word in msg:
      if word in [' ','£','’','鈥','〨']:
         continue
      try:
         word_counts[word][index] += 1
      except KeyError:
         pass
transformed_train_data = pd.DataFrame(word_counts)
transformed_train_data = pd.concat([training_data, transformed_train_data], axis=1)
transformed_train_data


# %%
##
spam_message_df = transformed_train_data.loc[transformed_train_data['Category'] == 'spam']
ham_message_df = transformed_train_data.loc[transformed_train_data['Category'] == 'ham']

#setting up priors:
prior_ham = len(ham_message_df)/len(transformed_train_data)
prior_spam = 1 - prior_ham
spam_message_df

# %%
#finding total number of token present in spam and ham messages.
total_spam_words = spam_message_df['Message'].apply(len).sum()
total_ham_words = ham_message_df['Message'].apply(len).sum()


# %%
#Now calculating class probabilities
probability_spam = {}
probability_ham = {}
alpha = 1
for term in vocabulary:
    occurrence_in_spam = spam_message_df[term].sum()
    p_spam = (occurrence_in_spam + alpha)/(total_spam_words + total_words)
    probability_spam[term] = p_spam

    occurrence_in_ham = ham_message_df[term].sum()
    p_ham = (occurrence_in_ham+alpha)/(total_ham_words + total_words)
    probability_ham[term] = p_ham



# %%
#now all words in the vocabulary have associated probabilities of spam or ham in the probability dictionary above.


# %%
test_data[test_data['Category'] == 'spam']

# %%
message_to_classify = test_data['Message'].values.tolist()
desired_labels = test_data['Category'].values.tolist()

# %%
def spam_or_ham(message):
    pS = pH = 1
    for word in message:
        if word in vocabulary:
            pS = pS * probability_spam[word]
            pH = pH * probability_ham[word]
    if pS > pH:
        return 'spam'
    else:
        return 'ham'

# %%
label = []
for message in message_to_classify:
    label.append(spam_or_ham(message))

# %%
def compute_accuracy(Y_true, Y_pred):  
    correctly_predicted = 0  
    # iterating over every label and checking it with the true sample  
    for true_label, predicted in zip(Y_true, Y_pred):  
        if true_label == predicted:  
            correctly_predicted += 1  
    # computing the accuracy score  
    accuracy_score = correctly_predicted / len(Y_true)  
    return accuracy_score  



# %%
#the accuracy of our model then is:
accuracy= compute_accuracy(desired_labels,label)

# %%
accuracy

# %%


# %%



