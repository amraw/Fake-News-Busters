## 1. Introduction
Fake news has been present in the society for a long time, but recent advances in social media platforms and technology, in general, have allowed these news articles to propagate very fast. Fake news, defined by the New York Times as “a made-up story with an intention to deceive, often for a secondary gain,” is arguably one of the most serious challenges facing the news industry today. In a December Pew Research poll, 64% of US adults said that “made-up news” has caused a “great deal of confusion” about the facts of current events. It is challenging and even unrealistic to identify rumors and hoaxes in the information age relying only on the traditional human-based fact checkers, due to both the tremendous scale and the real-time nature. We can leverage on
machine learning / deep learning techniques for the problem of stance detection.

In this project, we are planning to develop a <b> stance detection </b> tool that can automatically reason about the relationship between a new headline and body text. In other words, the task is to identify whether a headline-article body pair <b> agrees, disagree, discuss, or unrelated </b>. This project is inspired from  <a href="http://www.fakenewschallenge.org/">Fake News Challange</a> and we are also using the <a href="https://github.com/FakeNewsChallenge/fnc-1"> database provided by FNC </a>. 
This tool can be used as a building block to develop an automated veracity checker which can quickly check the veracity of a new claim based on the agreement with the claims made by other news organization with the known level of reliability and bias.

## 2. Related Work
Since <a href="https://github.com/FakeNewsChallenge/fnc-1"> this fake news dataset </a> is new, there are not enough papers to address this dataset. However, the stance detection and text classification task are thoroughly studied.

Reasearch in [1] is closely related to our problem where stance detection has been carried out on Twitter tweets. In this task, given the tweet and the target, i.e., a politician, the goal is to estimate whether the tweet is in favor, against, or neutral toward the given target. The previous task is quite similar to our job given the article body we have to predict whether it is agreeing, disagreeing, discussing or unrelated to a given headline. In twitter stance detection they are using bi-directional LSTM to read tweet condition on the target. We are also planning something quite similar, but instead of bi-directional LSTM, we are using LSTM only that conditions the analysis of the headline on the article representation.

The Fake news challenge team provided a baseline model where they extracted features from the headline and body and passed it to a gradient boosting classifier. On the train set, they achieved an accuracy of 77%, and on the test set, they achieved an accuracy of 75%.  In our model, we combined the features extracted from the headline and body and combined it with LSTM output and achieve an accuracy $$$$$.

## 3. Method

LSTM have been great in natural language processing task. That's why we choose it for our project.

Let (xh<sub>1</sub>,xh<sub>2</sub>, xh<sub>3</sub>, ...., xh<sub>n</sub>) denote the sequence of word vectors corresponding to the words in the headline and (xa<sub>1</sub>,xa<sub>2</sub>, xa<sub>3</sub>, ...., xa<sub>n</sub>) denote sequence of words in the article. Each word in the sequence is respresented by a D dimensional embedding that was pretrained using GloVe. We seperatly ecode the headline and body text. Firstly, we pass the headline embeddings to the first encoder and obtain an ecoding H = [h<sub>1</sub>, h<sub>2</sub>, ...., h<sub>n</sub>] and then we pass the artical body to the second encoder to obtain an ecoding A = [a<sub>1</sub>, a<sub>2</sub>, ...., a<sub>n</sub>].

The features extracted from the headline and body are mentioned below:- 
1. Cosine similarity: -  Firstly the headline and body are converted to TF-IDF form and then cosine similarity is calculated between them.
2. Word overlap: This feature depicts overlap between News headline and News Body. Higher the overlap, more the certainty that body of the News article agrees with headline. 

3. Presence of refuting words: If the news article contains refuting words such as 'fraud', 'hoax', 'deny', it wouldn't inspire much confidence into the news article. 

4. Polarity: The polarity of headline and body is determined by counting number of negative sentiment words. We used NRC word-emotion Association Lexicon is used to obtain a list of negative emotions associated with English words. Polarity of corpus is assigned based on whether it has odd/even number of negative words. 

5. Discuss Features: In this feature we are detecting presence of “discuss” words in the news article viz. tell, claim, verify etc. 

6. Common N-Grams : This features indicates the extent of commonality in terms of N-grams between headline and body.  We experimented with N=2,3,5,6 grams.

7. Binary co-occurrence: This feature counts how many times token in the headline occurs in body. 

8. TF-IDF : In addition to all above features, we are also feeding raw TF-IDF of headline and body to the feedforward network.  


<img src="https://raw.githubusercontent.com/amraw/Fake-News-Busters/master/fnc-1-master/feed1.png" alt="Mountain View" width="500" height="377">

