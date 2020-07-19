# Quora_QuestionPair-Similarity
<h1>Introduction</h1>
This case study is called Quora Question Pairs Similarity Problem. In this case study we will be dealing with the task of pairing up the duplicate questions from quora. More formally, the followings are our problem statements

 * Identify which questions asked on Quora are duplicates of questions that have already been asked.
 * This could be useful to instantly provide answers to questions that have already been answered.
 * We are tasked with predicting whether a pair of questions are duplicates or not.
 
Note- we are talking about the semantic similarity of the questions.
source : https://www.kaggle.com/c/quora-question-pairs

Let`s look at few objectives and Constraints.

 * The cost of a mis-classification can be very high.
 * We need the probability of a pair of questions to be duplicates so that we can choose any threshold of choice.
 * No strict latency concerns. We can take more than a millisecond (let`s say) to return the probability of that the given pair of question is similar.
 * Interpretability is partially important.
 <H1>Machine learning problem</H1>

<H4>Data</H4>

The data is in a csv file named “Train.csv” which can be downloaded from kaggle itself( https://www.kaggle.com/c/quora-question-pairs).

 * Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate
 * Size of Train.csv — 60MB
 * Number of rows in Train.csv = 404,290.
 
‘qid1’ and ‘qid2’ are the ids of the respective questions, ‘question1’ and ‘queston2’ are the question bodies themselves and ‘is_duplicate’ is the target label which is 0 for non similar questions and 1 for similar questions.

This can be also thought as if ‘qid1, qid2, question1, question2,’ are the x labels and ‘is_duplicate’ is are the y labels.

<H4>Performance metrics</H4>

It is a binary classification problem, for a given pair of questions we need to predict if they are duplicate or not. The only modification is that we will be using probability scores to set the threshold.
Metrics-

 * log-loss : https://www.kaggle.com/wiki/LogarithmicLoss

 * Binary Confusion Matrix
 
Since we will be dealing with probability scores , it is best to choose log loss as our metric .Log loss always penalizes for small deviations in probability scores. Binary Confusion matrix will provide us a number of metrics like TPR, FPR , TNR, FNR, Precision and recall.

Source :https://www.kaggle.com/c/quora-question-pairs/overview/evaluation
<H1>Train and Test split</H1>
A better way of splitting the data would have been time based splitting as the types of questions change over time.But we have not been the given time stamps. Hence ,we will build train and test by randomly splitting in the ratio of 70:30 or 80:20 whatever we choose as we have sufficient points to work with.
  
<H1>Exploratory Data Analysis</H1>
In this section we will do the analyse the data to get sense of what`s happening in our data.
Each of the features has 404290 non-null values except ‘question1’ and ‘question2’ which have 1 and 2 null objects respectively. We will process these rows a differently. So this is the high level view of the data.
We have 63.08% of non duplicate pairs and 36.92% duplicate pairs.We have

 * Number of unique questions is 537933
 * Number of unique questions that appear more than ones is 111780 which is equal to 20.78 % of all the unique questions.
 * Max number of times a single question is repeated: 157
 * We have zero duplicate questions.
 This looks like an exponential distribution.
 
It is clear from the above graph that most of the questions appear less than 40 times. We have very few outliers that happen to appear more than 60 times and an extreme case of a question that appeared 157 times.

As far as null values are concerned we will just replace them with an empty space.

<H1>Basic feature Extraction (before cleaning)</H1>
We will be extracting few basic features, before cleaning . These features may or may not work with our problem.

 * freq_qid1 = Frequency of qid1's
 * freq_qid2 = Frequency of qid2's
 * q1len = Length of q1
 * q2len = Length of q2
 * q2_n_words = Number of words in Question 2
 * word_Common = (Number of common unique words in Question 1 and Question 2)
 * word_Total =(Total num of words in Question 1 + Total num of words in Question 2)
 * word_share = (word_common)/(word_Total)
 * freq_q1+freq_q2 = sum total of frequency of qid1 and qid2
 * freq_q1-freq_q2 = absolute difference of frequency of qid1 and qid2
The distributions for normalized word_share have some overlap on the far right-hand side, i.e., there are quite a lot of questions with high word similarity
The average word share and Common no. of words of qid1 and qid2 is more when they are duplicate(Similar)
This means that this feature has some value in separating the classes.
The distributions of the word_Common feature in similar and non-similar questions are highly overlapping. Hence this feature cannot be used for classification. Or in other words we can say that it has a very less value of predictive power.

<H1>Preprocessing of Text</H1>
Before we go into complex feature engineering ,we need to clean up the data. Some of the steps of preprocessing includes-

 * Removing html tags
 * Removing punctuation.
 * Performing stemming , process of reducing inflected (or sometimes derived) words to their word stem, base or root form.
 * Removing Stop words. Some examples of stop words are: “a,” “and,” “but,” “how,” “or,” and “what.”
 * Expanding contractions such as replacing “`ll” to “ will”, “n`t” to “ not”,"$" to " dollar " etc.
 
<H1>Advanced Feature Extraction (NLP and Fuzzy Features)</H1>
We will extract some advanced features. First we should familiarize ourselves with few terms.

 * Token: You get a token by splitting sentence a space
 * Stop_Word : stop words as per NLTK. NLTK is the nlp library which we are using.
 * Word : A token that is not a stop_word
Now let`s have a look at the features.

 * cwc_min : Ratio of common_word_count to min lenghth of word count of Q1 and Q2
 * cwc_min = common_word_count / (min(len(q1_words), len(q2_words))
 * cwc_max : Ratio of common_word_count to max lenghth of word count of Q1 and Q2
 * cwc_max = common_word_count / (max(len(q1_words), len(q2_words))
 * csc_min : Ratio of common_stop_count to min lenghth of stop count of Q1 and Q2
 * csc_min = common_stop_count / (min(len(q1_stops), len(q2_stops))
 * csc_max : Ratio of common_stop_count to max lenghth of stop count of Q1 and Q2
 * csc_max = common_stop_count / (max(len(q1_stops), len(q2_stops))
 * ctc_min : Ratio of common_token_count to min lenghth of token count of Q1 and Q2
 * ctc_min = common_token_count / (min(len(q1_tokens), len(q2_tokens))
 * ctc_max : Ratio of common_token_count to max lenghth of token count of Q1 and Q2
 * ctc_max = common_token_count / (max(len(q1_tokens), len(q2_tokens))
 * last_word_eq : Check if First word of both questions is equal or not
 * last_word_eq = int(q1_tokens[-1] == q2_tokens[-1])
 * first_word_eq : Check if First word of both questions is equal or not
 * first_word_eq = int(q1_tokens[0] == q2_tokens[0])
 * abs_len_diff : Abs. length difference
 * abs_len_diff = abs(len(q1_tokens) — len(q2_tokens))
 * mean_len : Average Token Length of both Questions
 * mean_len = (len(q1_tokens) + len(q2_tokens))/2
 * fuzz_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
 * fuzz_partial_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
 * token_sort_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
 * token_set_ratio : https://github.com/seatgeek/fuzzywuzzy#usage http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
 * longest_substr_ratio : Ratio of length longest common substring to min lenghth of token count of Q1 and Q2
 * longest_substr_ratio = len(longest common substring) / (min(len(q1_tokens), len(q2_tokens)).
  
 <H1>Analysis of extracted features</H1>
We will first create word clouds. Word cloud is an image composed of words used in a particular text or subject, in which the size of each word indicates its frequency or importance.
Here we can clearly see that words such as “donald ” , “trump” , “best” etc have a bigger size which implies that they have a large frequency in duplicate question pairs.
In non duplicate questions pairs we see words like “not”, “India”, “will” etc.One this to note is that thee word ‘best’ has a substantial frequency even in non duplicate pairs, but here its frequency is quite less as its image has a smaller size.
These are the pair plots of few of the advanced features. One this we can observe is that almost all the plots have partial overlapping. Hence we can conclude that these features can provide partial separability. They all provide some predictive power.
Similarly features token_sort_ratio and fuzz_ratio also provides some separability as their PDFs have partial overlapping.
<H1>Visualization</H1>
This visualization was created by performing dimensionality reduction on a sample of 5000 data points (due to limitation of computational resources) using t-SNE with perplexity = 30 and max_iter = 1000. The dimensionality was reduced from 15 to 2 . As you can see there are few regions ,which are highlighted , where we are able to separate points completely . This means we are on the right track.
<H1>Featurizing text data with tfidf weighted word-vectors</H1>
There is value in words that are present in questions . For eg we noticed that some words occur more often in duplicate question pairs (like “donald trump”) that non- duplicate pairs and vice versa. As of now, we will be using tfidf weighted word vectors. tf–idf or TFIDF, short for term frequency–inverse document frequency, is a numerical way of vectorizing text data that is intended to reflect how important a word is to a document in a collection or corpus.

 * After we find TF-IDF scores, we convert each question to a weighted average of word2vec vectors by these scores.
 * Here we use a pre-trained GLOVE model which comes free with “Spacy”. https://spacy.io/usage/vectors-similarity
 * It is trained on Wikipedia and therefore, it is stronger in terms of word semantics.
 or every question we will be having a 96 dimensional numeric vector.
After combining it with the previous features i.e. nlp and simple features , total dimensionality of the data will be 221.
We first have our advanced nlp features, then simple features and finaly our vectors of question one and question 2.

<h1>Training the models</h1>
<h3>Building a random model (Finding worst-case log-loss)</h3>
Our key performance metrics ‘log-loss’ is a function with range (0,∞] . Hence we need a random model to get an upper bound for the metric . A random model is one which when given x_i will randomly produce either 1 or 0 where both labels are equiprobable.
Image for post
0.88 turns out to be the value of log loss for our random model. A ‘decent’ model for our problem will have a value of log loss which isn`t close to 0.88.Note that have more data points for class 0 than for class 1.

<h1>Logistic Regression with hyperparameter tuning</h1>
Since our data is neither high dimensional (eg 1000 ) nor low dimensional (eg 30 ), it lies somewhat in the middle with 221 dimensions . Hence we will be first trying Logistic regression model with hyper parameter tuning. We will be using grid search.
We got our best alpha(hyper-parameter ) to be 0.1 with a log loss of about 0.4986 on the test data which is slightly better than the random model. Few things we observed about this model are-

 * The model is not suffering from over fitting since it`s log loss on train and test data are quite close. It may be suffering from high bias or under fitting
 * Model is able to predict class 0 decently but under performs in case of class 1.
 * Precision for both classes is around .85 which is not very high.
 * Recall for class zero is high , but for class 1 it is quite low.
 <h1>Linear SVM with hyperparameter tuning</h1>
 
  * Similar to logistic regression model ,linear SVM model is not suffering from over fitting since it`s log loss on train and test data are quite close. It may be suffering from high bias or under fitting.
 * Similar problem of precision and recall is with linear SVM.
 * We don`t observe any significant improvement in the model since log loss on the test data remain quite similar.
 <H1>XGboost with hyperparameter tuning</H1>
 
 * There is substantial amount of difference between the training loss and test loss which means that our model is suffering from a problem of over fitting. Still our test loss is better that the linear models .
 * We have an improvement in our precision and recall for the class 1. This means that our model is able to perform well even for class 1.
 
 
 In conclusion , XGboost tend to perform much better that the linear model. This indicates that the data is not linearly separable and we need a complex non linear model like XGboost.I will furher try to further improve performance and decrease the  log loss using deep neural network.
 
