# CS779-IITK
Coursework from CS779: Statistical Natural Language Processing, taken during my third semester.

# Assignment 1

**DEADLINE: Aug 26 2025**


### Question 1: Zipf's Law

- [x] dataset and language selection (100) 
- [x] word frequency analysis (50)
- [x] word frequency analysis contd. (50)
- [x] plotting zipfs law curves (100+100)
- [x] finding k (100)
- [x] observations and interpretations (20 + 20 + 20 + 20)
- [x] Results compilation and description.
- [x] documentation


### Question 2: RegEx 

- [x] learning regex and updating nlp/topic-notes (personal)
- [x] download enron corpus and analyse
- [x] extracting words, email, phone, url using regex (20 + 20 + 20 + 20)
- [x] extracting by finding patterns in the corpus (100)
- [x] additional tasks
- [x] visualisation (20 + 20 + 20 + 20 + 20 + 20)
- [x] word frequency analysis (20)
- [x] phone number patterns (20)
- [x] email analysis (20 + 20)
- [x] url structure analysis (20 + 20)



### Question 3: Playing with Wikipedia

- [x] setting up environemnt
- [x] loading wiki corpus
- [x] corpus analysis (10 + 10 + 20 + 20 + 50)
	- [x] find number of articles
	- [x] extract subheadings using regex
	- [x] create new dataset containing subheadings and content
	- [x] find max/min number of subheadings artciles
	- [ ] challenge: generate titles using some model -> Didnt attempt this part.
- [x] tokenization:
	- [x] (50) tokenization and length calculation with token display
	- [x] (10) longest and shortest documents
	- [x] (10) average length of documents
	- [x] (10) most frequent token
	- [x] (10) histogram -> frequency vs top 200 tokens
	- [x] (20) percentage of punctuation marks (and interpretation)
	- [x] (10) normalized frequency calcultion
	- [x] (20) most frequent token based on normalised frequency
	- [x] (20) histogram -> normalised frequency vs tokens (and observation/interpretation)
	- [x] (20) unigram probability
	- [x] (10) comments on unigram probability
	- [x] (100) probability of a document
	- [x] (100)  log probability of a document
	- [x] (50) observations and interpretations (additional comments)
- [x] POS Tagging
	- [x] (50) pos tagging for each token
	- [x] (20) frequency of each tag and histogram
	- [x] (100) unigram probability of pos tags in a document
	- [x] (100) entropy of pos tag dist.
	- [x] (50) observations
	- [x] (20) validity of unigram assumption
- [x] building dict of words 
	- [x] build the dict
	- [x] (100) calculate unigram pos tag probabilities at corpus level
	- [x] (100) calculate probability dist of pos tags for each token
	- [x] (100) calculate pos tag entropy for each token
	- [x] (20) token with highest/lowest entropy
	- [x] (20) interpretation of above
- [x] named entity recognition (NER)
	- [x] (50) use spacy to predict NER tags for each token
	- [x] (100) calculate corpus wide unigram NER tag probability dist.
	- [x] (100) calculate NER entropy of each document
	- [x] (20) docs with highest NER entropy
	- [x] (20) docs with lowest NER entropy
	- [x] interpretation of above


### Question 4: N-Gram Models

- [x] load dataset
- [x] tokenize and perform pos tagging using spacy
- [x] pos tag probabilities
	- [x] (500) compute probabilities
	- [x] (50) prove formula
	- [x] (20) most frequent unigrams, bigrams, trigrams
	- [x] (60) plot bar charts of above
	- [x] (10) any interesting patterns?
	- [x] (20) repeated tags analysis and interpretation
- [x] sentence probability
	- [x] (100) 1000 random sentences, calculate prob
	- [x] (100) predict next pos tag using n gram models


### Question 5: Topic Classification
### Question 5.1: Classf using TF-IDF + KNN
- [x] read about TF-IDF and KNN
- [x] data preperation
- [x] KNN implementation from scratch
- [x] experimentation and optimization
	- [x] testing diff k values
	- [x] distance metrics
	- [x] comparitive analysis (scratch implementation vs sklearn)
- [ ] Optimizing KNN for large datasets
	- [ ] Approximate Nearest Neighbours (ANN)
	- [ ] KD Trees
	- [ ] Ball Trees
### Question 5.2: Classf using TF-IDF + K-Means
- [x] read about KMeans
- [x] K-Means implementation (sklearn only)
- [ ] Cluster Topic Assignment (markdown writeup pending)
- [ ] classification with clusters (markdown writeup pending)
- [x] performance evaluation via metrics
- [x] Testing different values of k
### Question 5.3: Comparing Classification
(just answer questions in markdown)
- [x] how do KNN and Kmeans differ in their fundamental approaches?
- [x] which method gives better results? why? discuss factors that couldve influenced the results.
- [x] how does the choice of k effect the answers?



### Question 6: Generating Word Embeddings

- [ ] math reading abut SVD but ik you're not going to do this part.
- [ ] building a word content matrix
	- [ ] load and clean dataset
	- [ ] lemmatization and stop word removal
	- [ ] build matrix
- [ ] applying SVD to the word content matrix
	- [ ] SVD application
	- [ ] Low-Rank Approximation
- [ ] word embeddings and comparision
	- [ ] generate word embeddings
	- [ ] cosine similarity
- [ ] analysis













