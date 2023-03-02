# Building a simple Information Retrieval System using BM25 and GPT-3 and evaluated in the CISI collection


Initial project for IA368V DD course, on first term of 2023.

Eduardo Seiti de Oliveira, RA 940011


## Introduction

Having no previous experience on BM25 ranking function, nor with the CISI collection, for this initial project I relied on ChatGPT for the initial explorations as well as for the basic structure of the test implementation.


## Implementation details

### Parsing CISI files

Most of the implementation effort was dedicated to parsing the CISI collection data, especially the ones containing the documents collection and the queries (CISI.ALL and CISI.QRY respectively) which shared the same format. For that task the ChatGPT interaction proved ineffective, as the both the source code provided, as well as the files format details were incorrect. Check the [Using ChatGPT](#Using-ChatGPT) section for more information. To implement the CISI files parsing, details provided by the [Kaggle](https://www.kaggle.com/datasets/dmaso01dsta/cisi-a-dataset-for-information-retrieval) were considered.


### Preprocessing the CISI files

Closely following the sample source code provided by ChatGPT, the data read from the documents and queries files is preprocessed:

1. The words are tokenized using the [nltk](https://www.nltk.org/) package tools.

2. Then stopwords are removed, also using **nlkt** tools for English.


### Computing the BM25 scores

In the implementation final part, each query is scored against each collection document in order to define which ones are related, i.e., which document is relevant to a given query, according to the BM25 scoring function. The resulting query/documents correlation can then be compared against the ground truth provided in the CISI collection (CISI.REL file).

However, to perform the analysis we considered 2 hyperparameters and a score relevance threshold values needed to be defined: 

* The BM25 scoring function has 2 hyperparameters — **k1** and **b** — which need to be optimized for each dataset.

* The final score, reported by the BM25 scoring function, meaning also depends on the dataset being considered. Hence, the score threshold to consider a query/document correlation relevant in the CISI collection can also be fine-tuned. 


As an initial approach, we performed a grid search on the BM25 hyperparameters, considering the query/document correlation performance against the ground truth applying a score threshold of 5, evaluated on the basic **recall** metric: the idea was to select a hyperparameters set able to return the maximun number of related documents under a fairly loose score threshold. Then, applying those selected hyperparameters the score threshold could be fine-tuned in order to improve the **precision** and **f1 score** metrics.


## Brief discussion on the results

The first grid search indicated the best BM25 hyperparameters would be **k1==2.0** and **b==0.2** considering both the **recall** and the **average precision** metrics, respectively **0.603635** and **0.05278**.

Applying those hyperparameters, the grid search for the score threshold resulted in different values, depending on the metric considered:


* score_threshold==10.5, for a best **precision** of 0.172477
* score_threshold==6.6, for a best **f1-score** of 0.11712
* score_threshold==13.67 for a best **Mean Average Precision** of 0.062508


If ChatGPT answers can be believed (check next section), those results are way worst than the state-of-the-art for the CISI collection.

Since the overall structure of the test implementation was done following the instructions provided by ChatGPT, checking for missing preprocessing steps, or even misuse of the BM25 APIs would be initial approaches to try enhancing the results.



## Using ChatGPT

In the context of this project, ChatGPT showed to be a powerful tool for the initial explorations, but also somewhat deceiving when providing information regarding implementation details.

Some examples of what can be considered successful examples of ChatGPT interaction are bellow:

**QUESTION: How does the CISI collection can be used?**

**ANSWER:**

	The CISI (Computer and Information Science Index) collection is a bibliographic database 
	that contains references to articles, conference proceedings, and other documents in the 
	field of computer science and information science. This collection can be used in a number of ways:

	1. Literature reviews: Researchers can use the CISI collection to identify relevant articles 
	and other publications on a particular topic, allowing them to conduct a comprehensive 
	literature review.

	2. Information retrieval research: The CISI collection has been widely used in information 
	retrieval research, as it provides a large and diverse set of documents for testing and 
	evaluating search algorithms and techniques.

	3. Education: Students and educators in computer science and information science can use the 
	CISI collection to learn about the history and development of the field, as well as to identify 
	key publications and researchers.

	4. Data analysis: The CISI collection can be used to perform bibliometric analysis, such as 
	examining the citation patterns of authors or identifying trends in research over time.

	Overall, the CISI collection is a valuable resource for anyone interested in the field of 
	computer science and information science, whether for research, education, or professional 
	development.



**QUESTION: What can you tell about the Okapi BM25?**

**ANSWER:**

	Okapi BM25 (Best Matching 25) is a widely used ranking function for information retrieval. 
	It was developed as part of the Okapi information retrieval system, which was developed in 
	the 1980s and 1990s by researchers at City University, London.

	BM25 is a probabilistic retrieval model that estimates the relevance of a document to a 
	query based on the frequency of query terms in the document and the collection as a whole. 
	It is a bag-of-words model, which means that it treats documents and queries as collections 
	of individual words, rather than taking into account the relationships between words.

	The BM25 ranking function takes into account three factors: term frequency, inverse document 
	frequency, and document length. The term frequency factor measures how often a query term 
	appears in a document, with more frequent occurrences being assigned a higher weight. The 
	inverse document frequency factor measures how common a term is across the collection, with 
	rarer terms being assigned a higher weight. The document length factor penalizes longer 
	documents, on the assumption that longer documents are less likely to be relevant.

	The values of the parameters in the BM25 ranking function can be tuned to optimize retrieval 
	performance for a particular collection or task. BM25 has been shown to perform well on a 
	wide range of information retrieval tasks, and it is one of the most widely used ranking 
	functions in the field.

	Overall, Okapi BM25 is a powerful and widely used ranking function for information retrieval, 
	which has been shown to be effective in a wide range of settings.


Now, here are some examples of wrong answers provided by ChatGPT:

**QUESTION: Can you indicate a python tutorial on how to apply BM25 to the CISI collection?** 

**ANSWER:**

	Yes, there are several Python libraries that implement the BM25 ranking function and can 
	be used to apply it to the CISI collection. One such library is the rank_bm25 library, which 
	provides an implementation of BM25 that can be used with Python's scikit-learn library.

	Here's a brief tutorial on how to use rank_bm25 to apply BM25 to the CISI collection:

	Install the rank_bm25 library by running the following command in your terminal:

		pip install rank_bm25

	Download the CISI collection from the University of Glasgow's website and extract the files.


	Load the CISI collection into Python using the pandas library:

		import pandas as pd

		# Load the CISI documents
		docs_df = pd.read_csv('cisi/CISI.ALL', sep='\n', header=None, names=['text'])

		# Load the CISI queries
		queries_df = pd.read_csv('cisi/CISI.QRY', sep='\n', header=None, names=['text'])

		# Load the relevance judgments
		qrels_df = pd.read_csv('cisi/CISI.REL', sep=' ', header=None, names=['query_id', 'Q0', 'doc_id', 'rel'])


Parsing the CISI files require much more work than showed. Even telling ChatGPT the answer was wrong, the new sample code did not work either.



Other example of wrong information provided by ChatGPT:

**QUESTION: How does the cross-reference part of CISI.ALL document shall be read?** 

**ANSWER:**


	The CISI.ALL file in the CISI collection includes a "cross-reference" section at the end 
	of each document, which lists other documents that are cross-referenced by the current 
	document. This section is denoted by the tag XREF and is followed by a list of document 
	numbers, each of which is preceded by the tag D.

	For example, here is an excerpt from document 1 in the CISI collection:

		.I 1
		.T
		experimental investigation of the aerodynamics of a wing in a slipstream .
		.A
		brenckman,m.
		.B
		journal of basic engineering , march 1967, pp. 39-50 . 
		.XREF
		D 960 D 1266 D 1843 D 1956 D 2148


The XREF section format is completely wrong, when comparing to the actual files. This indicates ChatGPT is hallucinating the information, while following the most probable sequence of words/tokens.


After seen such results, all the ChatGPT answers need to be put in perspective: if it was hallucinating the XREF section format, it could also be providing wrong ― also invented ― information on the other answers. A simple question like the following cannot be completely trusted at first:


**QUESTION: What are the best results in the CISI collection?** 

**ANSWER:**

	The best results achieved on the CISI collection using various information retrieval methods are 
	reported in the literature. Here are some examples of the best results achieved on the CISI collection:

	* In a 1973 study by Harman and others, the best result on the CISI test set was achieved using a
	 combination of query expansion and relevance feedback, with a mean average precision (MAP) of 
	 0.365. MAP is a standard evaluation metric for information retrieval that measures the average 
	 precision across all queries in the test set.

	* In a 1994 study by Robertson and others, the best result on the CISI test set was achieved 
	using a combination of statistical and linguistic techniques, with a MAP of 0.478.

	* In a 1995 study by Turtle and Croft, the best result on the CISI test set was achieved using 
	a language modeling approach, with a MAP of 0.537.

	* In a 2004 study by Zhai and Lafferty, the best result on the CISI test set was achieved using 
	a variant of the language modeling approach called Dirichlet smoothed language modeling, with a 
	MAP of 0.556.

	It's worth noting that these results were achieved using different experimental setups and 
	evaluation metrics, so they may not be directly comparable. Additionally, these results were reported 
	in the literature several years ago and may no longer be the state of the art on the CISI collection.
	 However, they provide some context for the performance that can be achieved on the CISI collection 
	 using different information retrieval techniques.

Do those reference really exists? If so, do they reach the exactly performance mentioned?


One aspect that makes using ChatGPT a challenge is the fact it does not clearly indicate the sources of the information it provides: when I Google "CISI collection" and reach a page inside Kaggle, I know the information can be trusted. This has been discussed in this [recent talk](https://www.youtube.com/watch?v=nzRYS9XCWKA) from USP ICMC.