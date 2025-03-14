
- [Cranfield collection in TREC XML format](#cranfield-collection-in-trec-xml-format)
  - [1. What is Cranfield dataset ?](#1-what-is-cranfield-dataset-)
  - [2. Documents](#2-documents)
    - [2.1. Sample of document transformed in TREC format](#21-sample-of-document-transformed-in-trec-format)
    - [2.2. Sample of original Cranfield document](#22-sample-of-original-cranfield-document)
  - [3. Queries (*Topics*)](#3-queries-topics)
  - [4. Query Relevance Judgment (*Qrels*)](#4-query-relevance-judgment-qrels)
  - [5. Where can I find Cranfield collection in the original (non TREC) format ?](#5-where-can-i-find-cranfield-collection-in-the-original-non-trec-format-)

# :bookmark_tabs: Cranfield collection in TREC XML format
A repository to share the Cranfield collection composed by **documents**, **queries** and **relevance judgments** (*Qrels*) in TREC format. TREC format considers documents and queries (also known as topics) in **XML**.

TREC format is commonly used in **Information Retrieval (IR)** systems and campaigns. Hence, this repository can help to get started with *indexing*, *retrieval* and *evaluation* tasks that are performed in TREC and CLEF conferences.

:warning: We suggest experimenting with Cranfield _for learning purpose only_. Indeed, the collection (having only 1 400 documents) is not so reliable when compared with Big Data IR collections that are used nowadays.

>:bulb: The new TREC formatted Cranfield collection available in this repository can be experimented using [Terrier platform](http://terrier.org/)
>
>I prepared a [detailed tutorial here :computer:](https://github.com/oussbenk/information-retrieval-tutorial-terrier-cranfield)
>
>![Information Retrieval Tutorial using Terrier](https://github.com/oussbenk/information-retrieval-tutorial-terrier-cranfield/blob/main/screenshots/tutorial-terrier-340x80.jpg?raw=true)

## 1. What is Cranfield dataset ?
A small corpus of **1 400 scientific abstracts** and **225 queries**. It is considered among the first Information Retrieval initiatives to perform IR tasks in the 1960s.

## 2. Documents
Documents are transformed in TREC XML format using common tags. Documents are delimited by ``<doc></doc>`` tags.


**Filename :** ``cran.all.1400.xml``

**Format :** XML

**Original filename :** ``cran.all.1400``

### 2.1. Sample of document transformed in TREC format

```xml
<doc>
<docno>67</docno>
<title>dynamic stability of vehicles traversing ascending
or descending paths through the atmosphere .</title>
<author>tobak and allen.</author>
<bib>naca tn.4275, 1958.</bib>
<text>dynamic stability of vehicles traversing ascending or descending paths through the atmosphere . an analysis is given of the oscillatory motions of vehicles which traverse ascending and descending paths through the atmosphere at high speed .  the specific case of a skip path is examined in detail, and this leads to a form of solution for the oscillatory motion which should recur over any trajectory .  the distinguishing feature of this form is the appearance of the bessel rather than the trigonometric function as the characteristic mode of oscillation .</text>
</doc>
```

Tags definitions :
- **docno** : Document unique number (identifier)
- **title** : Document title
- **author** : Document author(s)
- **bib** : Bibliography
- **text** : Main document content (the article Abstract)

### 2.2. Sample of original Cranfield document
```
.I 67
.T
dynamic stability of vehicles traversing ascending
or descending paths through the atmosphere .
.A
tobak and allen.
.B
naca tn.4275, 1958.
.W
dynamic stability of vehicles traversing ascending
or descending paths through the atmosphere .
an analysis is given of the oscillatory motions of vehicles which
traverse ascending and descending paths through the atmosphere at high
speed .  the specific case of a skip path is examined in detail, and
this leads to a form of solution for the oscillatory motion which should
recur over any trajectory .  the distinguishing feature of this form is
the appearance of the bessel rather than the trigonometric function as
the characteristic mode of oscillation .
```


## 3. Queries (*Topics*)

**Filename :** ``cran.qry.xml``

**Format :** XML

**Original filename :** ``cran.qry``

**Sample of a topic in TREC format :**
```xml
<top>
<num> 1</num> 
<title>
what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft .
</title>
</top>
```

## 4. Query Relevance Judgment (*Qrels*)
**Filename :** ``cranqrel.trec.txt``

**Original filename :** ``cranqrel``

In the original Cranfield collection, relevance judgments were made in 5 levels like follows :

| Relevance | Definition | Query/Docs <br>Match Count | % |
|:---:|---|:---:|:---:|
| -1 | References of no interest. | 225 | 12.2% |
| 1 | References of minimum interest, <br>for example, those that have been included from an historical viewpoint. | 128 | 7.0% |
| 2 | References which were useful, either as general background to the work <br>or as suggesting methods of tackling certain aspects of the work. | 387 | 21.1% |
| 3 | References of a high degree of relevance, the lack of which <br>either would have made the research impracticable <br>or would have resulted in a considerable amount of extra work. | 734 | 40.0% |
| 4 | References which are a complete answer to the question. | 363 | 19.8% |

:warning: This way of assessing relevance was aborted in future TREC campaings as detailed in the following reference :

*Voorhees, Ellen M.* "The philosophy of information retrieval evaluation." Workshop of the cross-language evaluation forum for european languages. Springer, Berlin, Heidelberg, 2001.

[PDF paper](https://www.inf.ed.ac.uk/teaching/courses/tts/handouts2017/VoorheesIREvaluation.pdf)

>"The vast majority of test collection experiments since then have also assumed that relevance is a **binary choice**, though the original Cranfield experiments used a five-point relevance scale."

[...]

>"TREC has almost always used binary relevance judgments either a document is relevant to the topic or it is not. To define relevance for the assessors, the assessors are told to assume that they are writing a report on the subject of the topic statement. If they would use any information contained in the document in the report, then the (entire) document should be marked relevant, otherwise it should be marked irrelevant. The assessors are instructed to judge a document as relevant regardless of the number of other documents that contain the same information."

Hence, we consider the new TREC format of a Qrels file as follows:

| TOPIC | ITERATION | DOCNO | RELEVANCY |
|:---:|---|:---:|:---:|

Where :
 - **TOPIC** is the topic (query) number,
 - **ITERATION** is the feedback iteration (almost always zero and not used). ***This column was added in the new file format***,
 - **DOCNO** is the official document number that corresponds to the "docno" field in the documents, and
 - **RELEVANCY** is a binary code of 0 for not relevant and 1 for relevant.

 :page_facing_up: [From Official TREC reference](https://trec.nist.gov/data/qrels_eng/)

:white_check_mark: All initial relevancies having the scores **1, 2, 3, or 4** are considered as relevant (**all replaced with 1 value**) in the new formatted Cranfield collection. However, **-1 non relavant** initial assessments are **replaced with 0**.

**Sample of new Qrels in TREC format**

| TOPIC | ITERATION | DOCNO | RELEVANCY |
|:---:|:---:|:---:|:---:|
| 5 | 0 | 552 | 1 |
| 5 | 0 | 401 | 1 |
| 5 | 0 | 1297 | 1 |
| 5 | 0 | 1296 | 1 |
| 5 | 0 | 488 | 0 |


## 5. Where can I find Cranfield collection in the original (non TREC) format ?
Cranfield collection can be downloaded in the original format (non TREC format) from the University of Glasgow website [here](http://ir.dcs.gla.ac.uk/resources/test_collections/cran/).
