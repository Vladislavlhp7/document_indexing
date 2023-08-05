# Document Indexing
## Project Description
The project aims to develop document indexing skills by creating a specialized search engine for The Simpsons series. The main task is to construct an inverted index from a corpus of episodes to efficiently retrieve episodes based on user queries. The index should support both single and multi-word terms, include positional information, and enable ranked results. Lexical pre-processing and normalization will be used to enhance search efficiency.

Task 1 involves building the inverted index and justifying design decisions with examples specific to The Simpsons dataset. A function will be provided to return index entries for given search terms, along with the index size and construction time. The report will discuss the implementation's features, their relevance to the dataset, and any potential issues.

Task 2 focuses on positional indexing and proximity search. A function will demonstrate how to retrieve documents containing two search terms within a specified window size. The report will discuss whether this indexing method can be used as an alternative to indexing multi-word terms and its applicability to finding entities of interest.

## Inverted Index
### Index Term Choices
The implemented system supports the following index term choices:

* Unigrams: Single words treated as the smallest morphological units after pre-processing. For example, 'Elementary' becomes 'elementari'.
* Multi-word expressions (MWE): Combinations of words from the knowledge base, assuming the proficient user already knows them. E.g., 'Springfield Elementary School' becomes 'springfield_elementari_school'.
* Numbers: Any numerical values except currency expressions, as TV shows typically refer to specific time periods or episodes.

### Critique of the System
* MWE: The knowledge base for MWEs is insufficient, leading to entities not being recognized. This can be improved by providing a more comprehensive database. Storing longer phrases can expand the vocabulary size, but it's a trade-off to consider.
* MWE Limitation: Encoding absolute term positions in each document causes a limitation in storing both MWEs and their individual words. This may lead to fast retrieval for specialized queries but reduced recall for over-specified queries.

### Pre-processing Choices
The pre-processing steps of the system are as follows:

1. Removal of Wikipedia words and characters.
2. Normalization by lowercasing all characters.
3. Removal of redundant characters and sequences.
4. Retaining stop words for consistency in word positioning and proximity searching.
5. Tokenization using NLTK's recommended word tokenizer.
6. Stemming the tokens using the Snowball method.
7. Tokenization of multiple-word expressions (MWEs).

### What is Stored in the Inverted Index
The inverted index stores the following information for each term:

* The term itself (unigram or MWE).
* Document IDs where the term occurs.
* Term frequency in each document.
* Positions of the term in each document.

### Data Structure
The implemented data structure is based on Python's dictionaries and lists, ensuring fast access and natural document order retention.

## Positional Indexing (Proximity Search)

### System Assumption: Query Equivalence
The proximity search for terms A and B in a window W does not assume any order (A -> B or B -> A) to provide relevant documents where both terms appear within the window. Query equivalence allows retrieval of relevant documents regardless of the term order.

### Multi-word Terms Search
Including MWEs in the index enables faster retrieval for specialized queries but may reduce recall for variant MWEs. Restricting MWEs to a phrase index can result in a more compact index size, improved recall, and lower memory load. However, stemming may still cause low precision.

## References
Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press, Cambridge, UK.