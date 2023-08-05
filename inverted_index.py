import os
import pathlib
import re
import string
import time
from collections import defaultdict
from typing import List, Union

import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.tokenize import MWETokenizer


class InvertedIndex:
    """
        Construct Inverted Index for a given corpus of documents and provide methods to query it.
    """

    def __init__(self):
        self.inverted_index = defaultdict(dict)
        self.documents_str = []
        self.documents_processed = []
        self.document_id = {}

        self.simpsons_location_df = pd.DataFrame()
        self.simpsons_character_df = pd.DataFrame()

        # To handle multi-word expressions we could use the MWE Tokenizer
        self.mwe_separator = "_"
        self.mwe_tokenizer = MWETokenizer(separator=self.mwe_separator)
        # Snowball Stemmer is an upgraded version of Porter Stemmer
        self.stemmer = SnowballStemmer(language="english")

    def read_data(self, path: str) -> List[str]:
        """
            Read files from a directory and then append the data of each file into a list.
        :param path: path to the directory
        :return:
        """
        files = os.listdir(path)  # to be iterated over in folder
        content_list = []  # to store the contents of each file in folder
        doc_iterator = (
            0  # to iterate only over the txt documents while storing the filename
        )
        for filename in files:
            file_path = f"{path}/{filename}"
            file_extension = pathlib.Path(file_path).suffix
            if ".txt" == file_extension:  # ensure we only read from *.txt files
                with open(file_path) as f_open:
                    content_lines = f_open.read()  # grab the contents of the file
                self.document_id[doc_iterator] = filename.replace(
                    ".txt", ""
                )  # map local document index to namespace
                doc_iterator += 1
                content_list.append(content_lines)
            elif ".csv" == file_extension:  # read additional csv files
                if "character" in file_path:
                    self.simpsons_character_df = pd.read_csv(file_path)
                elif "location" in file_path:
                    self.simpsons_location_df = pd.read_csv(file_path)
        return content_list

    def process_document(self, document: str) -> List[str]:
        """
                Pre-process a document and return a list of its terms
        :param document: document to be processed
        :return:
        """
        self.documents_str.append(document)

        # Remove obvious non-Simpsons vocabulary / Wikipedia words
        vocab_to_be_dropped = [
            "→",
            "←",
            "From Wikipedia, the free encyclopedia",
            "Jump to navigation",
            "Jump to search",
            "[edit]",
        ]
        for w in vocab_to_be_dropped:
            document = document.replace(w, "")
        # Normalize text by converting to lower case to minimise query time, computational and memory costs
        document = document.lower()
        # Remove redundant characters directly with custom regex to ease tokenization
        reg_to_drop = r"""(?x)           # flag to allow comments and multi-line regex
                    \[[0-9]*]            # remove all citation brackets, eg. "[12]"
                  | \( | \)              # remove ()-brackets as their contents should be indexed
                  | ^\$[\d,]+(\.\d*)?$   # cut out the money expressions
                  | \` | \.\.\.          # or any irrelevant characters
                  | \w+.(png|jpeg|jpg)   # remove PNG files
                  | [^(!"#$%&'()*+, -.\/:;<=>?@[\]^_`{|}~) | (a-zA-Z0-9\s)]  # remove non-word, non-digit, non-punctuation characters, non-space, e.g. '•', '≠'
                  | production\scode\s[0-9][a-z][0-9]+  # remove production codes
        """
        reg_compiler = re.compile(reg_to_drop)
        document = reg_compiler.sub(r"", document)

        # Handle apostrophes
        document = re.sub(r"won\'t", "will not", document)
        document = re.sub(r"n\'t", " not", document)
        document = re.sub(r"\'t", " not", document)
        document = re.sub(r"\'m", " am", document)
        document = re.sub(r"\'re", " are", document)
        document = re.sub(r"\'ll", " will", document)
        document = re.sub(r"\'ve", " have", document)

        # Retain STOP words to preserve ordering when doing proximity search
        # for w in stopwords.words():
        #   document = document.replace(w, '')

        # Apply NLTK’s recommended word tokenizer for single word expressions (swe)
        swe_list = word_tokenize(document)
        # Apply stemming to the list of tokens
        swe_list = [self.stemmer.stem(w) for w in swe_list]
        # Create stemmed multi-word expressions (MWE) from stemmed single words
        # In this way we could still keep track of multi-form MWEs
        # MWETokenizer 'adds on' words by being agnostic to the previous tokenizer
        mwe_list = self.mwe_tokenizer.tokenize(swe_list)
        # Remove punctuation tokens that are not part of multi- or single-word expressions
        mwe_list = [w for w in mwe_list if w not in string.punctuation]
        return mwe_list

    def index_corpus(self, documents: list, verbose: bool = False) -> None:
        """
            Index given documents
        :param documents: list of documents to be indexed
        :param verbose: whether to print out the time taken to index the documents
        :return:
        """
        global_time_start = time.time()
        start = None
        # ###########################################################################
        # Load Location & Character multi-word expressions prior to document indexing
        # ###########################################################################
        if verbose:
            start = time.time()
        self.mwe_tokenizer = self.insert_mwe_from_df(
            mwe_tokenizer=self.mwe_tokenizer,
            stemmer=self.stemmer,
            df=self.simpsons_character_df,
            col_name="name",
        )
        self.mwe_tokenizer = self.insert_mwe_from_df(
            mwe_tokenizer=self.mwe_tokenizer,
            stemmer=self.stemmer,
            df=self.simpsons_location_df,
            col_name="name",
        )
        if verbose:
            end = time.time()
            print(
                f"Time to read Locations & Characters csvs took {np.round(end - start, 4)} sec"
            )

        # ###########################################################################
        # Process the documents
        # ###########################################################################
        if verbose:
            start = time.time()
        for doc in documents:
            self.documents_processed.append(self.process_document(doc))
        if verbose:
            end = time.time()
            print(f"Time to process the documents took {np.round(end - start, 4)} sec")

        # ###########################################################################
        # Index documents
        # ###########################################################################
        if verbose:
            start = time.time()
        for doc_id, doc in enumerate(self.documents_processed):
            # Documents are now in token list format
            for tkn_id, tkn in enumerate(doc):
                # token -> { doc_1 -> {[occurrences, [pos_1, pos_2, ...]]} , ...}
                if self.inverted_index.get(tkn) and self.inverted_index.get(tkn).get(
                    doc_id
                ):
                    self.inverted_index[tkn][doc_id][0] += 1
                    self.inverted_index[tkn][doc_id][1].append(tkn_id)
                else:
                    self.inverted_index[tkn][doc_id] = [1, [tkn_id]]

        # Sorting is unnecessary when using Python's dictionary
        # Hence, it could be omitted to relax computational complexity
        # self.inverted_index = dict(sorted(self.inverted_index.items()))
        if verbose:
            end = time.time()
            print(
                f"Time to build the inverted index from documents took {np.round(end - start, 4)} sec"
            )
        global_time_end = time.time()
        global_time_elapsed = np.round(global_time_end - global_time_start, 4)
        index_size = len(self.inverted_index.keys())
        print(f"Size of index is: {index_size}")
        if verbose:
            print(f"Start time: {global_time_start}")
            print(f"End time: {global_time_end}")
        print(f"Total time to create the inverted index: {global_time_elapsed} sec")

    def proximity_search(self, term1: str, term2: str, window: int = 5) -> dict:
        """
            1) check whether given two terms appear within a window
            2) calculate the number of their co-existence in a document
            3) add the document id and the number of matches into a dict
        :param term1: first term
        :param term2: second term
        :param window: window size
        :return: dict of document ids and the number of matches
        """
        # Pre-process terms and ensure they are a single expression
        term1_processed = self.process_document(term1)
        term2_processed = self.process_document(term2)
        # When a multi-word expression (MWE) is not recognized it will be split during preprocessing
        # Which is why we need to assemble it into one expression (SWE)
        # This operation does not affect a normal SWE
        term1_processed = str(self.mwe_separator).join(term1_processed)
        term2_processed = str(self.mwe_separator).join(term2_processed)

        # ######################################################################
        # I. Check whether given two terms appear within a window
        # ######################################################################
        # Find intersection of documents where both terms occur
        posting1 = self.get_postings_per_word(term1_processed)
        posting2 = self.get_postings_per_word(term2_processed)
        merged_postings = self.merge_postings(posting1, posting2)

        # Find documents which have both terms within the same window
        coexistance_dict = {}
        for doc_id in merged_postings:  # iterate over all documents of interest
            # Extract the positions of the terms in the particular document
            positions1 = self.inverted_index[term1_processed][doc_id][
                1
            ]  # [number of occurrences, [occurrences]]
            positions2 = self.inverted_index[term2_processed][doc_id][1]
            doc_filename = self.get_doc_filenames_from_ids([doc_id])[0]
            proximity = self.proximity_of_arrays(
                positions1=positions1, positions2=positions2, window=window
            )
            if len(proximity) > 0:
                print(
                    f'In document {doc_filename} pre-processed terms: "{term1_processed}" & "{term2_processed}" are within a window of {window} at positions: {proximity}'
                )
                # Add the document id and the number of matches into a dict
                coexistance_dict[doc_filename] = len(proximity)
        return coexistance_dict

    def dump(self, examples: list) -> List[Union[str, None]]:
        """
            Provide a dump function to show index entries for a given set of terms
            If the provided examples are multiple term, the respective posting lists will be merged
        :param examples: list of terms
        :return: list of index entries
        """
        merged_postings = []
        processed_examples = []
        omitted_examples = []
        assert len(examples) > 0
        # Ensure our query is processed in the same way as the index entries
        for example in examples:
            processed_example = self.process_document(example)
            # When a multi-word expression (MWE) is not recognized it will be split during preprocessing
            # Which is why we need to assemble it into one expression (SWE)
            # This operation does not affect a normal SWE
            processed_example = str(self.mwe_separator).join(processed_example)
            self.print_index_entry(term=example, processed_term=processed_example)
            if processed_example in self.inverted_index.keys():
                # Look for only examples which are index so that they could retrieve something
                processed_examples.append(processed_example)
            else:
                # The ones which are missing will be reported to user
                # A reasonable strategy is to ignore this words that do not occur in any documents
                # That way, the user retrieves at least something from the query
                omitted_examples.append(processed_example)
        if len(omitted_examples) > 0:
            print(f"These terms are not in the inverted index: {str(omitted_examples)}")
        if len(processed_examples) == 0:
            print("Input is not in the inverted index")
            return [None]
        elif len(processed_examples) == 1:
            single_postings_list = self.get_postings_per_word(processed_examples[0])
            return self.get_doc_filenames_from_ids(single_postings_list)
        else:  # when processed examples are >= 2
            # Pull the sorted postings per processed word from the inverted index
            posting1 = self.get_postings_per_word(processed_examples[0])
            posting2 = self.get_postings_per_word(processed_examples[1])
            # Find the intersection between the posting lists
            merged_postings = self.merge_postings(posting1, posting2)
        # Use first intersection and merge it with the following posting lists
        for i in range(2, len(processed_examples)):
            posting_tmp = self.get_postings_per_word(processed_examples[i])
            # if there is at least one document containing this word, merge with previous postings
            merged_postings = self.merge_postings(merged_postings, posting_tmp)
        if not merged_postings:
            print("No results for the set of terms provided:", str(processed_examples))
            return [None]
        doc_ids = self.get_doc_filenames_from_ids(merged_postings)
        print(f"{str(processed_examples)} are in {doc_ids}")
        return doc_ids

    def print_index_entry(self, term: str, processed_term: str):
        """
            Pretty print the term, document frequency and document ids
        :param term: original term
        :param processed_term: processed term
        :return: None
        """
        print(f'"{term}" is processed as "{processed_term}"')
        if processed_term not in self.inverted_index.keys():
            print(f'"{processed_term}" does not occur in the inverted index')
        else:
            print(
                f"Document frequency: {len(self.inverted_index[processed_term].keys())}"
            )
            print("Document IDs:", end=" ")
            print(
                self.get_doc_filenames_from_ids(
                    list(self.inverted_index[processed_term].keys())
                )
            )
        print()

    def get_postings_per_word(self, processed_word: str) -> List[int]:
        """
            Return the sorted posting list for a provided processed word.
        :param processed_word: processed word
        :return: sorted posting list
        """
        return list(self.inverted_index[processed_word].keys())

    def get_doc_filenames_from_ids(self, postings_ids: List[int]) -> List[str]:
        """
            Return the filenames of the document from a posting list.
        :param postings_ids: posting list
        :return: list of filenames
        """
        return [self.document_id[i] for i in postings_ids]

    @staticmethod
    def insert_mwe_from_df(
        mwe_tokenizer: MWETokenizer,
        stemmer: SnowballStemmer,
        df: pd.DataFrame,
        col_name: str,
    ) -> MWETokenizer:
        """
            Load a data column with multi-word expressions to be added to the MWE Tokenizer
        :param mwe_tokenizer: MWE Tokenizer
        :param stemmer: Stemmer
        :param df: Dataframe
        :param col_name: Column name
        :return: MWE Tokenizer
        """
        if not df.empty:
            for name in df[col_name]:
                # Normalize (lowercase) expression and split into separate units
                name_split_arr = str(name).lower()
                # Tokenize each word in the name
                name_tokenized_arr = word_tokenize(name_split_arr)
                # Apply Stemming on each part of the name
                name_stemmed_arr = [stemmer.stem(w) for w in name_tokenized_arr]
                mwe_tuple = tuple(name_stemmed_arr)
                mwe_tokenizer.add_mwe(mwe_tuple)
        return mwe_tokenizer

    @staticmethod
    def merge_postings(postings1: List[int], postings2: List[int]) -> List[int]:
        """
            Merge two sorted postings lists to find the intersection
        :param postings1: sorted postings list
        :param postings2: sorted postings list
        :return: merged postings list
        """
        i, j = 0, 0
        merged_postings = []
        while i < len(postings1) and j < len(postings2):
            if postings1[i] == postings2[j]:
                merged_postings.append(postings1[i])
                j += 1
                i += 1
            elif postings1[i] > postings2[j]:
                j += 1
            else:
                i += 1
        return merged_postings

    @staticmethod
    def proximity_of_arrays(
        positions1: List[int], positions2: List[int], window: int = 5
    ) -> List[List[Union[int, int]]]:
        """
            Implement proximity of array elements within a window
        :param positions1: sorted positions list
        :param positions2: sorted positions list
        :param window: window size
        :return: list of combinations within the window
        """
        i, j = 0, 0
        coexistance_list = (
            []
        )  # to hold the combinations which are within the window frame
        window_match_list = []  # to keep the
        while i < len(positions1):  # fix the start of the window with the first term
            while j < len(positions2):  # explore the options in proximity
                if np.abs(positions1[i] - positions2[j]) <= window:
                    window_match_list.append(positions2[j])
                elif (
                    positions2[j] > positions1[i]
                ):  # no options are located in proximity
                    break
                j += 1
            # Clean up the proximity list with respect to the start of the window
            while (
                window_match_list
                and np.abs(window_match_list[0] - positions1[i]) > window
            ):
                window_match_list.pop(0)
            # Combinations are in proximity
            for match in window_match_list:
                coexistance_list.append([positions1[i], match])
            i += 1
        return coexistance_list
