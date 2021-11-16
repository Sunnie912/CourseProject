
import nltk
import gensim
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim import corpora, models
nltk.download()


from os import listdir
from pprint import pprint

class Corpus(object):
    #A collection of documents.\n",
    def __init__(self, documents_path):
        
    #Initialize empty document list
        self.word_tokens = []
        self.documents = []
        self.vocabulary = []
        self.filtered_words = []
        self.documents_path = documents_path
        self.number_of_documents = 0
        self.vocabulary_size = 0
        self.ECallDocuments =[]
    def tokenize(self):
        filenames = listdir(self.documents_path)
        for filename in filenames:
               file_path = f\"{self.documents_path}/{filename}
                str = ""
                with open(file_path, \"r\") as path:
                lines = path.readlines()\n",
                    for line in lines:\n",
                        words = re.split(\"\\t|\\n| \", line)
                        for word in words:
                            str += word +
                tokens = word_tokenize(str)
                filtered_tokens = []
                tags = nltk.pos_tag(tokens)
                for word, pos in tags:
                    if (pos == 'NN' or pos == 'VERB' and len(word) > 3):
                        self.word_tokens.append(word)
               
                # Append the content to the list\n",
                self.ECallDocuments.append(self.word_tokens) # Build the training corpus 'list of lists'
         
    def preprocess_text(self):

        #remove stop words ans stemming\n",
        stop_words = set(stopwords.words(\"english\")) #choose english as the languagse
        ps = PorterStemmer()
        for word in self.word_tokens:
            if word not in stop_words and len(word) > 3:
                self.filtered_words.append(ps.stem(word))
        self.filtered_words_freDis = nltk.FreqDist(self.filtered_words)
            
        print(\"most common 20\", self.filtered_words_freDis.most_common(20))
        
    def lda_bow(self):
        dictionary = corpora.Dictionary(self.ECallDocuments)
        #covert tokenized documents into a documents-term matrix
        bow_corpus = [dictionary.doc2bow(self.filtered_words)] #bag-of-words
        bow_corpus = [dictionary.doc2bow(doc) for doc in self.ECallDocuments] # Apply Bag of Wor
        #generate LDA model
        ldamodel = gensim.models.ldamodel.LdaModel(bow_corpus, num_topics = 3, id2word = dictionary, passes = 2)
        pprint(ldamodel.print_topics(num_topics=3, num_words=20))
 
        ### GET TOPIC ALLOCATIONS FOR TRAINING CORPUS DOCUMENTS ###\n",
        # Set up Bag of Words and TFIDF
        corpus = [dictionary.doc2bow(doc) for doc in self.ECallDocuments] # Apply Bag of Wor\n",
        doc_no = 0 # Set document counter\n",
        for doc in self.ECallDocuments:
            bof_doc = ldamodel[corpus[doc_no]] # Apply TFIDF model to individual documents
            print(ldamodel.get_document_topics(bof_doc)) # Get and print document topic allocations
            doc_no += 1
    
        print('-'*50)
    
    def main():
        documents_path = 'ami-transcripts'
        corpus = Corpus(documents_path)
        corpus.tokenize()
        corpus.preprocess_text()
        print("filtered_word size: \" + str(len(corpus.filtered_words)))
        corpus.lda_bow()\n

    if __name__ == '__main__':
        main()
