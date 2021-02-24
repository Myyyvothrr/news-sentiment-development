import spacy
from spacy_sentiws import spaCySentiWS
from sentiment_analysis.negation_handling import *

# class that ranks sentiment based on a dicitionary apporach
# based on sentiws by the leipzig university
#   see: https://wortschatz.uni-leipzig.de/en/download
# implemented using a Singelton pattern


class SentimentDictionary():
    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if SentimentDictionary.__instance == None:
            SentimentDictionary()
        return SentimentDictionary.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if SentimentDictionary.__instance != None:
            raise Exception("Class sentimentDictionary is a singleton!")
        else:
            SentimentDictionary.__instance = self

        # load spacy for german
        self.nlp = spacy.load('de')

        # loads the sentiment ws data
        self.sentiws = spaCySentiWS("data/sentiws")
        self.nlp.add_pipe(self.sentiws)

    sentimentText = 0.0
    sentencesWithSentiment = {}
    compound = {}
    sentimentTextIsAdditiv = False
    saveSentencesWithSentiment = False

    def setSentimentTextAddititv(self, Boolean):
        # additiv sentiment can be enabled
        # if the text is to long to be read at once
        # or text is given piece by piece
        self.sentimentTextIsAdditiv = Boolean
    
    def saveSenteneces(self, Boolean):
        # per default sentences with sentiment are saved to check for double usage
        # This can be disabled for faster runtime or less memory usage
        self.saveSentencesWithSentiment = Boolean

    # main function
    # takes the text and a list of search Terms
    def predict_sentiment(self, text: str, searchTermList: list) -> float:
        
        if not self.sentimentTextIsAdditiv:
            # new sentiment is calculated for every function call
            self.sentimentText = 0.0

        # read the text into spacy
        doc = self.nlp(text)

        # the counter is used for normalization 
        counter = 0

        # iterate through all sentences in the document
        for sentence in doc.sents:
            sentenceText = sentence.text

            # to get a sentiment related to the search terms only sentences
            # containing a search term are evaluated
            if any(term in sentenceText.lower() for term in searchTermList):
                if sentenceText in self.sentencesWithSentiment:
                    # check that sentences are not entered twice
                    continue
                sentimentSentence = 0.0
                for word in sentence:
                    if any(term in word.text.lower() for term in searchTermList):
                        # The sentiment of search terms is neglected to reduce bias
                        self.count_this(self.compound, word.text)
                        continue
                    if word._.sentiws is not None:
                        counter += 1
                        # if word has a sentiment weight it is added to the sentiment value 
                        # and the counter in increased
                        sentimentSentence += float(word._.sentiws) * check_for_negation(sentence,word)
                if self.saveSentencesWithSentiment:
                    # save the sentences with sentiment 
                    self.count_this(self.sentencesWithSentiment, sentenceText, sentimentSentence)
                self.sentimentText += sentimentSentence
        if counter>0:
            self.sentimentText /= counter 
        return self.sentimentText

    def count_this(self, dictionary: dict, key: str, value: float = 1.0):
        # adds the given value or 1 to the key in the provided dictionary
        # is used to count occourences of words or save the sentiment of sentences
        if key in dictionary:
            dictionary[key] += value
        else:
            dictionary[key] = value


def analyse_sentiment(text: str, listSearchTerms: list) -> float:
    if not any([searchTerm in text.lower() for searchTerm in listSearchTerms]):
        return 0.0
    sd = SentimentDictionary.getInstance()
    sd.predict_sentiment(text, listSearchTerms)
    return sd.sentimentText


if __name__ == "__main__":
    
    texts = ["Flüchtlinge nehmen uns die Arbeitsplätze weg.",
             "Wir müssen uns gemeinsam anstregenen Flüchtlinge gut zu intigrieren.",
             "Wir schaffen das!", 
             "Flüchtlinge sind scheiße!", 
             "Flüchtlinge sind nicht scheiße!"]

    for t in texts:
        print(t, analyse_sentiment(t, ["flüchtlinge"]))
