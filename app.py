#-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x
# Importing Libraries

# Running Streamlit
import streamlit as st
st.set_page_config( # Added favicon and title to the web app
     page_title="Youtube Summariser",
     page_icon='favicon.ico',
     layout="wide",
     initial_sidebar_state="expanded",
 )
import base64
import streamlit as st
from pytube import YouTube  # Add this line to import the YouTube module

def youtube_summarizer_app():
    st.title("YouTube Video Summarizer")

    # Input for user to paste URL
    video_url = st.text_input("Paste the YouTube video URL here:")

    if st.button("Summarize"):
        try:
            # Download YouTube video
            st.write("Downloading the video...")
            yt = YouTube(video_url)

            # Get video title
            video_title = yt.title
            st.write(f"Video Title: {video_title}")

            # Other summarization steps...
            # (You can add the summarization process here as you described earlier)

        except Exception as e:
            st.write("Error:", str(e))

# Extracting Transcript from YouTube
from bs4 import BeautifulSoup
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse
from textwrap import dedent
from pytube import YouTube

#Translation and Audio stuff
from deep_translator import GoogleTranslator
from gtts import gTTS

#Abstractive Summary
import transformers
from transformers import T5ForConditionalGeneration, T5Tokenizer

#-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x
# All Funtions

# Sumy Summarization
import nltk
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.stemmers import Stemmer
import spacy

nlp = spacy.load("en_core_web_lg")



# Download the "punkt" resource
nltk.download('punkt')

def sumy_summarize(text_content, percent):
    parser = PlaintextParser.from_string(text_content, Tokenizer("english"))
    stemmer = Stemmer("english")
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = {"english"}

    num_sentences = int(len(parser.document.sentences) * (int(percent) / 100))
    summary = " ".join(str(s) for s in summarizer(parser.document, num_sentences))

    return summary

# Sample Paragraph
a = '''Ukraine and Russia made tentative progress in talks Monday but failed to reach a deal on creating "humanitarian corridors" from pummelled cities, as the bloodshed from Moscow's invasion mounted.
Kyiv said there had been "positive results" from the third round of negotiations, focused on giving civilians evacuation routes from besieged towns, but Russia said its expectations from the talks were "not fulfilled".

Ukraine today accepted Russia's proposal of setting up humanitarian corridors.

Russian President Vladimir Putin said he is not sending conscripts or reservists to fight and that "professional" soldiers fulfilling "fixed objectives" are leading the war in Ukraine.

Ukraine's President Volodymyr Zelensky renewed calls for the West to boycott Russian exports, particularly oil, and to impose a no-fly zone to stop the carnage.

More than 1.7 million people have fled Ukraine since Russia launched its full-scale invasion on February 24.'''

# Printing Output
print(sumy_summarize(a, 50))



# NLTK Summarization
import nltk
from string import punctuation
from heapq import nlargest
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

def nltk_summarize(text_content, percent):
    tokens = word_tokenize(text_content)
    stop_words = stopwords.words('english')
    punctuation_items = punctuation + '\n'

    word_frequencies = {}
    for word in tokens:
        if word.lower() not in stop_words:
            if word.lower() not in punctuation_items:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1
    max_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency
    sentence_token = sent_tokenize(text_content)
    sentence_scores = {}
    for sent in sentence_token:
        sentence = sent.split(" ")
        for word in sentence:
            if word.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.lower()]

    select_length = int(len(sentence_token) * (int(percent) / 100))
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    final_summary = [word for word in summary]
    summary = ' '.join(final_summary)
    return summary

# Spacy Summarization
from youtube_transcript_api import YouTubeTranscriptApi

from IPython.display import YouTubeVideo

video=input("Enter the link of your YouTube Video: ")
id_video=video.split("=")[1]
print(id_video)

YouTubeVideo(id_video)
transcript = YouTubeTranscriptApi.get_transcript(id_video)
transcript
doc = ""
for line in transcript:
    doc =doc+ ' ' + line['text']
print(type(doc))
print(doc)
#print(len(result))
doc=[]
for line in transcript:
  if "\n" in line['text']:
    x=line['text'].replace("\n"," ")
    doc.append(x)
  else:
    doc.append(line['text'])
print(doc)

paragraph=" ".join(doc)
print(paragraph)

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
nltk.download('stopwords')

import nltk
nltk.download('punkt')

mytext= paragraph
stops = set(stopwords.words('english'))
word_array = word_tokenize(mytext)

wordfreq=dict()
for word in word_array:
  word=word.lower()
  if word in stops:
    continue
  elif word in wordfreq:
    wordfreq[word]+=1
  else:
    wordfreq[word]=1

#word_array
#frequencytable
sent_array=sent_tokenize(mytext)

sentfreq=dict()
for sentence in sent_array:
  for word,freq in wordfreq.items():
    if word in sentence.lower():
      if sentence in sentfreq:
        sentfreq[sentence]+=freq
      else:
        sentfreq[sentence]=freq  

#sentfreq

averageval=0
for sentence in sentfreq:
  averageval+=sentfreq[sentence]

average=int(averageval/len(sentfreq))
summary=''
for sentence in sent_array:
  if(sentence in sentfreq) and (sentfreq[sentence]>(1.5*average)):
    summary=summary+" "+sentence
print(summary)
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

stopwords=list(STOP_WORDS)
from string import punctuation
punctuation=punctuation+ '\n'
#punctuation

text=paragraph
space = spacy.load('en_core_web_lg')
doc= space(text)
word_frequencies={}
for word in doc:
  if word.text.lower() not in stopwords:
      if word.text.lower() not in punctuation:
          if word.text not in word_frequencies.keys():
              word_frequencies[word.text] = 1
          else:
              word_frequencies[word.text] += 1


max_frequency=max(word_frequencies.values())
for word in word_frequencies.keys():
  word_frequencies[word]=word_frequencies[word]/max_frequency
  sentence_tokens= [sent for sent in doc.sents]
sentence_scores = {}
for sent in sentence_tokens:
  for word in sent:
      if word.text.lower() in word_frequencies.keys():
          if sent not in sentence_scores.keys():                            
            sentence_scores[sent]=word_frequencies[word.text.lower()]
          else:
            sentence_scores[sent]+=word_frequencies[word.text.lower()]
#sentence_scores  
percent=int(input("How much percentage of summary you want? "))
ratio=(int(percent)) / 100
#ratio

from heapq import nlargest
select_length=int(len(sentence_tokens)*ratio)
select_length
summary=nlargest(select_length, sentence_scores,key=sentence_scores.get)
summary
     
final_summary=[word.text for word in summary]
final_summary
summary=''.join(final_summary)
summary

     

# TF-IDF Summary
import math
from nltk import sent_tokenize, word_tokenize, PorterStemmer
from nltk.corpus import stopwords

def _generate_summary(sentences, sentenceValue, threshold):
    sentence_count = 0
    summary = ''

    for sentence in sentences:
        if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= (threshold):
            summary += " " + sentence
            sentence_count += 1

    return summary

def _find_average_score(sentenceValue) -> int:
    """
    Find the average score from the sentence value dictionary
    :rtype: int
    """
    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]

    # Average value of a sentence from original summary_text
    average = (sumValues / len(sentenceValue))

    return average

def _score_sentences(tf_idf_matrix) -> dict:
    """
    score a sentence by its word's TF
    Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
    :rtype: dict
    """

    sentenceValue = {}

    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0

        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score

        sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence

    return sentenceValue

def _create_tf_idf_matrix(tf_matrix, idf_matrix):
    tf_idf_matrix = {}

    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

        tf_idf_table = {}

        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  # here, keys are the same in both the table
            tf_idf_table[word1] = float(value1 * value2)

        tf_idf_matrix[sent1] = tf_idf_table

    return tf_idf_matrix

def _create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
    idf_matrix = {}

    for sent, f_table in freq_matrix.items():
        idf_table = {}

        for word in f_table.keys():
            idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

        idf_matrix[sent] = idf_table

    return idf_matrix

def _create_documents_per_words(freq_matrix):
    word_per_doc_table = {}

    for sent, f_table in freq_matrix.items():
        for word, count in f_table.items():
            if word in word_per_doc_table:
                word_per_doc_table[word] += 1
            else:
                word_per_doc_table[word] = 1

    return word_per_doc_table

def _create_tf_matrix(freq_matrix):
    tf_matrix = {}

    for sent, f_table in freq_matrix.items():
        tf_table = {}

        count_words_in_sentence = len(f_table)
        for word, count in f_table.items():
            tf_table[word] = count / count_words_in_sentence

        tf_matrix[sent] = tf_table

    return tf_matrix

def _create_frequency_matrix(sentences):
    frequency_matrix = {}
    stopWords = set(stopwords.words("english"))
    ps = PorterStemmer()

    for sent in sentences:
        freq_table = {}
        words = word_tokenize(sent)
        for word in words:
            word = word.lower()
            word = ps.stem(word)
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:15]] = freq_table

    return frequency_matrix


#Get Key value from Dictionary
def get_key_from_dict(val,dic):
    key_list=list(dic.keys())
    val_list=list(dic.values())
    ind=val_list.index(val)
    return key_list[ind]

#Coreference Resolution
import nltk
from string import punctuation
from heapq import nlargest
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

def nltk_summarize(text_content, percent):
    tokens = word_tokenize(text_content)
    stop_words = stopwords.words('english')
    punctuation_items = punctuation + '\n'

    word_frequencies = {}
    for word in tokens:
        if word.lower() not in stop_words:
            if word.lower() not in punctuation_items:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1
    max_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency
    sentence_token = sent_tokenize(text_content)
    sentence_scores = {}
    for sent in sentence_token:
        sentence = sent.split(" ")
        for word in sentence:
            if word.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.lower()]

    select_length = int(len(sentence_token) * (int(percent) / 100))
    summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
    final_summary = [word for word in summary]
    summary = ' '.join(final_summary)
    return summary

#-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x

# Hide Streamlit Footer and buttons
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Adding logo for the App
file_ = open("app_logo.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.sidebar.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="" style="height:300px; width:400px;">',
    unsafe_allow_html=True,
)

# Input Video Link
url = st.sidebar.text_input('Enter YouTube video URL')
# Display Video and Title
if url.startswith("https://www.youtube.com/"):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")

    link = soup.find_all(name="title")[0]
    title = str(link)
    title = title.replace("<title>", "")
    title = title.replace("</title>", "")
    title = title.replace("&amp;", "&")

    value = title
    st.info("### " + value)
    st.video(url)
else:
    st.sidebar.error("Please enter a valid YouTube video URL")



#Specify Summarization type
sumtype = st.sidebar.selectbox(
     'Specify Summarization Type',
     options=['Extractive', 'Abstractive (T5 Algorithm)'])

#-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x
if sumtype == 'Extractive':
     
     # Specify the summarization algorithm
     sumalgo = st.sidebar.selectbox(
          'Select a Summarisation Algorithm',
          options=['Sumy', 'NLTK', 'Spacy', 'TF-IDF'])

     # Specify the summary length
     length = st.sidebar.select_slider(
          'Specify length of Summary',
          options=['10%', '20%', '30%', '40%', '50%'])

     # Select Language Preference
     languages_dict = {'en':'English' ,'af':'Afrikaans' ,'sq':'Albanian' ,'am':'Amharic' ,'ar':'Arabic' ,'hy':'Armenian' ,'az':'Azerbaijani' ,'eu':'Basque' ,'be':'Belarusian' ,'bn':'Bengali' ,'bs':'Bosnian' ,'bg':'Bulgarian' ,'ca':'Catalan' ,'ceb':'Cebuano' ,'ny':'Chichewa' ,'zh-cn':'Chinese (simplified)' ,'zh-tw':'Chinese (traditional)' ,'co':'Corsican' ,'hr':'Croatian' ,'cs':'Czech' ,'da':'Danish' ,'nl':'Dutch' ,'eo':'Esperanto' ,'et':'Estonian' ,'tl':'Filipino' ,'fi':'Finnish' ,'fr':'French' ,'fy':'Frisian' ,'gl':'Galician' ,'ka':'Georgian' ,'de':'German' ,'el':'Greek' ,'gu':'Gujarati' ,'ht':'Haitian creole' ,'ha':'Hausa' ,'haw':'Hawaiian' ,'he':'Hebrew' ,'hi':'Hindi' ,'hmn':'Hmong' ,'hu':'Hungarian' ,'is':'Icelandic' ,'ig':'Igbo' ,'id':'Indonesian' ,'ga':'Irish' ,'it':'Italian' ,'ja':'Japanese' ,'jw':'Javanese' ,'kn':'Kannada' ,'kk':'Kazakh' ,'km':'Khmer' ,'ko':'Korean' ,'ku':'Kurdish (kurmanji)' ,'ky':'Kyrgyz' ,'lo':'Lao' ,'la':'Latin' ,'lv':'Latvian' ,'lt':'Lithuanian' ,'lb':'Luxembourgish' ,'mk':'Macedonian' ,'mg':'Malagasy' ,'ms':'Malay' ,'ml':'Malayalam' ,'mt':'Maltese' ,'mi':'Maori' ,'mr':'Marathi' ,'mn':'Mongolian' ,'my':'Myanmar (burmese)' ,'ne':'Nepali' ,'no':'Norwegian' ,'or':'Odia' ,'ps':'Pashto' ,'fa':'Persian' ,'pl':'Polish' ,'pt':'Portuguese' ,'pa':'Punjabi' ,'ro':'Romanian' ,'ru':'Russian' ,'sm':'Samoan' ,'gd':'Scots gaelic' ,'sr':'Serbian' ,'st':'Sesotho' ,'sn':'Shona' ,'sd':'Sindhi' ,'si':'Sinhala' ,'sk':'Slovak' ,'sl':'Slovenian' ,'so':'Somali' ,'es':'Spanish' ,'su':'Sundanese' ,'sw':'Swahili' ,'sv':'Swedish' ,'tg':'Tajik' ,'ta':'Tamil' ,'te':'Telugu' ,'th':'Thai' ,'tr':'Turkish' ,'uk':'Ukrainian' ,'ur':'Urdu' ,'ug':'Uyghur' ,'uz':'Uzbek' ,'vi':'Vietnamese' ,'cy':'Welsh' ,'xh':'Xhosa' ,'yi':'Yiddish' ,'yo':'Yoruba' ,'zu':'Zulu'}
     add_selectbox = st.sidebar.selectbox(
         "Select Language",
         ( 'English' ,'Afrikaans' ,'Albanian' ,'Amharic' ,'Arabic' ,'Armenian' ,'Azerbaijani' ,'Basque' ,'Belarusian' ,'Bengali' ,'Bosnian' ,'Bulgarian' ,'Catalan' ,'Cebuano' ,'Chichewa' ,'Chinese (simplified)' ,'Chinese (traditional)' ,'Corsican' ,'Croatian' ,'Czech' ,'Danish' ,'Dutch' ,'Esperanto' ,'Estonian' ,'Filipino' ,'Finnish' ,'French' ,'Frisian' ,'Galician' ,'Georgian' ,'German' ,'Greek' ,'Gujarati' ,'Haitian creole' ,'Hausa' ,'Hawaiian' ,'Hebrew' ,'Hindi' ,'Hmong' ,'Hungarian' ,'Icelandic' ,'Igbo' ,'Indonesian' ,'Irish' ,'Italian' ,'Japanese' ,'Javanese' ,'Kannada' ,'Kazakh' ,'Khmer' ,'Korean' ,'Kurdish (kurmanji)' ,'Kyrgyz' ,'Lao' ,'Latin' ,'Latvian' ,'Lithuanian' ,'Luxembourgish' ,'Macedonian' ,'Malagasy' ,'Malay' ,'Malayalam' ,'Maltese' ,'Maori' ,'Marathi' ,'Mongolian' ,'Myanmar (burmese)' ,'Nepali' ,'Norwegian' ,'Odia' ,'Pashto' ,'Persian' ,'Polish' ,'Portuguese' ,'Punjabi' ,'Romanian' ,'Russian' ,'Samoan' ,'Scots gaelic' ,'Serbian' ,'Sesotho' ,'Shona' ,'Sindhi' ,'Sinhala' ,'Slovak' ,'Slovenian' ,'Somali' ,'Spanish' ,'Sundanese' ,'Swahili' ,'Swedish' ,'Tajik' ,'Tamil' ,'Telugu' ,'Thai' ,'Turkish' ,'Ukrainian' ,'Urdu' ,'Uyghur' ,'Uzbek' ,'Vietnamese' ,'Welsh' ,'Xhosa' ,'Yiddish' ,'Yoruba' ,'Zulu')
     )
     
     # If Summarize button is clicked
     if st.sidebar.button('Summarize'):
         st.success(dedent("""### \U0001F4D6 Summary
     > Success!
         """))

         # Generate Transcript by slicing YouTube link to id 
         url_data = urlparse(url)
         id = url_data.query[2::]

         def generate_transcript(id):
                 transcript = YouTubeTranscriptApi.get_transcript(id)
                 script = ""

                 for text in transcript:
                         t = text["text"]
                         if t != '[Music]':
                                 script += t + " "

                 return script, len(script.split())
         transcript, no_of_words = generate_transcript(id)

        
         model = T5ForConditionalGeneration.from_pretrained("t5-base")
         tokenizer = T5Tokenizer.from_pretrained("t5-base")
         inputs = tokenizer.encode("summarize: " + transcript, return_tensors="pt",max_length= 512, truncation=True)
          

         outputs = model.generate(
              inputs, 
              max_length=150, 
              min_length=40, 
              length_penalty=2.0, 
              num_beams=4, 
              early_stopping=True)
          
         summ = tokenizer.decode(outputs[0])

         # Translate and Print Summary
         translated = GoogleTranslator(source='auto', target= get_key_from_dict(add_selectbox,languages_dict)).translate(summ)
         html_str3 = f"""
         

<style>
p.a {{
text-align: justify;
}}
</style>
<p class="a">{translated}</p>
"""
         st.markdown(html_str3, unsafe_allow_html=True)

         # Generate Audio
         st.success("###  \U0001F3A7 Hear your Summary")
         no_support = ['Amharic', 'Azerbaijani', 'Basque', 'Belarusian', 'Cebuano', 'Chichewa', 'Chinese (simplified)', 'Chinese (traditional)', 'Corsican', 'Frisian', 'Galician', 'Georgian', 'Haitian creole', 'Hausa', 'Hawaiian', 'Hmong', 'Igbo', 'Irish', 'Kazakh', 'Kurdish (kurmanji)', 'Kyrgyz', 'Lao', 'Lithuanian', 'Luxembourgish', 'Malagasy', 'Maltese', 'Maori', 'Mongolian', 'Odia', 'Pashto', 'Persian', 'Punjabi', 'Samoan', 'Scots gaelic', 'Sesotho', 'Shona', 'Sindhi', 'Slovenian', 'Somali', 'Tajik', 'Uyghur', 'Uzbek', 'Xhosa', 'Yiddish', 'Yoruba', 'Zulu']
         if add_selectbox in no_support:
             st.warning(" \U000026A0 \xa0 Audio Support for this language is currently unavailable\n")
             lang_warn = GoogleTranslator(source='auto', target= get_key_from_dict(add_selectbox,languages_dict)).translate("\U000026A0 \xa0 Audio Support for this language is currently unavailable")
             st.warning(lang_warn)
         else:
             speech = gTTS(text = translated,lang=get_key_from_dict(add_selectbox,languages_dict), slow = False)
             speech.save('user_trans.mp3')          
             audio_file = open('user_trans.mp3', 'rb')    
             audio_bytes = audio_file.read()    
             st.audio(audio_bytes, format='audio/ogg',start_time=0)


#-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x

elif sumtype == 'Abstractive (T5 Algorithm)':
     
     # Select Language Preference
     languages_dict = {'en':'English' ,'af':'Afrikaans' ,'sq':'Albanian' ,'am':'Amharic' ,'ar':'Arabic' ,'hy':'Armenian' ,'az':'Azerbaijani' ,'eu':'Basque' ,'be':'Belarusian' ,'bn':'Bengali' ,'bs':'Bosnian' ,'bg':'Bulgarian' ,'ca':'Catalan' ,'ceb':'Cebuano' ,'ny':'Chichewa' ,'zh-cn':'Chinese (simplified)' ,'zh-tw':'Chinese (traditional)' ,'co':'Corsican' ,'hr':'Croatian' ,'cs':'Czech' ,'da':'Danish' ,'nl':'Dutch' ,'eo':'Esperanto' ,'et':'Estonian' ,'tl':'Filipino' ,'fi':'Finnish' ,'fr':'French' ,'fy':'Frisian' ,'gl':'Galician' ,'ka':'Georgian' ,'de':'German' ,'el':'Greek' ,'gu':'Gujarati' ,'ht':'Haitian creole' ,'ha':'Hausa' ,'haw':'Hawaiian' ,'he':'Hebrew' ,'hi':'Hindi' ,'hmn':'Hmong' ,'hu':'Hungarian' ,'is':'Icelandic' ,'ig':'Igbo' ,'id':'Indonesian' ,'ga':'Irish' ,'it':'Italian' ,'ja':'Japanese' ,'jw':'Javanese' ,'kn':'Kannada' ,'kk':'Kazakh' ,'km':'Khmer' ,'ko':'Korean' ,'ku':'Kurdish (kurmanji)' ,'ky':'Kyrgyz' ,'lo':'Lao' ,'la':'Latin' ,'lv':'Latvian' ,'lt':'Lithuanian' ,'lb':'Luxembourgish' ,'mk':'Macedonian' ,'mg':'Malagasy' ,'ms':'Malay' ,'ml':'Malayalam' ,'mt':'Maltese' ,'mi':'Maori' ,'mr':'Marathi' ,'mn':'Mongolian' ,'my':'Myanmar (burmese)' ,'ne':'Nepali' ,'no':'Norwegian' ,'or':'Odia' ,'ps':'Pashto' ,'fa':'Persian' ,'pl':'Polish' ,'pt':'Portuguese' ,'pa':'Punjabi' ,'ro':'Romanian' ,'ru':'Russian' ,'sm':'Samoan' ,'gd':'Scots gaelic' ,'sr':'Serbian' ,'st':'Sesotho' ,'sn':'Shona' ,'sd':'Sindhi' ,'si':'Sinhala' ,'sk':'Slovak' ,'sl':'Slovenian' ,'so':'Somali' ,'es':'Spanish' ,'su':'Sundanese' ,'sw':'Swahili' ,'sv':'Swedish' ,'tg':'Tajik' ,'ta':'Tamil' ,'te':'Telugu' ,'th':'Thai' ,'tr':'Turkish' ,'uk':'Ukrainian' ,'ur':'Urdu' ,'ug':'Uyghur' ,'uz':'Uzbek' ,'vi':'Vietnamese' ,'cy':'Welsh' ,'xh':'Xhosa' ,'yi':'Yiddish' ,'yo':'Yoruba' ,'zu':'Zulu'}
     add_selectbox = st.sidebar.selectbox(
         "Select Language",
         ( 'English' ,'Afrikaans' ,'Albanian' ,'Amharic' ,'Arabic' ,'Armenian' ,'Azerbaijani' ,'Basque' ,'Belarusian' ,'Bengali' ,'Bosnian' ,'Bulgarian' ,'Catalan' ,'Cebuano' ,'Chichewa' ,'Chinese (simplified)' ,'Chinese (traditional)' ,'Corsican' ,'Croatian' ,'Czech' ,'Danish' ,'Dutch' ,'Esperanto' ,'Estonian' ,'Filipino' ,'Finnish' ,'French' ,'Frisian' ,'Galician' ,'Georgian' ,'German' ,'Greek' ,'Gujarati' ,'Haitian creole' ,'Hausa' ,'Hawaiian' ,'Hebrew' ,'Hindi' ,'Hmong' ,'Hungarian' ,'Icelandic' ,'Igbo' ,'Indonesian' ,'Irish' ,'Italian' ,'Japanese' ,'Javanese' ,'Kannada' ,'Kazakh' ,'Khmer' ,'Korean' ,'Kurdish (kurmanji)' ,'Kyrgyz' ,'Lao' ,'Latin' ,'Latvian' ,'Lithuanian' ,'Luxembourgish' ,'Macedonian' ,'Malagasy' ,'Malay' ,'Malayalam' ,'Maltese' ,'Maori' ,'Marathi' ,'Mongolian' ,'Myanmar (burmese)' ,'Nepali' ,'Norwegian' ,'Odia' ,'Pashto' ,'Persian' ,'Polish' ,'Portuguese' ,'Punjabi' ,'Romanian' ,'Russian' ,'Samoan' ,'Scots gaelic' ,'Serbian' ,'Sesotho' ,'Shona' ,'Sindhi' ,'Sinhala' ,'Slovak' ,'Slovenian' ,'Somali' ,'Spanish' ,'Sundanese' ,'Swahili' ,'Swedish' ,'Tajik' ,'Tamil' ,'Telugu' ,'Thai' ,'Turkish' ,'Ukrainian' ,'Urdu' ,'Uyghur' ,'Uzbek' ,'Vietnamese' ,'Welsh' ,'Xhosa' ,'Yiddish' ,'Yoruba' ,'Zulu')
     )
     
     #If summarize button is clicked
     if st.sidebar.button('Summarize'):
          st.success(dedent("""### \U0001F4D6 Summary
> Success!
    """))
    

          # Generate Transcript by slicing YouTube link to id 
          url_data = urlparse(url)
          id = url_data.query[2::]

          def generate_transcript(id):
               transcript = YouTubeTranscriptApi.get_transcript(id)
               script = ""

               for text in transcript:
                    t = text["text"]
                    if t != '[Music]':
                         script += t + " "

               return script, len(script.split())
          transcript, no_of_words = generate_transcript(id)

          
          model = T5ForConditionalGeneration.from_pretrained("t5-base")
          tokenizer = T5Tokenizer.from_pretrained("t5-base")
          inputs = tokenizer.encode("summarize: " + transcript, return_tensors="pt",max_length=512, truncation=True)
          
          outputs = model.generate(
              inputs, 
              max_length=150, 
              min_length=40, 
              length_penalty=2.0, 
              num_beams=4, 
              early_stopping=True)
          
          summ = tokenizer.decode(outputs[0])
          
          # Translate and Print Summary
          translated = GoogleTranslator(source='auto', target= get_key_from_dict(add_selectbox,languages_dict)).translate(summ)
          html_str3 = f"""


<style>
p.a {{
text-align: justify;
}}
</style>
<p class="a">{translated}</p>
"""
          st.markdown(html_str3, unsafe_allow_html=True)

          # Generate Audio
          st.success("###  \U0001F3A7 Hear your Summary")
          no_support = ['Amharic', 'Azerbaijani', 'Basque', 'Belarusian', 'Cebuano', 'Chichewa', 'Chinese (simplified)', 'Chinese (traditional)', 'Corsican', 'Frisian', 'Galician', 'Georgian', 'Haitian creole', 'Hausa', 'Hawaiian', 'Hmong', 'Igbo', 'Irish', 'Kazakh', 'Kurdish (kurmanji)', 'Kyrgyz', 'Lao', 'Lithuanian', 'Luxembourgish', 'Malagasy', 'Maltese', 'Maori', 'Mongolian', 'Odia', 'Pashto', 'Persian', 'Punjabi', 'Samoan', 'Scots gaelic', 'Sesotho', 'Shona', 'Sindhi', 'Slovenian', 'Somali', 'Tajik', 'Uyghur', 'Uzbek', 'Xhosa', 'Yiddish', 'Yoruba', 'Zulu']
          if add_selectbox in no_support:
              st.warning(" \U000026A0 \xa0 Audio Support for this language is currently unavailable\n")
              lang_warn = GoogleTranslator(source='auto', target= get_key_from_dict(add_selectbox,languages_dict)).translate("\U000026A0 \xa0 Audio Support for this language is currently unavailable")
              st.warning(lang_warn)
          else:
              speech = gTTS(text = translated,lang=get_key_from_dict(add_selectbox,languages_dict), slow = False)
              speech.save('user_trans.mp3')          
              audio_file = open('user_trans.mp3', 'rb')    
              audio_bytes = audio_file.read()    
              st.audio(audio_bytes, format='audio/ogg',start_time=0)
              

#-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x