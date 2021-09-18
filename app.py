#streamlit run dashboard.py --server.port 4000 --server.address 0.0.0.0 --server.baseUrlPath test
import streamlit as st
import plotly.graph_objects as go
from annotated_text import annotated_text
import json
# import SessionState
from transformers.pipelines import Conversation
import pandas as pd
import numpy as np
import tensorflow as tf
import torch as pt
import logging
import json
from datetime import datetime


class JsonFormatter(logging.Formatter):

    def format(self, record):
        return json.dumps({"time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "name": record.name,
                    "level": record.levelname, "message": record.msg})

@st.cache(allow_output_mutation=True)
def get_logger():
    logger = logging.getLogger('allennlp')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('allennlp.log')
    #formatter = logging.Formatter('time="%(asctime)s" name=%(name)s level=%(levelname)s option="%(option)s" message="%(message)s"')
    fh.setFormatter(JsonFormatter())
    logger.addHandler(fh)
    return logger


query_params = st.experimental_get_query_params()
app_state = st.experimental_get_query_params()

# session_state = SessionState.get(first_query_params=query_params, conversation=Conversation())
# first_query_params = session_state.first_query_params

options = ["Welcome", "Criticality Detection"]

# default = options.index(first_query_params["option"][0]) if "option" in app_state else 0

option = st.sidebar.radio(
        'Select area of interest',
        options)#, index=default)

app_state["option"] = option
st.experimental_set_query_params(**app_state)

if option == "Welcome":
    '''
    # Welcome to Justice Served Team
    - [Criticality Detection](https://nlp.novartis.net/?option=Criticality+Detection)

    [source code of this booth](path to be given)
    `  `  
    `  `  
    `  `  
    '''
    html_string = '''
                    <p style="font-size:80%;">
                        Primary Contributors in order of contribution:<br>
                        <a href="mailto:"></a><br>
                        <a href="mailto:"></a><br>
                        <a href="mailto:"></a><br>
                        <a href="mailto:"></a>
                    </p>
                    '''

    st.markdown(html_string, unsafe_allow_html=True)

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TFAutoModelForSequenceClassification,\
    TFAutoModelForTokenClassification, AutoModelForQuestionAnswering, TFAutoModelForQuestionAnswering, TFAutoModelWithLMHead, \
        TFAutoModelForCausalLM, TFGPT2LMHeadModel, TFT5ForConditionalGeneration, \
        TFRobertaForMaskedLM, AutoModelForTokenClassification, AutoModelWithLMHead
from transformers import pipeline


if option == "Criticality Detection":
    '''
    ## Criticality Detection
    '''
    @st.cache(suppress_st_warning=True, allow_output_mutation=True)
    def get_sentiment_classifier():
        tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment", cache_dir="/data/nlp_booth/cache")
        model = TFAutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment", from_pt=True, cache_dir="/data/nlp_booth/cache")
        classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
        return classifier

    @st.cache
    def get_sentiment(text):
        sentiment_classifier = get_sentiment_classifier()
        return sentiment_classifier(text)

    sentiment_example = st.radio(
            'Select example text (English, Dutch, German, French, Spanish)',
            ("This movie doesn't care about cleverness, wit or any other kind of intelligent humor.",
            "Sci-fi manhunt, via Ridley Scott. Formulaic but great-looking. A classic now.",
            "In seiner früheren Fassung war der Film ein Meisterwerk mit Fehlern; in Scotts restaurierter Fassung ist er einfach ein Meisterwerk."))
    sentiment_text = st.text_input('Enter text to analyze', sentiment_example, key="sentiment")
    if sentiment_text:
        res = get_sentiment(sentiment_text)
        i = int(res[0]["label"][0])
        color = "lightgreen"
        if i==1: 
            color = "red"
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = i,
            title = {'text': "Sentiment"},
            gauge = {'axis': {'range': [0, 1]},
                    'bar': {'color': color}}
                    ))
        st.plotly_chart(fig, use_container_width=True)
        '''
        `  `  
        **Did not get what you were expecting?**

        The possible reasons could be:
        - Out of Domain Inference - The text which you have typed is out of domain with training data. The model may not give accurate outcome when domain changes.
        - Annotated Differently - Similar text in the training data could have been annotated differently.
        - Gray Zone - The model predicts with slightely lower confidence on the correct class.
        '''
