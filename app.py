import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import pickle

pipe_lr = pickle.load(open('text_emotion.pkl','rb'))

emotions_emoji_dict = {'anger': "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}



def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_probab(docx):
    results = pipe_lr.predict_proba([docx])
    return results


st.title('Text Emotion Detection')
st.subheader('Detect Emotions in Text')
with st.form(key='my_form'):
    raw_text= st.text_area('Type Here')
    submit_text = st.form_submit_button(label='Submit')

if submit_text:
    col1,col2=st.columns(2)
    prediction = predict_emotions(raw_text)
    probability = get_prediction_probab(raw_text)

    with col1:
        st.success('Original Text')
        st.write(raw_text)
        st.success('Prediction')
        emoji_icon = emotions_emoji_dict[prediction]
        st.write('{}:{}'.format(prediction,emoji_icon))
        st.write('Confidence:{}'.format(np.max(probability)))

    with col2:
        st.success('Prediction Probability')
        probe_df = pd.DataFrame(probability,columns= pipe_lr.classes_)
        probe_df_clean= probe_df.T.reset_index()
        probe_df_clean.columns = ['emotion','probability']
        fig = alt.Chart(probe_df_clean).mark_bar().encode(x='emotion',y='probability',color='emotion')
        st.altair_chart(fig,use_container_width=True)
             
