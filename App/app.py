import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import neattext.functions as nfx
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import joblib
from threading import Thread
from streamlit.runtime.scriptrunner import add_script_run_ctx
import altair as alt
import datetime
import base64
from sklearn.metrics import accuracy_score


# Xtest = pd.read_csv('Models\emotion_det_X_test.csv')
# Ytest = pd.read_csv('Models\emotion_det_Y_test.csv')
pipe_logreg = joblib.load(open('Models\emotion_det_lr_pipe.pkl','rb'))

def predict_emotions(docx):
    results = pipe_logreg.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_logreg.predict_proba([docx])
    return results

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )



# def monitor(X_test, Y_test):
#     Y_pred = pipe_logreg.predict(X_test['Cleaned_Text'], errors = 'ignore')
#     results = accuracy_score(Y_test,Y_pred)
#     return results



def main():
    st.title("Text Emotion Detector", )
    menu = ["Home", "About"]
    add_bg_from_local('Emoji.jpg') 
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        st.markdown(f'<h1 style="color:brown;font-size:25px;"> Home - Emotion Detector </h1>', unsafe_allow_html=True)
        # st.subheader("Home - Emotion Detector")
        with st.form(key = 'emotion_clf_form'):
            raw_text = st.text_area("Text Here")
            submit_text = st.form_submit_button(label='Submit')
        
        if submit_text:
            col1, col2 = st.columns(2)
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                # st.success("Original Text")
                st.markdown(f'<h1 style="color:black;font-size:18px;"> Text Emotion: </h1>', unsafe_allow_html=True)
                st.markdown(f'<h1 style="color:Brown;font-size:20px;">  Text : {raw_text}</h1>', unsafe_allow_html=True)
                # st.write(raw_text)
                txt = "Emotion "
                emoji_icon = emotions_emoji_dict[prediction]
                # st.write(f"{prediction}:{emoji_icon}",unsafe_allow_html=True, style="color: red")
                st.markdown(f'<h1 style="color:#2E86C1;font-size:18px;">{txt}  -  {prediction} : {emoji_icon}</h1>', unsafe_allow_html=True)
                st.markdown(f'<h1 style="color:#2E86C1;font-size:18px;"> Confidence: {np.max(probability)}</h1>', unsafe_allow_html=True)
                # st.write(f"Confidence: {np.max(probability)}")

            with col2:
                # st.success("Prediction Probability")
                st.markdown(f'<h1 style="color:black;font-size:18px;"> Prediction Probability Chart: </h1>', unsafe_allow_html=True)
                proba_df = pd.DataFrame(probability,columns=pipe_logreg.classes_)
                proba_df_set = proba_df.T.reset_index()
                proba_df_set.columns = ["emotions", "probability"]
                fig = alt.Chart(proba_df_set).mark_bar().encode(x= "emotions", y="probability", color ='emotions')
                st.altair_chart(fig, use_container_width=True)

    # elif choice == "Monitor":
    #     st.subheader("Monitor App")
    #     score = monitor(Xtest,Ytest)
    #     st.markdown(f'<h1 style="color:black;font-size:18px;"> {score} </h1>', unsafe_allow_html=True)


    else:
        # st.subheader("About")
        st.markdown(f'<h1 style="color:brown;font-size:25px;"> About </h1>', unsafe_allow_html=True)
        # st.write("It is a Text Emotion Classifier with Emoji",datetime.date.today())
        st.markdown(f'<h1 style="color:black;font-size:18px;"> Text Emotion Classifier with Emoji , 14 Aug 2023 </h1>', unsafe_allow_html=True)

t = Thread(target=main())
add_script_run_ctx(t)

if __name__ == "main":
    t.start()
