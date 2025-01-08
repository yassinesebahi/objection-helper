import streamlit as st
import pandas as pd
import altair as alt

# Function to display the Landing Page
def landing_page():
    st.title("Gemeente Amsterdam")
    st.header("Objection Helper")
    if st.button("Start"):
        st.session_state.page = "Submit Information"

# Function to display the Submit Information Page
def submit_information_page():
    st.title("Submit Information")
    id_input = st.text_input("ID of Objection*")
    subject_input = st.text_input("Subject*")
    objection_input = st.text_area("Objection*")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Quit"):
            st.session_state.page = "Landing"
    with col2:
        if st.button("Proceed"):
            st.session_state.page = "Result"
            st.session_state.result = {
                "ID": id_input,
                "Subject": subject_input,
                "Objection": objection_input
            }

# Function to display the Result Page
def result_page():
    st.title("Result")
    st.write("Result: Gegrond")

    st.text_input("Objection ID", value=st.session_state.result["ID"], disabled=True)
    st.text_input("Subject", value=st.session_state.result["Subject"], disabled=True)
    st.text_area("Objection", value=st.session_state.result["Objection"], disabled=True)

    st.subheader("Top 5 Important Words for Prediction")
    data = pd.DataFrame({
        'Word': ['vergunning', 'ontheffing', 'wijziging', 'voorwaarden', 'herziening'],
        'Importance Score': [0.25, -0.15, 0.1, 0.05, -0.05]
    })
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('Importance Score', scale=alt.Scale(domain=(-0.3, 0.3))),
        y='Word',
        color=alt.condition(
            alt.datum['Importance Score'] > 0,
            alt.value('green'),  # Positive values are green
            alt.value('red')  # Negative values are red
        )
    )
    st.altair_chart(chart, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Quit"):
            st.session_state.page = "Landing"
    with col2:
        if st.button("Save"):
            st.write("Result saved!")

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Landing"

# Page routing
if st.session_state.page == "Landing":
    landing_page()
elif st.session_state.page == "Submit Information":
    submit_information_page()
elif st.session_state.page == "Result":
    result_page()
