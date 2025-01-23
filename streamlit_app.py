import streamlit as st
import pandas as pd
import altair as alt
from ScriptXLM_RoBERTa import predict_with_loaded_model, predict_with_top_5_laws, predict_with_top_5_words_and_sentences
from io import BytesIO
from docx import Document
import os
from git import Repo
import tempfile

def hide_streamlit_menu_footer():
    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stButton>button {
            width: 100%;
            margin: 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def style_buttons():
    st.markdown(
        """
        <style>
        .red-button {
            background-color: #ff4b4b !important;
            color: white !important;
            padding: 15px;
            border-radius: 5px;
        }
        .green-button {
            background-color: #28a745 !important;
            color: white !important;
            padding: 15px;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Example usage
def landing_page():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("Objection Helper")
        st.markdown("Welcome to the Objection Helper. Click start to proceed.")
    with col2:
        st.image("images/GASD_1.png", use_container_width=True)  # Use the correct parameter
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        # Using a green button for Start
        if st.button("Start", key="start_button", help="Click to start the process", use_container_width=True):
            st.session_state.page = "Submit Information"  # Set the session state to navigate to the second page

def submit_information_page():
    st.title("Submit Information")

    # Set default values for session state variables if not already set
    if "id_input" not in st.session_state:
        st.session_state.id_input = ""
    if "subject_input" not in st.session_state:
        st.session_state.subject_input = "Waste Fines"  # Default value for the dropdown
    if "objection_input" not in st.session_state:
        st.session_state.objection_input = ""

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            # Use the session state directly for the input widgets
            id_input = st.text_input("ID of Objection*", value=st.session_state.id_input, key="id_input")
            
            # Change subject_input to a dropdown (selectbox)
            subject_input = st.selectbox(
                "Subject*",
                ["Waste Fines", "Vehicle Towing"],
                index=["Waste Fines", "Vehicle Towing"].index(st.session_state.subject_input),  # Default selection
                key="subject_input"
            )

            # Keep the text area for the objection
            objection_input = st.text_area("Objection*", value=st.session_state.objection_input, key="objection_input")
            
        with col2:
            st.image("images/GASD_1.png", use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Quit", key="quit_button", use_container_width=True):
                st.session_state.page = "Landing"
        with col2:
            if st.button("Proceed", key="proceed_button", use_container_width=True):
                if objection_input:  # Ensure there is input before proceeding
                    prediction = predict_with_loaded_model(objection_input)
                    st.session_state.result = prediction
                    st.session_state.page = "Result"
                else:
                    st.error("Please enter the objection details before proceeding.")

# Function to compute highlighted objection
def compute_highlighted_objection(top_3_sentences):
    highlighted_objection = st.session_state.get("objection_input", "")
    for sentence, score in top_3_sentences:
        highlighted_objection = highlighted_objection.replace(
            sentence, f'<span class="highlight">{sentence}</span>'
        )
    return highlighted_objection

def result_page():
    st.title("Result")

    # Get the result from session state
    result = st.session_state.get("result", ("No result available", [0, 0]))
    predicted_label, laws, probabilities = result
    objection_input = st.session_state.get("objection_input", "")

    # Map for labels
    label_map = {0: "Ongegrond", 1: "Gegrond"}

    # Predict with top 5 laws and words
    dictum_prediction, top_5_laws = predict_with_top_5_laws(objection_input)
    dictum_prediction_words, top_5_laws_2, top_5_words, top_3_sentences = predict_with_top_5_words_and_sentences(objection_input)

   # Inside result_page
    if 'top_5_laws' not in st.session_state:
        st.session_state.top_5_laws = top_5_laws
    if 'top_5_words' not in st.session_state:
        st.session_state.top_5_words = top_5_words



    # Label mapping
    label_text = label_map.get(dictum_prediction, "Unknown")
    
    # Display prediction and details
    if isinstance(predicted_label, int):
        label_text = label_map.get(predicted_label, "Unknown")
        st.subheader(f"Prediction: {label_text}")
        prob_text = probabilities[predicted_label]
        st.write(f"Probability: {prob_text}")
    else:
        st.write("No valid result available")

    id_input = st.text_input("Objection ID", value=st.session_state.id_input, key="id_input", disabled=True)
    subject_input = st.text_input("Subject", value=st.session_state.subject_input, key="subject_input", disabled=True)
    st.write("Objection")
    # Check if the highlighted objection is already stored in session_state
    if 'highlighted_objection' not in st.session_state:
        # Compute and store the highlighted objection in session_state if it doesn't exist
        st.session_state.highlighted_objection = compute_highlighted_objection(top_3_sentences)

    # Apply dynamic CSS for light/dark mode
    st.markdown(
        """
        <style>
        .highlight {
            background-color: #FFD700;  /* Default Yellow for Light Mode */
            padding: 0 5px;
        }
        @media (prefers-color-scheme: dark) {
            .highlight {
                background-color: #FFA500;  /* Orange for Dark Mode */
                color: black;  /* Adjust Text Color for Readability */
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display the highlighted objection text using st.markdown from session_state
    st.markdown(f"<p>{st.session_state.highlighted_objection}</p>", unsafe_allow_html=True)

    # Display Top 5 Laws
    st.subheader("Top 5 Relevant Laws")
    laws_df = pd.DataFrame(st.session_state.top_5_laws, columns=["Law", "Score"])
    laws_chart = alt.Chart(laws_df).mark_bar().encode(
        x=alt.X("Score", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("Law", sort="-x"),
        color=alt.value("blue")
    )
    st.altair_chart(laws_chart, use_container_width=True)

    # Display Top 5 Words
    st.subheader("Top 5 Words Contributing to the Prediction")
    words_df = pd.DataFrame(st.session_state.top_5_words[:5], columns=["Word", "Score"])
    words_chart = alt.Chart(words_df).mark_bar().encode(
        x=alt.X("Score", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("Word", sort="-x"),
        color=alt.condition(
            alt.datum.Score > 0,
            alt.value("green"),
            alt.value("red")
        )
    )
    st.altair_chart(words_chart, use_container_width=True)

    # Display Sentence Scores
    # st.subheader("Sentence-Level Scores")
    # if top_3_sentences:
    #     for sentence, score in top_3_sentences:  # Loop through all sentence-score pairs
    #         st.markdown(f"- **{sentence}**: {score:.4f}")
    # else:
    #     st.write("No sentences available for display.")

    # Create a Word document with the result details
    def create_result_doc():
        doc = Document()
        
        # Add basic information to the document
        doc.add_heading("Objection Details", 0)
        doc.add_paragraph(f"Objection ID: {st.session_state.get('id_input', '')}")
        doc.add_paragraph(f"Subject: {st.session_state.get('subject_input', '')}")
        doc.add_paragraph(f"Objection: {st.session_state.get('objection_input', '')}")
        
        # Add prediction details
        doc.add_heading("Prediction", level=1)
        doc.add_paragraph(f"Prediction: {label_text}")
        doc.add_paragraph(f"Probability: {prob_text}")
        
        # Add Top 5 Laws
        doc.add_heading("Top 5 Relevant Laws", level=1)
        for law, score in top_5_laws:
            doc.add_paragraph(f"{law}: {score}")
        
        # Add Top 5 Words
        doc.add_heading("Top 5 Words Contributing to the Prediction", level=1)
        for word, score in top_5_words[:5]:
            doc.add_paragraph(f"{word}: {score}")

        # Add Sentence-Level Scores
        doc.add_heading("Sentence-Level Scores", level=1)
        if top_3_sentences:
            for sentence, score in top_3_sentences:
                doc.add_paragraph(f"{sentence}: {score:.4f}")
        else:
            doc.add_paragraph("No sentences available.")

        # Save the document in memory
        doc_stream = BytesIO()
        doc.save(doc_stream)
        doc_stream.seek(0)
        
        return doc_stream

    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Restart", key="quit_button_result", use_container_width=True):
            st.session_state.clear()
            st.session_state.page = "Landing"
    with col2:
        st.download_button(
        label="Download Result",
        use_container_width=True,
        key="download_button_result",
        data=create_result_doc(),
        file_name="result.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Landing"

# Apply CSS
hide_streamlit_menu_footer()
style_buttons()

# Page routing
if st.session_state.page == "Landing":
    landing_page()
elif st.session_state.page == "Submit Information":
    submit_information_page()
elif st.session_state.page == "Result":
    result_page()
