import streamlit as st
import pandas as pd
import altair as alt
from ScriptXLM_RoBERTa import predict_with_loaded_model

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
        st.session_state.subject_input = ""
    if "objection_input" not in st.session_state:
        st.session_state.objection_input = ""

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            # Use the session state directly for the input widgets
            id_input = st.text_input("ID of Objection*", value=st.session_state.id_input, key="id_input")
            subject_input = st.text_input("Subject*", value=st.session_state.subject_input, key="subject_input")
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

def result_page():
    st.title("Result")

    # Get the result from session state
    result = st.session_state.get("result", ("No result available", [0, 0]))
    predicted_label, laws, probabilities = result

    # Map for labels
    label_map = {0: "Ongegrond", 1: "Gegrond"}

    # Display the result with the mapped label
    if isinstance(predicted_label, int):
        label_text = label_map.get(predicted_label, "Unknown")
        st.write(f"Result: {label_text}")
        prob_text = probabilities[predicted_label]
        st.write(f"Probability: {prob_text}")
    else:
        st.write("No valid result available")

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Objection ID", value=st.session_state.get("id_input", ""), disabled=True)
            st.text_input("Subject", value=st.session_state.get("subject_input", ""), disabled=True)
            st.text_area("Objection", value=st.session_state.get("objection_input", ""), disabled=True)
        with col2:
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
                    alt.value('green'),
                    alt.value('red')
                )
            )
            st.altair_chart(chart, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Restart", key="quit_button_result", use_container_width=True):
                st.session_state.page = "Landing"
        with col2:
            if st.button("Save", key="save_button", use_container_width=True):
                st.write("Result saved!")


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
