import streamlit as st
import pandas as pd
import altair as alt

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
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            id_input = st.text_input("ID of Objection*")
            subject_input = st.text_input("Subject*")
            objection_input = st.text_area("Objection*")
        with col2:
            st.image("images/GASD_1.png", use_container_width=True)  # Use the correct parameter
        
        col1, col2 = st.columns(2)
        with col1:
            # Styled red "Quit" button
            if st.button("Quit", key="quit_button", use_container_width=True):
                st.session_state.page = "Landing"
        with col2:
            # Styled green "Proceed" button
            if st.button("Proceed", key="proceed_button", use_container_width=True):
                st.session_state.page = "Result"
                st.session_state.result = {
                    "ID": id_input,
                    "Subject": subject_input,
                    "Objection": objection_input
                }

def result_page():
    st.title("Result")
    st.write("Result: Gegrond")
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Objection ID", value="12345", disabled=True)
            st.text_input("Subject", value="Subject Example", disabled=True)
            st.text_area("Objection", value="Objection details go here.", disabled=True)
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
                    alt.value('green'),  # Positive values are green
                    alt.value('red')  # Negative values are red
                )
            )
            st.altair_chart(chart, use_container_width=True)
            

        col1, col2 = st.columns(2)
        with col1:
            # Styled red "Quit" button
            if st.button("Quit", key="quit_button_result", use_container_width=True):
                st.session_state.page = "Landing"
        with col2:
            # Styled green "Save" button
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
