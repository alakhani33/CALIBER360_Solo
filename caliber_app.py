import streamlit as st
import pandas as pd
from datetime import datetime
import re
import matplotlib.pyplot as plt
import seaborn as sns
# from langchain_openai import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from PIL import Image
import os
from langchain.schema import HumanMessage



# from dotenv import load_dotenv
# load_dotenv()


from reportlab.lib.pagesizes import LETTER
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

import re

def clean_markdown(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)   # remove bold
    text = re.sub(r'__([^_]+)__', r'\1', text)     # remove underline
    text = re.sub(r'#+ ', '', text)               # remove headings
    text = re.sub(r'^\s*[-*]\s+', '• ', text, flags=re.MULTILINE)  # bullets
    text = re.sub(r'<span style="[^"]*">', '', text)  # remove span styles
    text = text.replace('</span>', '')  # remove closing span
    return text.replace("\n\n", "<br/><br/>").replace("\n", "<br/>")  # line breaks

# def clean_markdown(text):
#     text = re.sub(r'<span style="[^"]*">', '', text)  # remove span styles
#     text = text.replace('</span>', '')  # remove closing span
#     return text
def add_footer(canvas, doc):
    footer_text = "© 2025 Dr. M.A. Lakhani | CALIBER Leadership Inventory"
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.drawCentredString(4.25 * inch, 0.5 * inch, footer_text)
    canvas.restoreState()

from utils_orig import get_openai_api_key
from fpdf import FPDF
import streamlit as st

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

folder_id = "1Vnm_oKNaYjWVB95SSEg8hqiwDmclY3H9"

from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gdrive"],
    scopes=["https://www.googleapis.com/auth/drive"]
)

# from google.oauth2 import service_account
# from googleapiclient.discovery import build
# from googleapiclient.http import MediaFileUpload

# def upload_to_drive(file_path, file_name, mime_type, folder_id):
#     # Sanity check: file must exist
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"{file_path} does not exist.")

#     credentials = service_account.Credentials.from_service_account_info(
#         st.secrets["gdrive"],
#         scopes=["https://www.googleapis.com/auth/drive"]
#     )
#     service = build("drive", "v3", credentials=credentials)

#     try:
#         file_metadata = {
#             "name": file_name,
#             "parents": [folder_id]
#         }

#         media = MediaFileUpload(file_path, mimetype=mime_type, resumable=False)

#         uploaded_file = service.files().create(
#             body=file_metadata,
#             media_body=media,
#             fields="id"
#         ).execute()

#         return uploaded_file.get("id")

#     except Exception as e:
#         raise RuntimeError(f"Drive upload failed: {e}")


def upload_to_drive(file_path, file_name, mime_type, folder_id):
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gdrive"],
        scopes=["https://www.googleapis.com/auth/drive"]
    )
    service = build("drive", "v3", credentials=credentials)

    try:
        file_metadata = {
            "name": file_name,
            "parents": [folder_id]
        }

        media = MediaFileUpload(file_path, mimetype=mime_type, resumable=False)

        uploaded_file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id",
            supportsAllDrives=True  # 🔑 Important for Shared Drive uploads
        ).execute()

        return uploaded_file.get("id")

    except Exception as e:
        raise RuntimeError(f"Drive upload failed: {e}")


def sanitize_text(text):
    return (
        text.replace("–", "-")
            .replace("—", "-")
            .replace("’", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("…", "...")
            .replace("•", "-")
    )


# llm = ChatOpenAI(
#     model='gpt-3.5-turbo',
#     temperature=0,
#     openai_api_key=st.secrets["openai_api_key"]
# )

from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.tools import Tool
from langchain.agents import initialize_agent, load_tools
from langchain_google_genai import ChatGoogleGenerativeAI
import json

# Set the model name for our LLMs
GEMINI_MODEL = "gemini-1.5-flash"
# GEMINI_MODEL = "models/gemini-pro"
# Store the API key in a variable.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# ✅ Initialize the model
llm = ChatGoogleGenerativeAI(google_api_key=GEMINI_API_KEY, model=GEMINI_MODEL, temperature=0.3)


########################################################
# from langchain_community.chat_models import ChatOllama
# from langchain.schema import HumanMessage

# llm = ChatOllama(model="llama3")

# result = llm.invoke([HumanMessage(content=culture_prompt)])
#############################################################





st.set_page_config(page_title="CALIBER Leadership Inventory©", layout="centered")
st.title("CALIBER Leadership Inventory©")

st.markdown(
    "<div style='text-align: center; font-size: 0.8em; color: gray; margin-top: -1rem;'>"
    "© 2025 M.A. Lakhani. All rights reserved."
    "</div>",
    unsafe_allow_html=True
)


# Define the 25 items with associated leadership and Hofstede dimensions
item_metadata = [
    (1, 'Reinforcement', 'Uncertainty Avoidance', "I ensure my team clearly understands what success looks like and how it's measured."),
    (2, 'Vision', 'Uncertainty Avoidance', "I see ambiguity as an opportunity to explore new ways forward."),
    (3, 'Communication', 'Individualism', "I check regularly for alignment by encouraging open dialogue and honest dissent."),
    (4, 'Empowerment', 'Power Distance', "I trust my team to make decisions without needing my constant approval."),
    (5, 'Stewardship', 'Long-Term Orientation', "I steward resources with long-term sustainability in mind, not just short-term efficiency."),
    (6, 'Confidence', 'Masculinity', "I project calm and confidence even when situations are volatile."),
    (7, 'Creativity', 'Individualism', "I seek out unconventional ideas actively, even if they challenge the status quo."),
    (8, 'Authenticity', 'Masculinity', "I lead by sharing both my strengths and my uncertainties transparently."),
    (9, 'Competence', 'Long-Term Orientation', "I invest in building my expertise and technical fluency continuously."),
    (10, 'Culture', 'Individualism', "I cultivate a team culture where diverse perspectives are invited and heard."),
    (11, 'Empowerment', 'Power Distance', "I believe hierarchy should be earned through trust and competence, not position."),
    (12, 'Vision', 'Uncertainty Avoidance', "I articulate a compelling vision that guides decisions even in ambiguity."),
    (13, 'Reinforcement', 'Masculinity', "I reward consistent follow-through on agreed-upon goals."),
    (14, 'Creativity', 'Uncertainty Avoidance', "I encourage my team to challenge traditional methods when appropriate."),
    (15, 'Communication', 'Power Distance', "I communicate differently based on individual needs and cultural norms."),
    (16, 'Stewardship', 'Individualism', "I mentor others with the intention of creating leaders, not followers."),
    (17, 'Confidence', 'Masculinity', "I remain calm and decisive during crises, even without full clarity."),
    (18, 'Authenticity', 'Individualism', "I speak up when my values are at odds with organizational directives."),
    (19, 'Competence', 'Masculinity', "I provide frequent, constructive feedback to develop team competence."),
    (20, 'Culture', 'Long-Term Orientation', "I adapt team rituals and symbols to strengthen our unique culture."),
    (21, 'Vision', 'Long-Term Orientation', "I make decisions that balance tradition with transformative change."),
    (22, 'Reinforcement', 'Power Distance', "I ensure recognition is distributed fairly and not reserved for top performers only."),
    (23, 'Culture', 'Individualism', "I challenge everyone to remain open to culturally different ideas."),
    (24, 'Confidence', 'Power Distance', "I am comfortable leading when formal authority is unclear or shared."),
    (25, 'Authenticity', 'Long-Term Orientation', "I reflect on past actions to inform future strategies and leadership growth.")
]

# Load national culture scores from CSV
culture_df = pd.read_csv("culture_scores.csv")

# Standardize column names
culture_df.rename(columns={
    'UAI': 'Uncertainty Avoidance',
    'IDV': 'Individualism',
    'PDI': 'Power Distance',
    'MAS': 'Masculinity'
}, inplace=True)

# # Replaced national_profiles = {

#     'USA': {'Power Distance': 40, 'Individualism': 91, 'Masculinity': 62, 'Uncertainty Avoidance': 46, 'Long-Term Orientation': 26},
#     'Japan': {'Power Distance': 54, 'Individualism': 46, 'Masculinity': 95, 'Uncertainty Avoidance': 92, 'Long-Term Orientation': 88},
#     'Sweden': {'Power Distance': 31, 'Individualism': 71, 'Masculinity': 5, 'Uncertainty Avoidance': 29, 'Long-Term Orientation': 53},
#     'Germany': {'Power Distance': 35, 'Individualism': 67, 'Masculinity': 66, 'Uncertainty Avoidance': 65, 'Long-Term Orientation': 83},
#     'India': {'Power Distance': 77, 'Individualism': 48, 'Masculinity': 56, 'Uncertainty Avoidance': 40, 'Long-Term Orientation': 51}
# }

# Initialize session state
if 'responses' not in st.session_state:
    st.session_state.responses = [None] * len(item_metadata)
if 'page' not in st.session_state:
    st.session_state.page = 0

max_page = 5

# Page content
if st.session_state.page == 0:
    st.subheader("Participant Information")
    st.session_state.name = st.text_input("Your Name").strip().title()
    st.session_state.email = st.text_input("Your Email Address").strip()  # ✅ ADD THIS LINE
    st.session_state.industry = st.text_input("Industry in which you work").strip().title()
    st.session_state.job_function = st.text_input("Your job function").strip().title()
    st.session_state.country_work = st.text_input("Country where you currently work").strip().title()
    st.session_state.birth_country = st.text_input("Country where you were born").strip().title()
    # st.text_input("Your Name", key="name")
    # st.text_input("Your Email Address", key="email")
    # st.text_input("Industry in which you work", key="industry")
    # st.text_input("Your job function", key="job_function")
    # st.text_input("Country where you currently work", key="country_work")
    # st.text_input("Country where you were born", key="birth_country")
    # st.write("Work country selected 1:", country_work)
    # st.write("Birth country selected 1:", birth_country)

    st.session_state.survey_for = st.radio("Who are you taking this survey for:", ["Myself", "Someone Else"])
    if st.session_state.survey_for == "Someone Else":
        st.session_state.subject_name = st.text_input("Name of the person you are evaluating").strip().title()
        relation = st.selectbox("Your relationship to that person:", ["The person is my Manager", "The person is my Direct Report", "The person is my Peer", "Other"])
        if relation == "Other":
            st.session_state.relationship = st.text_input("Please describe your relationship")
        else:
            st.session_state.relationship = relation


    # known_countries = culture_df['Country'].str.lower().tolist()
    # if st.session_state.country_work.lower() not in known_countries:
    #     # st.warning("⚠️ Country of Work not found in our database. Results will use closest match available.")
    #     st.session_state.invalid_country_work = True
    # else:
    #     st.session_state.invalid_country_work = False

    # if st.session_state.birth_country.lower() not in known_countries:
    #     # st.warning("⚠️ Country of Birth not found in our database. Results will use closest match available.")
    #     st.session_state.invalid_birth_country = True
    # else:
    #     st.session_state.invalid_birth_country = False

# Added to avert a .csv capture issue
    country_work = st.session_state.get("Country where you currently work", "").strip().title()


    occupation_text = st.session_state.get("job_function", "").lower()
    st.session_state.is_retired = any(kw in occupation_text for kw in ["retired", "not working", "unemployed", "none"])

else:
    st.subheader("Rate each statement from 1 (Strongly Disagree) to 5 (Strongly Agree):")
    st.markdown(f"### Page {st.session_state.page} of {max_page}")
    start_idx = (st.session_state.page - 1) * 5
    for i in range(start_idx, start_idx + 5):
        item = item_metadata[i]
        statement = item[3]
        if st.session_state.survey_for == "Someone Else":
            # Simple pronoun replacement
            statement = statement.replace("I am ", "This individual is ")
            statement = statement.replace("I have ", "This individual has ")
            statement = statement.replace("I was ", "This individual was ")
            statement = statement.replace("I will ", "This individual will ")
            statement = statement.replace("I ", "This individual ").replace(" my ", " his/her ")
            if statement.startswith("I"):
                statement = "This individual" + statement[1:]
                # Correct verb conjugation for "This individual" if statement begins with it
            replacements = {
                "ensure": "ensures",
                "steward": "stewards",
                "seek": "seeks",
                "see": "sees",
                "check": "checks",
                "trust": "trusts",
                "steward": "stewards",
                "project": "projects",
                "lead": "leads",
                "invest": "invests",
                "cultivate": "cultivates",
                "believe": "believes",
                "articulate": "articulates",
                "reward": "rewards",
                "encourage": "encourages",
                "communicate": "communicates",
                "mentor": "mentors",
                "remain": "remains",
                "speak": "speaks",
                "provide": "provides",
                "adapt": "adapts",
                "make": "makes",
                "challenge": "challenges",
                "reflect": "reflects",
                "am comfortable leading": "is comfortable leading"
            }
            for k, v in replacements.items():
                prefix = f"This individual {k}"
                if statement.lower().startswith(prefix.lower()):
                    words = statement.split()
                    if len(words) > 2:
                        words[2] = v  # replace the verb directly
                        statement = ' '.join(words)
                    break
                    
            # if k in statement:
            #     statement = statement.replace(k, v, 1)
            # break
        else:
            # For "Myself", use original statement with no change
            statement = item[3]

        st.session_state.responses[i] = st.slider(f"{item[0]}. {statement}", 1, 5, 3, key=f"q{i}")
        # st.session_state.responses[i] = st.slider(f"{item[0]}. {statement}", 1, 5, 3, key=f"q{i}")

country_work = st.session_state.get("country_work")
country_birth = st.session_state.get("country_birth")
# st.write("Work country selected 2:", country_work)
# st.write("Birth country selected 2:", birth_country)

# # Always store both, regardless of match
# demographics["Country of Work"] = country_work
# demographics["Country of Birth"] = country_birth

# Navigation buttons (bottom only, no form requirement)
col1, col2 = st.columns([1, 1])
with col1:
    if st.session_state.page > 0:
        if st.button("⬅️ Previous Page"):
            st.session_state.page -= 1
            st.rerun()
with col2:
    if st.session_state.page < max_page:
        if st.button("Next Page ➡️"):
            st.session_state.page += 1
            st.rerun()

from collections import defaultdict

dimension_items = defaultdict(list)
for idx, item in enumerate(item_metadata):
    dimension = item[1]  # Leadership_Dimension
    dimension_items[dimension].append(idx)

# Submit logic
if st.session_state.page == max_page:
    st.subheader("Submit Survey")
    st.write("Please review your answers. When you're ready, click below to submit and download your results.")

    if st.button("Submit Survey"):
        with st.spinner("Creating your personalized leadership report..."):
            df = pd.DataFrame({
                "Question Number": [item[0] for item in item_metadata],
                "Leadership Dimension": [item[1] for item in item_metadata],
                "Hofstede Dimension": [item[2] for item in item_metadata],
                "Statement": [item[3] for item in item_metadata],
                "Response": st.session_state.responses
            })

            from gspread_dataframe import get_as_dataframe, set_with_dataframe

            import gspread
            from google.oauth2 import service_account
            import streamlit as st
            from gspread_dataframe import get_as_dataframe

            # Authenticate with service account
            credentials = service_account.Credentials.from_service_account_info(
                st.secrets["gdrive"],  # assuming you stored credentials in Streamlit secrets
                scopes=["https://www.googleapis.com/auth/spreadsheets"]
            )

            gc = gspread.authorize(credentials)

            # Open the spreadsheet and worksheet
            spreadsheet = gc.open("CALIBER_Survey_Responses")  # Replace with your actual sheet name
            sheet = spreadsheet.worksheet("Sheet1")  # Replace with your actual tab name

            # Get existing data
            existing_df = get_as_dataframe(sheet)

            # Combine
            combined_df = pd.concat([existing_df, df], ignore_index=True)

            # Write back full sheet
            set_with_dataframe(sheet, combined_df)


            # Manual Hofstede dimension calculations (matching Excel logic)
            dimension_custom_scores = {
                'High Uncertainty Avoidance': st.session_state.responses[0] + (5 - st.session_state.responses[1]) + (5 - st.session_state.responses[12]) + (5 - st.session_state.responses[13]),
                'High Individualism': st.session_state.responses[2] + st.session_state.responses[6] + st.session_state.responses[9] + st.session_state.responses[15] + st.session_state.responses[17] + st.session_state.responses[22],
                'High Power Distance': st.session_state.responses[3] + st.session_state.responses[10] + st.session_state.responses[14] + st.session_state.responses[23],
                'Long-Term Orientation': st.session_state.responses[4] + st.session_state.responses[8] + st.session_state.responses[19] + st.session_state.responses[20] + st.session_state.responses[24],
                'High Masculinity': st.session_state.responses[5] + st.session_state.responses[7] + st.session_state.responses[16] + st.session_state.responses[18],
                'High Uncertainty Avoidance PCT': (st.session_state.responses[0] + (5 - st.session_state.responses[1]) + (5 - st.session_state.responses[12]) + (5 - st.session_state.responses[13])-4)/16,
                'High Individualism PCT': (st.session_state.responses[2] + st.session_state.responses[6] + st.session_state.responses[9] + st.session_state.responses[15] + st.session_state.responses[17] + st.session_state.responses[22]-6)/24,
                'High Power Distance PCT': (st.session_state.responses[3] + st.session_state.responses[10] + st.session_state.responses[14] + st.session_state.responses[23]-4)/16,
                'Long-Term Orientation PCT': (st.session_state.responses[4] + st.session_state.responses[8] + st.session_state.responses[19] + st.session_state.responses[20] + st.session_state.responses[24]-5)/20,
                'High Masculinity PCT': (st.session_state.responses[5] + st.session_state.responses[7] + st.session_state.responses[16] + st.session_state.responses[18]-4)/16,
                'Reinforcement': st.session_state.responses[0]+st.session_state.responses[12]+st.session_state.responses[21],
                'Vision': st.session_state.responses[1]+st.session_state.responses[11]+st.session_state.responses[20],
                'Communication':st.session_state.responses[2]+st.session_state.responses[14],
                'Authenticity':st.session_state.responses[7]+st.session_state.responses[17]+st.session_state.responses[24],
                'Competence':st.session_state.responses[8]+st.session_state.responses[18],
                'Confidence':st.session_state.responses[5]+st.session_state.responses[16]+st.session_state.responses[23],
                'Creativity':st.session_state.responses[6]+st.session_state.responses[13],
                'Culture':st.session_state.responses[9]+st.session_state.responses[19]+st.session_state.responses[22],
                'Empowerment':st.session_state.responses[3]+st.session_state.responses[10],
                'Stewardship':st.session_state.responses[4]+st.session_state.responses[15],
                'Reinforcement PCT': (st.session_state.responses[0]+st.session_state.responses[12]+st.session_state.responses[21]-3)/(3*4),
                'Vision PCT': (st.session_state.responses[1]+st.session_state.responses[11]+st.session_state.responses[20]-3)/(3*4),
                'Communication PCT':(st.session_state.responses[2]+st.session_state.responses[14]-2)/(2*4),
                'Authenticity PCT':(st.session_state.responses[7]+st.session_state.responses[17]+st.session_state.responses[24]-3)/(3*4),
                'Competence PCT':(st.session_state.responses[8]+st.session_state.responses[18]-2)/(2*4),
                'Confidence PCT':(st.session_state.responses[5]+st.session_state.responses[16]+st.session_state.responses[23]-3)/(3*4),
                'Creativity PCT':(st.session_state.responses[6]+st.session_state.responses[13]-2)/(2*4),
                'Culture PCT':(st.session_state.responses[9]+st.session_state.responses[19]+st.session_state.responses[22]-3)/(3*4),
                'Empowerment PCT':(st.session_state.responses[3]+st.session_state.responses[10]-2)/(2*4),
                'Stewardship PCT':(st.session_state.responses[4]+st.session_state.responses[15]-2)/(2*4)
            }

            
            # Extract user Hofstede cultural profile
            user_profile = {
                'Uncertainty Avoidance': dimension_custom_scores['High Uncertainty Avoidance PCT'] * 100,
                'Individualism': dimension_custom_scores['High Individualism PCT'] * 100,
                'Power Distance': dimension_custom_scores['High Power Distance PCT'] * 100,
                'Masculinity': dimension_custom_scores['High Masculinity PCT'] * 100
            }

            from scipy.spatial.distance import euclidean

            # Compute Euclidean distance from each country in the dataset
            def compute_distance(row):
                return euclidean([
                    row['Uncertainty Avoidance'],
                    row['Individualism'],
                    row['Power Distance'],
                    row['Masculinity']
                ], list(user_profile.values()))

            culture_df['Distance'] = culture_df.apply(compute_distance, axis=1)
            closest_cultures = culture_df.nsmallest(5, 'Distance')['Country'].tolist()


            leadership_custom_scores = {
                            'Innovation PCT': (dimension_custom_scores['Communication PCT'] + dimension_custom_scores['Vision PCT'] + dimension_custom_scores['Authenticity PCT'] + dimension_custom_scores['Empowerment PCT'] + dimension_custom_scores['Creativity PCT'])/5,
                            'Operations PCT': (dimension_custom_scores['Stewardship PCT'] + dimension_custom_scores['Competence PCT'] + dimension_custom_scores['Confidence PCT'] + dimension_custom_scores['Reinforcement PCT'] + dimension_custom_scores['Culture PCT'])/5,
                            'Overall Leadership PCT': (dimension_custom_scores['Communication PCT'] + dimension_custom_scores['Vision PCT'] + dimension_custom_scores['Authenticity PCT'] + dimension_custom_scores['Empowerment PCT'] + dimension_custom_scores['Creativity PCT'] + dimension_custom_scores['Stewardship PCT'] + dimension_custom_scores['Competence PCT'] + dimension_custom_scores['Confidence PCT'] + dimension_custom_scores['Reinforcement PCT'] + dimension_custom_scores['Culture PCT'])/10
                        }


            # Convert to a DataFrame (transposed to get dimensions as rows)
            dimension_df = pd.DataFrame(list(dimension_custom_scores.items()), columns=['Dimension', 'Score'])

            # Convert to a DataFrame (transposed to get dimensions as rows)
            leadership_df = pd.DataFrame(list(leadership_custom_scores.items()), columns=['Dimension', 'Score'])
            
            # Optional: add a blank row for separation
            blank_row = pd.DataFrame([['', '']], columns=['Dimension', 'Score'])

            def get_country(field_name):
                val = st.session_state.get(field_name, "").strip()
                return val if val else "United States"
            
            def get_email(field_name):
                val = st.session_state.get(field_name, "").strip()
                return val if val else "Unknown"

            # Prepare metadata (demographics)
            meta_info = pd.DataFrame({
                'Field': [
                    'Name',
                    'Email',
                    'Job Function',
                    'Industry',
                    'Country of Work',
                    'Country of Birth',
                    'Survey Taken For',
                    'Subject Name',
                    'Relationship'
                ],
                'Value': [
                    st.session_state.get("name", ""),
                    # st.session_state.get("email", ""),
                    get_email("email"),
                    st.session_state.get("job_function", ""),
                    st.session_state.get("industry", ""),
                    # st.session_state.get("country_work", ""),
                    # st.session_state.get("birth_country", ""),
                    get_country("country_work"),
                    get_country("birth_country"),
                    st.session_state.get("survey_for", ""),
                    st.session_state.get("subject_name", "") if st.session_state.get("survey_for") == "Someone Else" else "",
                    st.session_state.get("relationship", "") if st.session_state.get("survey_for") == "Someone Else" else ""
                ]
            })

            # Optional spacing row
            blank_row = pd.DataFrame([['', '']], columns=['Field', 'Value'])

            # Combine metadata + survey results
            meta_and_scores = pd.concat([meta_info, blank_row], ignore_index=True)

            # Combine original df with new section
            df_combined = pd.concat([df, blank_row, dimension_df, blank_row, leadership_df, blank_row, meta_and_scores], ignore_index=True)

            # st.write("Work country selected 3:", country_work)
            # st.write("Birth country selected 3:", birth_country)

    
            # Clean name for filename
            # Collect contextual inputs
            participant_role = st.session_state.get("job_function", "a professional")
            participant_industry = st.session_state.get("industry", "their industry")
            # country_work = st.session_state.get("country_work", "their country of work")
            # birth_country = st.session_state.get("birth_country", "their country of origin")
            country_work = get_country("country_work"),
            birth_country = get_country("birth_country"),
            email = get_email("email")

            participant_name = st.session_state.get("name", "anonymous")
            clean_name = re.sub(r'\W+', '_', participant_name.strip())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"caliber_survey_{clean_name}_{timestamp}.csv"

            df_combined.to_csv(filename, index=False)


            # If survey was for someone else, skip the rest
            if st.session_state.get("survey_for") == "Someone Else":
                csv_drive_id = upload_to_drive(filename, filename, "text/csv", folder_id)
                st.success("✅ Thank you for providing your assessment. The results have been saved.")
                st.markdown("You may now close this window or return to the home page.")
                st.stop()  # Stop further execution (no report generation)


            import streamlit as st
            from PIL import Image

            # Determine leadership level from score
            score_pct = leadership_custom_scores['Overall Leadership PCT'] * 100
            if score_pct <= 33.33:
                level = "Aspiring Leader"
            elif score_pct <= 66.66:
                level = "Developing Leader"
            else:
                level = "Performing Leader"


            
    
            import matplotlib.pyplot as plt
            import seaborn as sns
            import os

            # === Generate and Save Bar Chart ===
            # === Create and Save Hofstede Chart First ===
            hofstede_keys = [
                "High Uncertainty Avoidance PCT",
                "High Individualism PCT",
                "High Power Distance PCT",
                "Long-Term Orientation PCT",
                "High Masculinity PCT"
            ]

            hofstede_scores = [dimension_custom_scores[k] * 100 for k in hofstede_keys]
            hofstede_labels = [
                "Uncertainty Avoidance",
                "Individualism",
                "Power Distance",
                "Long-Term Orientation",
                "Masculinity"
            ]

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=hofstede_scores, y=hofstede_labels, palette="Blues_d", ax=ax)
            ax.set_xlim(0, 100)
            ax.set_title("Cultural Dimensions Profile (Hofstede)")
            ax.set_xlabel("Score")
            ax.set_ylabel("")
            sns.despine()

            hofstede_path = f"hofstede_chart_{clean_name}_{timestamp}.png"
            fig.tight_layout()
            fig.savefig(hofstede_path, dpi=150)
            plt.close(fig)

            import matplotlib.pyplot as plt
            import seaborn as sns

            # Bar chart for leadership dimensions (Innovation vs Operations)
            dimensions = [
                "Communication PCT", "Vision PCT", "Authenticity PCT", "Empowerment PCT", "Creativity PCT",
                "Stewardship PCT", "Competence PCT", "Confidence PCT", "Reinforcement PCT", "Culture PCT"
            ]

            scores = [
                dimension_custom_scores[dim] * 100 for dim in dimensions
            ]

            labels = [
                "Communication", "Vision", "Authenticity", "Empowerment", "Creativity",
                "Stewardship", "Competence", "Confidence", "Reinforcement", "Culture"
            ]

            category = ["Innovation"] * 5 + ["Operations"] * 5
            palette = sns.color_palette("Set2", 2)

            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x=scores, y=labels, hue=category, dodge=False, palette=palette, ax=ax)
            ax.set_title("Leadership Dimension Scores")
            ax.set_xlim(0, 100)
            ax.set_xlabel("Score")
            ax.set_ylabel("")
            sns.despine()

            bar_chart_path = f"leadership_dimensions_{clean_name}_{timestamp}.png"
            fig.tight_layout()
            fig.savefig(bar_chart_path, dpi=150)
            plt.close(fig)

            def generate_overall_leadership_plot(score_pct, save_path):
                fig, ax = plt.subplots(figsize=(10, 2))
                ax.axhspan(0, 1, xmin=0.0, xmax=0.3333, facecolor='#ff9999', alpha=0.5)
                ax.axhspan(0, 1, xmin=0.3333, xmax=0.6666, facecolor='#ffe066', alpha=0.5)
                ax.axhspan(0, 1, xmin=0.6666, xmax=1.0, facecolor='#99ff99', alpha=0.5)

                ax.axvline(score_pct, color='black', linewidth=3)
                ax.text(10, 0.8, 'Aspiring Leader', fontsize=10, color='black')
                ax.text(40, 0.8, 'Developing Leader', fontsize=10, color='black')
                ax.text(75, 0.8, 'Performing Leader', fontsize=10, color='black')

                ax.set_title('Overall Leadership Score', fontsize=14, weight='bold')
                ax.set_xlim(0, 100)
                ax.set_yticks([])
                ax.set_xlabel('Score')
                sns.despine(left=True, bottom=True)
                plt.tight_layout()
                fig.savefig(save_path, dpi=150)
                plt.close(fig)

            score = leadership_custom_scores['Overall Leadership PCT'] * 100
            sumplot_path = f"leadership_score_{clean_name}_{timestamp}.png"
            generate_overall_leadership_plot(score, sumplot_path)
            
            from datetime import datetime

            report_date = datetime.now().strftime("%B %d, %Y")
            pdf_filename = f"leadership_summary_{clean_name}_{timestamp}.pdf"


            # Define the interpretation task
            # summary_description = ("""
            # Please note: This is the first page of the CALIBER Leadership Inventory report. In addition to this expert analysis, the full report includes detailed scores, a national culture profile, and specific actions and development recommendations. Encourage the participant to carefully review the complete document.

            # """ + 
            #     f"Write a 1-page report for {participant_name} who works in {participant_industry} as {participant_role}. "
            #     f"They scored {score_pct:.1f}/100 on the CALIBER Leadership Inventory. "
            #     f"Label their leadership category as '{level}'. Reflect on the implications of this level of leadership capability "
            #     f"on team performance and organizational culture within the context of {participant_industry}. Use positive, constructive tone. "
            #     f"Also take into account that the participant currently works in {country_work} but was born in {birth_country}. "
            #     f"Comment on how cultural dimensions might influence their leadership style and how cultural awareness can enhance their effectiveness. "
            #     "Explain why leadership development is vital in their role and industry, and include a motivational call to action for growth." + f" Their leadership practices culturally align best with: {', '.join(closest_cultures)}."
            # )

#             summary_description = f"""
# You are generating a personalized executive summary for a professional leadership report. DO NOT include fictional elements like the current date or salutations. DO NOT include the participant's name in a heading. 

# Write a clear, well-organized executive summary (maximum 400 words) based on the following inputs:

# - Participant Name: {participant_name}
# - Industry: {participant_industry}
# - Job Function: {participant_role}
# - Leadership Score: {score_pct:.1f}/100
# - Leadership Category: {level}
# - Country of Work: {country_work}
# - Country of Birth: {birth_country}


# Instructions:
# 1. Begin with a professional overview of the participant’s leadership score and category.  Use a personal tone.
# 2. Reflect on what this level of capability means for their team and organizational culture in the context of their industry.
# 3. Discuss how their cultural background (birth country vs work country) might influence their leadership style, based on Hofstede’s dimensions.
# 4. Avoid headers like “Call to Action” or fake dates. Use a constructive, motivational, personal tone in paragraph format.
# 5. Conclude with an uplifting statement encouraging leadership development.

# Keep the entire response concise, insightful, and under 400 words.
# """
#             summary_description = f"""
# You are generating an executive summary for a professional leadership report. Avoid fake elements like the current date or salutations. DO NOT include a header with the participant’s name or title. Instead, refer to the participant by name naturally within the body of the summary.  Speak to the participant directly.

# Write a clear, well-organized executive summary (under 500 words) based on the following:

# - Participant Name: {participant_name}
# - Industry: {participant_industry}
# - Job Function: {participant_role}
# - Leadership Score: {score_pct:.1f}/100
# - Leadership Category: {level}
# - Country of Work: {country_work}
# - Country of Birth: {birth_country}

# Instructions:
# 1. Begin by introducing {participant_name} and referencing their leadership score and category.
# 2. Reflect on what this level of leadership means for their team and organizational culture in the context of their industry {participant_industry} and role {participant_role}.
# 3. Include some current challenges for those playing the role of {participant_role} in this industry {participant_industry}.
# 4. Discuss how being born in {birth_country} and working in {country_work} may influence {participant_name}'s leadership style, referencing Hofstede’s cultural dimensions.
# 5. Use a constructive, professional tone in paragraph form.
# 6. End with a motivational call to action encouraging leadership growth.

# Avoid formal section titles like “Call to Action.” Write in paragraph format and keep it personal, supportive, and inspiring.  Fit within one page.
# """

            summary_description = f"""
You are generating the **first-page executive summary** of a professional leadership development report for {participant_name}. The summary should feel insightful, credible, and encouraging — written in a voice that balances warmth with authority. Avoid artificial elements like the current date, salutations, or headings such as “Executive Summary.” Do not insert boilerplate content.

### Participant Context:
- Name: {participant_name}
- Industry: {participant_industry}
- Job Function: {participant_role}
- Leadership Score: {score_pct:.1f}/100
- Leadership Category: {level}
- Country of Work: {country_work}
- Country of Birth: {birth_country}

### Writing Instructions:
- Speak directly to {participant_name}, using second person ("you") throughout.
- Open with a strong lead that introduces their leadership score and what it reflects about their style and potential.
- Reflect on what this level of leadership means for team culture and strategic influence in the context of their specific industry ({participant_industry}) and role ({participant_role}).
- Identify 1–2 challenges commonly faced by leaders in {participant_industry}, particularly in the role of {participant_role}, and tie these to {participant_name}'s leadership development opportunity.
- Weave in cultural insight by interpreting the influence of being born in {birth_country} and now working in {country_work}, referencing Hofstede’s cultural dimensions without naming the model explicitly (e.g., refer to tendencies around hierarchy, individualism, or long-term thinking).
- End with an empowering and specific call to action that motivates {participant_name} to build upon their strengths and navigate cross-cultural dynamics effectively.

### Tone and Constraints:
- Frame insights in the second person (e.g., “You may find…” or “You might experience…”) and avoid speaking **as** the participant. Instead, write from an expert coaching perspective, offering guidance with warmth, clarity, and respect.
- Keep the tone reflective, personalized, constructive, and grounded in insight — not hype.
- Avoid clichés, filler phrases, or superficial praise.
- The content should read as if written by a seasoned executive coach or leadership strategist.
- Total length: one well-structured page (under 500 words), in paragraph form with no titles or bullet points.
"""          
            
            
            
#             summary_description = f"""
# You are generating the **first-page executive summary** of a professional leadership development report for {participant_name}. The summary should feel insightful, credible, and encouraging — written in a voice that balances warmth with authority. Avoid artificial elements like the current date, salutations, or headings such as “Executive Summary.” Do not insert boilerplate content.

# ### Participant Context:
# - Name: {participant_name}
# - Industry: {participant_industry}
# - Job Function: {participant_role}
# - Leadership Score: {score_pct:.1f}/100
# - Leadership Category: {level}
# - Country of Work: {country_work}
# - Country of Birth: {birth_country}

# ### Writing Instructions:
# - Speak directly to {participant_name}, using second person ("you") throughout.
# - Open with a strong lead that introduces their leadership score and what it reflects about their style and potential.
# - Reflect on what this level of leadership means for team culture and strategic influence in the context of their specific industry ({participant_industry}) and role ({participant_role}).
# - Identify 1–2 challenges commonly faced by leaders in {participant_industry}, particularly in the role of {participant_role}, and tie these to {participant_name}'s leadership development opportunity.
# - Weave in cultural insight by interpreting the influence of being born in {birth_country} and now working in {country_work}, referencing Hofstede’s cultural dimensions without naming the model explicitly (e.g., refer to tendencies around hierarchy, individualism, or long-term thinking).
# - End with an empowering and specific call to action that motivates {participant_name} to build upon their strengths and navigate cross-cultural dynamics effectively.

# ### Tone and Constraints:
# - Frame insights in the second person (e.g., “You may find…” or “You might experience…”) and avoid speaking **as** the participant. Instead, write from an expert coaching perspective, offering guidance with warmth, clarity, and respect.
# - Keep the tone reflective, personalized, constructive, and grounded in insight — not hype.
# - Avoid clichés, filler phrases, or superficial praise.
# - The content should read as if written by a seasoned executive coach or leadership strategist.
# - Total length: one well-structured page (under 500 words), in paragraph form with no titles or bullet points.
# """



            # Run the crew

            
            # summary_prompt = f"""
            # Write a 1-page report for {participant_name} who works in {participant_industry} as {participant_role}.
            # They scored {score_pct:.1f}/100 on the CALIBER Leadership Inventory.
            # Label their leadership category as '{level}'.
            # Reflect on implications for team performance and culture within the context of {participant_industry}.
            # Include how being born in {birth_country} and working in {country_work} affects leadership style.
            # Align analysis with Hofstede cultural dimensions: {', '.join(closest_cultures)}.
            # Use the official CALIBER tone: positive, structured, and actionable.
            # """
            # result = llm.predict(summary_prompt)
            result = llm.predict(summary_description)
            # result = llm.invoke([HumanMessage(content=summary_description)])



            # Compose interpretation task for dimensions

            
            # page2_prompt = f"""
            # Write a summary interpreting the leadership scores in 10 dimensions.
            # Separate discussion into Innovation (Communication, Vision, Authenticity, Empowerment, Creativity) and Operations (Stewardship, Competence, Confidence, Reinforcement, Culture).
            # Explain the significance of each score, leadership potential, and team/organizational impact.
            # Use CALIBER tone, structure, and style.
            # """
#             page2_prompt = f"""
# You are writing a personalized interpretation of the participant’s leadership profile based on their scores across 10 dimensions.

# Organize your response in two cohesive sections:
# - **Innovation**: Communication, Vision, Authenticity, Empowerment, Creativity
# - **Operations**: Stewardship, Competence, Confidence, Reinforcement, Culture

# For each cluster:
# 1. Interpret the participant’s scores, highlighting both strengths and opportunities.
# 2. Discuss how these traits may manifest in their leadership behavior, influence team dynamics, and shape organizational outcomes.
# 3. Use a warm, insightful, and empowering tone aligned with CALIBER's coaching style.

# Avoid listing scores mechanically. Instead, weave them naturally into a narrative that reflects the participant’s potential and leadership journey.
# """

            # page2_prompt = f"""
            # {participant_name}, let's take a closer look at your leadership profile. Reflect on your strengths and growth opportunities across two core areas: Innovation and Operations. Speak to the participant directly.

            # - **Innovation** covers Communication, Vision, Authenticity, Empowerment, and Creativity.
            # - **Operations** includes Stewardship, Competence, Confidence, Reinforcement, and Culture.

            # Write two clear paragraphs — one for Innovation and one for Operations — highlighting standout scores and opportunities. Offer concrete suggestions for improvement and tie each insight to potential team or organizational impact. Use a confident, encouraging tone and address {participant_name} directly.  Fit within one page.
            # """
            page2_prompt = f"""
            {participant_name}, let’s explore how your leadership profile expresses itself across two critical dimensions: **Innovation** and **Operations** — both of which are essential to effective leadership in your role as a {participant_role} within the {participant_industry} industry.

            ### Innovation:
            This domain includes Communication, Vision, Authenticity, Empowerment, and Creativity. In the context of your industry, where agility and forward-thinking often differentiate impactful leaders, highlight your highest strengths and any dimensions where further development could enhance innovation. Offer examples or strategies relevant to the {participant_industry} space that could help {participant_name} push boundaries, inspire teams, or drive transformational thinking.

            ### Operations:
            This includes Stewardship, Competence, Confidence, Reinforcement, and Culture. Address how {participant_name}'s current operational strengths align with the expectations and pressures of a {participant_role} in the {participant_industry} sector. Offer constructive suggestions to enhance consistency, clarity, and executional excellence — all framed in a way that connects individual growth with broader team or organizational outcomes.

            Use a coaching tone that’s direct yet supportive. Speak to {participant_name} as a capable and evolving leader. Keep the writing clear and concise — structured as **two focused paragraphs** (one for Innovation, one for Operations), with practical insights and a sense of positive momentum. Fit all content on a single page, avoiding titles or bullet points.

            ### Tone and Constraints:
- Frame insights in the second person (e.g., “You may find…” or “You might experience…”) and avoid speaking **as** the participant. Instead, write from an expert coaching perspective, offering guidance with warmth, clarity, and respect.
- Keep the tone reflective, personalized, constructive, and grounded in insight — not hype.
- Avoid clichés, filler phrases, or superficial praise.
- The content should read as if written by a seasoned executive coach or leadership strategist.
- Total length: one well-structured page (under 500 words), in paragraph form with no titles or bullet points.
            """
            
            
#             page2_prompt = f"""
#             {participant_name}, let’s explore how your leadership profile expresses itself across two critical dimensions: **Innovation** and **Operations** — both of which are essential to effective leadership in your role as a {participant_role} within the {participant_industry} industry.

#             ### Innovation:
#             This domain includes Communication, Vision, Authenticity, Empowerment, and Creativity. In the context of your industry, where agility and forward-thinking often differentiate impactful leaders, highlight your highest strengths and any dimensions where further development could enhance innovation. Offer examples or strategies relevant to the {participant_industry} space that could help {participant_name} push boundaries, inspire teams, or drive transformational thinking.

#             ### Operations:
#             This includes Stewardship, Competence, Confidence, Reinforcement, and Culture. Address how {participant_name}'s current operational strengths align with the expectations and pressures of a {participant_role} in the {participant_industry} sector. Offer constructive suggestions to enhance consistency, clarity, and executional excellence — all framed in a way that connects individual growth with broader team or organizational outcomes.

#             Use a coaching tone that’s direct yet supportive. Speak to {participant_name} as a capable and evolving leader. Keep the writing clear and concise — structured as **two focused paragraphs** (one for Innovation, one for Operations), with practical insights and a sense of positive momentum. Fit all content on a single page, avoiding titles or bullet points.

#             ### Tone and Constraints:
# - Frame insights in the second person (e.g., “You may find…” or “You might experience…”) and avoid speaking **as** the participant. Instead, write from an expert coaching perspective, offering guidance with warmth, clarity, and respect.
# - Keep the tone reflective, personalized, constructive, and grounded in insight — not hype.
# - Avoid clichés, filler phrases, or superficial praise.
# - The content should read as if written by a seasoned executive coach or leadership strategist.
# - Total length: one well-structured page (under 500 words), in paragraph form with no titles or bullet points.
#             """



            page2_result = llm.predict(page2_prompt)
            # page2_result = llm.invoke([HumanMessage(content=page2_prompt)])

            # pdf.chapter_title("Interpretation of Innovation & Operations Dimensions")
            # pdf.chapter_body(sanitize_text(page2_result))
            # # pdf.add_image(bar_chart_path, "Leadership Dimension Breakdown")
            # # Page 3 – National Culture Analysis
            # pdf.add_page()
            # pdf.chapter_title("Cultural Context and Implications")
            
            # culture_prompt = f"""
            # Provide a concise analysis of how being born in {birth_country} but currently working in {country_work} might shape leadership expectations.
            # Reference Hofstede’s dimensions.
            # Include potential cultural tensions or synergies and leadership guidance.
            # Use the official CALIBER tone and structure.
            # """
#             culture_prompt = f"""
# Write a thoughtful reflection on how being born in {birth_country} and currently working in {country_work} may influence the participant’s leadership style.

# Reference Hofstede’s cultural dimensions to explore how national values—such as attitudes toward hierarchy, uncertainty, or individualism—might shape expectations and behavior in the workplace.

# Highlight potential cultural tensions or synergies the participant may encounter, and offer supportive, actionable guidance for leading effectively across these cultural dynamics.

# Maintain a constructive, growth-focused tone consistent with CALIBER’s personalized coaching approach.
# """

            # culture_prompt = f"""
            #     Explore how being born in {birth_country} and now working in {country_work} might shape {participant_name}’s leadership expectations and behaviors.

            #     - Use Hofstede’s dimensions to interpret cultural contrasts.
            #     - Highlight where {participant_name}'s cultural values may align or conflict with workplace expectations.
            #     - Offer constructive guidance on how {participant_name} can adapt to thrive across cultural settings.

            #     Write in the second person (e.g., “You may find...”) and maintain CALIBER's supportive and thoughtful tone.  Fit within one page.
            #     """
            culture_prompt = f"""
{participant_name}, your leadership journey is shaped not only by your role and industry, but also by your cultural lens. Being born in {birth_country} and now working in {country_work} offers you a unique cross-cultural perspective that influences how you lead, communicate, and respond to organizational norms.

Use this section to thoughtfully explore how cultural influences might shape your expectations and behaviors in the workplace. Consider how values formed in {birth_country} — such as attitudes toward hierarchy, uncertainty, collaboration, or long-term orientation — may align with or differ from the norms in {country_work}. Use Hofstede’s dimensions to guide your interpretation, but keep your insights conversational and accessible.

Offer constructive, culturally aware guidance on how you can navigate these contrasts to become an even more adaptive and effective leader. Use second person language ("You may find...") and speak directly to {participant_name} with CALIBER’s trademark tone: personal, thoughtful, and empowering. Keep the reflection concise and impactful, fitting within a single page and written in paragraph form.

### Tone and Constraints:
- Frame insights in the second person (e.g., “You may find…” or “You might experience…”) and avoid speaking **as** the participant. Instead, write from an expert coaching perspective, offering guidance with warmth, clarity, and respect.
- Keep the tone reflective, personalized, constructive, and grounded in insight — not hype.
- Avoid clichés, filler phrases, or superficial praise.
- The content should read as if written by a seasoned executive coach or leadership strategist.
- Total length: one well-structured page (under 500 words), in paragraph form with no titles or bullet points.
"""            
            
#             culture_prompt = f"""
# {participant_name}, your leadership journey is shaped not only by your role and industry, but also by your cultural lens. Being born in {birth_country} and now working in {country_work} offers you a unique cross-cultural perspective that influences how you lead, communicate, and respond to organizational norms.

# Use this section to thoughtfully explore how cultural influences might shape your expectations and behaviors in the workplace. Consider how values formed in {birth_country} — such as attitudes toward hierarchy, uncertainty, collaboration, or long-term orientation — may align with or differ from the norms in {country_work}. Use Hofstede’s dimensions to guide your interpretation, but keep your insights conversational and accessible.

# Offer constructive, culturally aware guidance on how you can navigate these contrasts to become an even more adaptive and effective leader. Use second person language ("You may find...") and speak directly to {participant_name} with CALIBER’s trademark tone: personal, thoughtful, and empowering. Keep the reflection concise and impactful, fitting within a single page and written in paragraph form.

# ### Tone and Constraints:
# - Frame insights in the second person (e.g., “You may find…” or “You might experience…”) and avoid speaking **as** the participant. Instead, write from an expert coaching perspective, offering guidance with warmth, clarity, and respect.
# - Keep the tone reflective, personalized, constructive, and grounded in insight — not hype.
# - Avoid clichés, filler phrases, or superficial praise.
# - The content should read as if written by a seasoned executive coach or leadership strategist.
# - Total length: one well-structured page (under 500 words), in paragraph form with no titles or bullet points.
# """



            culture_result = llm.predict(culture_prompt)
            # culture_result = llm.invoke([HumanMessage(content=culture_prompt)])

            
            # coach_prompt = f"""
            # Write a structured and accessible development plan for {participant_name}.
            # Suggest 3–5 growth areas across Innovation and Operations dimensions.
            # Provide short rationale for each.
            # Comment on Hofstede cultural scores and alignments: {', '.join(closest_cultures)}.
            # Offer guidance for cross-cultural adaptability and leadership effectiveness.
            # Use CALIBER tone and format.
            # """
#             coach_prompt = f"""
# Develop a personalized leadership growth plan for {participant_name} based on their assessment results.

# Identify 3 to 5 focused areas for development, drawing from both Innovation (e.g., Communication, Vision, Creativity) and Operations (e.g., Stewardship, Competence, Culture) dimensions. For each area, provide a brief rationale that highlights its significance for their role and leadership journey.

# Incorporate insights from Hofstede’s cultural dimensions, especially as they relate to alignment with cultural profiles like {', '.join(closest_cultures)}. Reflect on how these cultural influences might support or challenge the participant's growth.

# Conclude with clear, supportive guidance for cultivating cross-cultural leadership effectiveness. Maintain CALIBER’s tone: personalized, motivational, and forward-looking.
# """

            # coach_prompt = f"""
            # Create a personalized leadership development plan for {participant_name}. Identify 3 to 5 growth areas across Innovation and Operations.  Fit within one page.

            # For each:
            # - Provide a short title (e.g., “Strategic Communication”) and a one-sentence rationale.
            # - Give 1-2 practical steps {participant_name} can take to improve.
            # - Briefly comment on how {', '.join(closest_cultures)} cultural patterns may influence leadership effectiveness.

            # End with an inspiring message encouraging reflection, application, and follow-up. Write as if coaching {participant_name} directly.
            # """
            coach_prompt = f"""
{participant_name}, based on your leadership profile, this page outlines a personalized development plan tailored to your role as a {participant_role} in the {participant_industry} industry. The goal is to help you grow as a leader in ways that align with the unique demands, pace, and culture of your professional environment.

Identify **3 to 5 key growth areas** drawn from your Innovation and Operations scores — areas that matter most for someone in your position. For each:

- Start with a short, descriptive title (e.g., “Strategic Communication”) and a one-sentence rationale that directly links the growth area to your responsibilities as a {participant_role} in {participant_industry}.
- Offer 1–2 practical, real-world actions you can begin implementing — ideally in ways that create value for your team or organization.
- Briefly reflect on how cultural patterns common in {', '.join(closest_cultures)} may impact how this skill is expressed or received, and suggest how you might navigate these dynamics effectively.
Maintain a confident and constructive tone, as if coaching {participant_name} personally. End with a motivating message that encourages commitment to intentional practice and ongoing leadership growth. Keep everything to **one page**, in paragraph form — no headers or bullet points.
### Tone and Constraints:
- Frame insights in the second person (e.g., “You may find…” or “You might experience…”) and avoid speaking **as** the participant. Instead, write from an expert coaching perspective, offering guidance with warmth, clarity, and respect.
- Keep the tone reflective, personalized, constructive, and grounded in insight — not hype.
- Avoid clichés, filler phrases, or superficial praise.
- The content should read as if written by a seasoned executive coach or leadership strategist.
- Total length: one well-structured page (under 500 words), in paragraph form with no titles or bullet points.
"""

            
            
#             coach_prompt = f"""
# {participant_name}, based on your leadership profile, this page outlines a personalized development plan tailored to your role as a {participant_role} in the {participant_industry} industry. The goal is to help you grow as a leader in ways that align with the unique demands, pace, and culture of your professional environment.

# Identify **3 to 5 key growth areas** drawn from your Innovation and Operations scores — areas that matter most for someone in your position. For each:

# - Start with a short, descriptive title (e.g., “Strategic Communication”) and a one-sentence rationale that directly links the growth area to your responsibilities as a {participant_role} in {participant_industry}.
# - Offer 1–2 practical, real-world actions you can begin implementing — ideally in ways that create value for your team or organization.
# - Briefly reflect on how cultural patterns common in {', '.join(closest_cultures)} may impact how this skill is expressed or received, and suggest how you might navigate these dynamics effectively.

# Maintain a confident and constructive tone, as if coaching {participant_name} personally. End with a motivating message that encourages commitment to intentional practice and ongoing leadership growth. Keep everything to **one page**, in paragraph form — no headers or bullet points.
# ### Tone and Constraints:
# - Frame insights in the second person (e.g., “You may find…” or “You might experience…”) and avoid speaking **as** the participant. Instead, write from an expert coaching perspective, offering guidance with warmth, clarity, and respect.
# - Keep the tone reflective, personalized, constructive, and grounded in insight — not hype.
# - Avoid clichés, filler phrases, or superficial praise.
# - The content should read as if written by a seasoned executive coach or leadership strategist.
# - Total length: one well-structured page (under 500 words), in paragraph form with no titles or bullet points.
# """



            coach_result = llm.predict(coach_prompt)
            # coach_result = llm.invoke([HumanMessage(content=coach_prompt)])

            
            # invite_prompt = """
            # Write a 1-page summary introducing the CALIBER 360-degree leadership inventory.
            # Explain how it exposes biases, highlights cultural fit, tracks progress, and improves self-awareness.
            # Encourage multi-source feedback and close with an invitation to contact admin@caliberleadership.com.
            # Use CALIBER style.
            # """
#             invite_prompt = """
# Craft a one-page introduction to the CALIBER 360-Degree Leadership Inventory.

# Begin by explaining the purpose of the tool: to offer a holistic perspective on leadership by gathering insights from self and others. Emphasize how the process increases self-awareness, uncovers potential biases, tracks leadership growth over time, and identifies cultural alignment.

# Encourage participation from diverse feedback sources—peers, direct reports, and supervisors—to gain a balanced view of leadership strengths and opportunities.

# Conclude with a warm, professional invitation to learn more by contacting admin@caliberleadership.com. Maintain the CALIBER style: clear, concise, and insight-driven.
# """
            # invite_prompt = """
            # Introduce the CALIBER 360-Degree Leadership Inventory in a clear, engaging tone.  Fit within one page.

            # - Explain its purpose: gathering feedback from peers, reports, and supervisors.
            # - Highlight benefits: increased self-awareness, cultural alignment, bias reduction, and progress tracking.
            # - Encourage participants to involve diverse raters and view feedback as a gift.
            # - Close with an invitation to contact ali.lakhani@caliber360ai.com.  Make the contact email address stand out.

            # Address the reader as “you” and maintain CALIBER’s warm, developmental tone.
            # """
            invite_prompt = f"""
As a {participant_role} in the {participant_industry} industry, your leadership is constantly being shaped by how you engage with your team, peers, and organization. To build on the insights in this report, consider taking the next step: participating in the **CALIBER 360-Degree Leadership Inventory**.

This powerful tool gathers confidential, structured feedback from those who work closely with you — including direct reports, peers, and supervisors. The purpose isn’t to evaluate your worth; it’s to help you see your leadership from multiple perspectives and identify actionable ways to grow in your specific professional context.

For a leader in {participant_industry}, the CALIBER 360 offers unique benefits:
- Greater self-awareness tied directly to how you lead in your {participant_role}
- Insight into how your leadership aligns with cultural and organizational norms
- A way to uncover blind spots and reduce personal bias
- A meaningful method to track growth over time with input from others

You're encouraged to invite a diverse set of raters who can speak to your leadership across different contexts. Feedback — even when surprising — is one of the most valuable tools you can receive. It provides the clarity needed to sharpen your influence and deepen your credibility.

If you’re ready to lead with even more purpose and perspective, reach out to **ali.lakhani@caliber360ai.com** to begin your CALIBER 360. We’ll support you through every step with care, professionalism, and confidentiality.

Your leadership journey is uniquely yours — and it doesn’t stop here.

### Tone and Constraints:
- Frame insights in the second person (e.g., “You may find…” or “You might experience…”) and avoid speaking **as** the participant. Instead, write from an expert coaching perspective, offering guidance with warmth, clarity, and respect.
- Keep the tone reflective, personalized, constructive, and grounded in insight — not hype.
- Avoid clichés, filler phrases, or superficial praise.
- The content should read as if written by a seasoned executive coach or leadership strategist.
- Total length: one well-structured page (under 500 words), in paragraph form with no titles or bullet points.
"""
            
            
            
#             invite_prompt = f"""
# As a {participant_role} in the {participant_industry} industry, your leadership is constantly being shaped by how you engage with your team, peers, and organization. To build on the insights in this report, consider taking the next step: participating in the **CALIBER 360-Degree Leadership Inventory**.

# This powerful tool gathers confidential, structured feedback from those who work closely with you — including direct reports, peers, and supervisors. The purpose isn’t to evaluate your worth; it’s to help you see your leadership from multiple perspectives and identify actionable ways to grow in your specific professional context.

# For a leader in {participant_industry}, the CALIBER 360 offers unique benefits:
# - Greater self-awareness tied directly to how you lead in your {participant_role}
# - Insight into how your leadership aligns with cultural and organizational norms
# - A way to uncover blind spots and reduce personal bias
# - A meaningful method to track growth over time with input from others

# You're encouraged to invite a diverse set of raters who can speak to your leadership across different contexts. Feedback — even when surprising — is one of the most valuable tools you can receive. It provides the clarity needed to sharpen your influence and deepen your credibility.

# If you’re ready to lead with even more purpose and perspective, reach out to **ali.lakhani@caliber360ai.com** to begin your CALIBER 360. We’ll support you through every step with care, professionalism, and confidentiality.

# Your leadership journey is uniquely yours — and it doesn’t stop here.

# Frame insights in the second person (e.g., “You may find…” or “You might experience…”) and avoid speaking **as** the participant. Instead, write from an expert coaching perspective, offering guidance with warmth, clarity, and respect.
# """




            invite_result = llm.predict(invite_prompt)
            # invite_result = llm.invoke([HumanMessage(content=invite_prompt)])


            section_dict = {
                "Executive Summary": result,
                "Interpretation of Innovation & Operations Dimensions": page2_result,
                "Cultural Context and Implications": culture_result,
                "Actionable Development Recommendations": coach_result,
                "Invitation to 360-Degree CALIBER Assessment": invite_result
            }


            # def generate_caliber_report_with_cover(
            #     output_path,
            #     participant_name,
            #     report_date,
            #     sections_dict,
            #     plot_path=None,
            #     bar_chart_path=None,
            #     hofstede_path=None
            # ):
            #     doc = SimpleDocTemplate(output_path, pagesize=LETTER,
            #                             rightMargin=72, leftMargin=72,
            #                             topMargin=72, bottomMargin=72)

            #     styles = getSampleStyleSheet()
            #     styles.add(ParagraphStyle(name="Heading", fontSize=14, leading=18, spaceAfter=12, spaceBefore=12, fontName="Helvetica-Bold"))
            #     styles.add(ParagraphStyle(name="Body", fontSize=11, leading=14, spaceAfter=12))
            #     styles.add(ParagraphStyle(name="CoverTitle", fontSize=24, leading=30, spaceAfter=24, alignment=1, fontName="Helvetica-Bold"))
            #     styles.add(ParagraphStyle(name="CoverSub", fontSize=16, leading=20, spaceAfter=12, alignment=1))

            #     story = []

            #     # Cover page
            #     story.append(Spacer(1, 2 * inch))
            #     story.append(Paragraph("CALIBER Leadership Inventory", styles["CoverTitle"]))
            #     story.append(Paragraph(participant_name, styles["CoverSub"]))
            #     story.append(Paragraph(f"Report generated on {report_date}", styles["CoverSub"]))
            #     story.append(PageBreak())

            #     from reportlab.platypus import Image as RLImage

            #     for section, content in sections_dict.items():
            #         story.append(Paragraph(section, styles["Heading"]))
            #         cleaned_content = clean_markdown(content.content)
            #         story.append(Paragraph(cleaned_content, styles["Body"]))
                    
                    
            #         # Optional image logic
            #         # if "Overall Leadership Score" in section and sumplot_path and os.path.exists(sumplot_path):
            #         if "Executive Summary" in section and sumplot_path and os.path.exists(sumplot_path):
            #             story.append(Spacer(1, 0.1 * inch))
            #             story.append(RLImage(sumplot_path, width=6.5*inch, height=1.8*inch))
            #             story.append(Paragraph("Overall Leadership Score", styles["Body"]))
            #             # story.append(PageBreak())
            #         if "Innovation & Operations" in section and bar_chart_path and os.path.exists(bar_chart_path):
            #             story.append(Spacer(1, 0.1 * inch))
            #             story.append(RLImage(bar_chart_path, width=6.5*inch, height=3*inch))
            #             story.append(Paragraph("Leadership Dimension Breakdown", styles["Body"]))
            #             # story.append(PageBreak())
            #         if "Cultural Context" in section and hofstede_path and os.path.exists(hofstede_path):
            #             story.append(Spacer(1, 0.1 * inch))
            #             story.append(RLImage(hofstede_path, width=6.5*inch, height=3*inch))
            #             story.append(Paragraph("Cultural Dimensions Profile (Hofstede)", styles["Body"]))
            #             # story.append(PageBreak())
            #         story.append(Spacer(1, 0.3 * inch))

            #     # template = PageTemplate(id='footer_template', frames=frame, onPage=add_footer)
            #     # doc.addPageTemplates([template])
            #     doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)

            #     # doc.build(story)
            #     return output_path

            def generate_caliber_report_with_cover(
                output_path,
                participant_name,
                report_date,
                sections_dict,
                plot_path=None,
                bar_chart_path=None,
                hofstede_path=None
            ):
                from reportlab.platypus import Image as RLImage

                doc = SimpleDocTemplate(output_path, pagesize=LETTER,
                                        rightMargin=72, leftMargin=72,
                                        topMargin=72, bottomMargin=72)

                styles = getSampleStyleSheet()
                styles.add(ParagraphStyle(name="Heading", fontSize=14, leading=18, spaceAfter=12, spaceBefore=12, fontName="Helvetica-Bold"))
                styles.add(ParagraphStyle(name="Body", fontSize=11, leading=14, spaceAfter=12))
                styles.add(ParagraphStyle(name="CoverTitle", fontSize=24, leading=30, spaceAfter=24, alignment=1, fontName="Helvetica-Bold"))
                styles.add(ParagraphStyle(name="CoverSub", fontSize=16, leading=20, spaceAfter=12, alignment=1))

                story = []

                # Cover page
                story.append(Spacer(1, 2 * inch))
                story.append(Paragraph("CALIBER Leadership Inventory", styles["CoverTitle"]))
                story.append(Paragraph(participant_name, styles["CoverSub"]))
                story.append(Paragraph(f"Report generated on {report_date}", styles["CoverSub"]))
                story.append(PageBreak())

                for section, content in sections_dict.items():
                    story.append(Paragraph(section, styles["Heading"]))

                    # Insert associated chart *before* the section content
                    if "Executive Summary" in section and sumplot_path and os.path.exists(sumplot_path):
                        story.append(Spacer(1, 0.1 * inch))
                        story.append(RLImage(sumplot_path, width=6.5 * inch, height=1.8 * inch))
                        story.append(Spacer(1, 0.2 * inch))
                        story.append(Paragraph("Overall Leadership Score", styles["Body"]))

                    if "Innovation & Operations" in section and bar_chart_path and os.path.exists(bar_chart_path):
                        story.append(Spacer(1, 0.1 * inch))
                        story.append(RLImage(bar_chart_path, width=6.5 * inch, height=3 * inch))
                        story.append(Spacer(1, 0.2 * inch))
                        story.append(Paragraph("Leadership Dimension Breakdown", styles["Body"]))

                    if "Cultural Context" in section and hofstede_path and os.path.exists(hofstede_path):
                        story.append(Spacer(1, 0.1 * inch))
                        story.append(RLImage(hofstede_path, width=6.5 * inch, height=3 * inch))
                        story.append(Spacer(1, 0.2 * inch))
                        story.append(Paragraph("Cultural Dimensions Profile (Hofstede)", styles["Body"]))

                    # Insert section content
                    # cleaned_content = clean_markdown(content.content)
                    cleaned_content = clean_markdown(content)
                    story.append(Spacer(1, 0.2 * inch))
                    story.append(Paragraph(cleaned_content, styles["Body"]))
                    story.append(Spacer(1, 0.3 * inch))

                doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
                return output_path


            generate_caliber_report_with_cover(
                output_path=pdf_filename,
                participant_name=participant_name,
                report_date=report_date,
                sections_dict=section_dict,
                plot_path=sumplot_path,
                bar_chart_path=bar_chart_path,
                hofstede_path=hofstede_path
            )



            # pdf.output(pdf_filename)

            # Display in Streamlit
            with open(pdf_filename, "rb") as f:
                st.download_button(
                    label="📄 Download Full Leadership Report (PDF)",
                    data=f,
                    file_name=pdf_filename,
                    mime="application/pdf"
                )

            csv_drive_id = upload_to_drive(filename, filename, "text/csv", folder_id)
            pdf_drive_id = upload_to_drive(pdf_filename, pdf_filename, "application/pdf", "your-folder-id")

            # pdf_id = upload_to_drive(pdf_filename, pdf_filename, "application/pdf", folder_id)
            # if pdf_id:
            #     st.success("✅ PDF uploaded to Drive!")
            # else:
            #     st.error("❌ PDF upload failed.")

            # st.success("✅ Uploaded to Google Drive!")
            # st.write(f"CSV File ID: {csv_drive_id}")
            # st.write(f"PDF File ID: {pdf_drive_id}")

            # try:
            #     csv_drive_id = upload_to_drive(filename, filename, "text/csv", folder_id)
            #     st.success(f"✅ CSV uploaded to Drive (File ID: {csv_drive_id})")
            # except Exception as e:
            #     st.error(f"❌ CSV upload failed: {e}")

            # try:
            #     pdf_drive_id = upload_to_drive(pdf_filename, pdf_filename, "application/pdf", folder_id)
            #     st.success(f"✅ PDF uploaded to Drive (File ID: {pdf_drive_id})")
            # except Exception as e:
            #     st.error(f"❌ PDF upload failed: {e}")
            
            
            # Show text result
            
            # st.markdown("**Note:** This is only the first page of your CALIBER Leadership Inventory report. It includes an overview of your leadership category, national cultural context, and key development themes. Be sure to review the full report for in-depth scores, your national culture profile, and specific actions and recommendations tailored to you.")

            # st.write(result)

#             # Display top 5 aligned countries
#             st.subheader("🌍 Best Cultural Matches")
#             st.markdown("Your leadership style aligns most closely with these countries:")
#             for country in closest_cultures:
#                 st.markdown(f"- {country}")
#  It includes an overview of your leadership category, national cultural context, and key development themes. Be sure to review the full report for in-depth scores, your national culture profile, and specific actions and recommendations tailored to you.")

#             st.write(result)

            # st.write(result)

            # Display top 5 aligned countries
            # st.subheader("🌍 Best Cultural Matches")
            # st.markdown("Your leadership style aligns most closely with these countries:")
            # for country in closest_cultures:
            #     st.markdown(f"- {country}")


            # # Display bar chart for dimension breakdown
            # try:
            #     bar_image = Image.open(bar_chart_path)
            #     # st.image(bar_image, caption="Dimension Breakdown (Innovation vs Operations)", use_column_width=True)
            # except Exception as e:
            #     # st.warning(f"Could not load bar chart image: {e}")

            # === Display Hofstede Cultural Dimension Chart ===
            # Create and display chart from dimension_custom_scores (Hofstede)
            # hofstede_keys = [
            #     "High Uncertainty Avoidance PCT",
            #     "High Individualism PCT",
            #     "High Power Distance PCT",
            #     "Long-Term Orientation PCT",
            #     "High Masculinity PCT"
            # ]

            # hofstede_scores = [dimension_custom_scores[k] * 100 for k in hofstede_keys]
            # hofstede_labels = [
            #     "Uncertainty Avoidance",
            #     "Individualism",
            #     "Power Distance",
            #     "Long-Term Orientation",
            #     "Masculinity"
            # ]

            # fig, ax = plt.subplots(figsize=(10, 5))
            # sns.barplot(x=hofstede_scores, y=hofstede_labels, palette="Blues_d", ax=ax)
            # ax.set_xlim(0, 100)
            # ax.set_title("Cultural Dimensions Profile (Hofstede)")
            # ax.set_xlabel("Score")
            # ax.set_ylabel("")
            # sns.despine()

            # hofstede_path = f"hofstede_chart_{clean_name}_{timestamp}.png"
            # fig.tight_layout()
            # fig.savefig(hofstede_path, dpi=150)
            # plt.close(fig)

            # try:
            #     # hofstede_img = Image.open(hofstede_path)
            #     # st.image(hofstede_img, caption="Cultural Dimensions Profile (Hofstede)", use_column_width=True)
            # except Exception as e:
            #     # st.warning(f"Could not load Hofstede chart image: {e}")

            # Display Crew-generated content from page 2–4
            # st.subheader("📊 Interpretation of Innovation & Operations Dimensions")
            # st.write(page2_result)

            # st.subheader("🌍 Cultural Context and Implications")
            # # st.write(culture_result)

            # st.subheader("🎯 Actionable Development Recommendations")
            # # st.write(coach_result)

            # report_filename = f"leadership_report_{clean_name}_{timestamp}.txt"
            # with open(report_filename, "w", encoding="utf-8") as f:
            #     f.write(result)

            # with open(pdf_filename, "rb") as f:
            #     st.markdown("<div style='text-align: center; font-size: 0.8em; color: gray; margin-top: 2rem;'>© 2025 M.A. Lakhani. All rights reserved.</div>", unsafe_allow_html=True)

            # with open(pdf_filename, "rb") as f:
            #     st.download_button(
            #         label="📄 Download Leadership Report (PDF)",
            #         data=f,  # ✅ this was missing
            #         file_name=pdf_filename,
            #         mime="application/pdf"
            #     )

