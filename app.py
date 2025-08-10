import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

from src.utils import load_jobs_from_file
from src.embed_utils import embed_text, embed_texts
from src.groq_utils import generate_groq_recommendation  # your Groq API wrapper


st.title("Job Recommendation Demo with Resume Upload")

# Load jobs once at the start
jobs = load_jobs_from_file("data/jobs.json")
st.write(f"Loaded {len(jobs)} jobs from scraped data.")

# Upload resume or enter skills manually
uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    # Extract text from PDF resume
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    resume_text = ""
    for page in pdf_reader.pages:
        resume_text += page.extract_text() or ""

    st.subheader("Extracted Resume Text")
    st.text_area("Resume content extracted", resume_text, height=200)

    user_input = resume_text
else:
    user_input = st.text_area("Or enter your skills or job preferences manually", height=150)

if st.button("Get Recommendations") and user_input.strip() != "":
    with st.spinner("Finding best jobs for you..."):
        # Embed user input and job descriptions locally
        user_embedding = embed_text(user_input)
        job_texts = [job["text"] for job in jobs]
        job_embeddings = embed_texts(job_texts)

        # Calculate cosine similarity and get top 5 jobs
        similarities = cosine_similarity([user_embedding], job_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:5]
        top_jobs = [jobs[i] for i in top_indices]

    st.subheader("Top Job Matches")
    for job in top_jobs:
        st.markdown(f"### {job.get('title', 'No Title')}")
        st.markdown(f"**Company:** {job.get('company', 'Unknown')}")
        st.markdown(f"**Location:** {job.get('location', 'Not Specified')}")
        st.markdown(f"{job.get('text', '')[:300]}...")  # short description
        st.markdown("---")

    # Groq API personalized recommendation explanation (optional)
    try:
        matched_texts = [job["text"] for job in top_jobs]
        recommendation = generate_groq_recommendation(user_input, matched_texts)
        st.subheader("Personalized Job Recommendations")
        st.write(recommendation)
    except Exception as e:
        st.warning(f"Groq API error or unavailable: {e}")
        st.info("Showing top job matches without personalized explanation.")

else:
    st.info("Enter your skills/preferences or upload your resume, then click 'Get Recommendations'")




st.markdown(
    """
    <style>
    /* Change background color */
    .stApp {
        background-color: #000000;  /* black */
        color: white;
    }

    /* Change text colors */
    .css-1d391kg, .css-ffhzg2 { 
        color: white;
    }

    /* Style headers */
    h1, h2, h3, h4, h5, h6 {
        color: white;
    }

    /* Style sidebar (if you have one) */
    [data-testid="stSidebar"] {
        background-color: #121212;
        color: white;
    }

    /* Inputs and buttons */
    .stTextInput>div>div>input {
        background-color: #222222;
        color: white;
        border: 1px solid #555555;
    }

    .stButton>button {
        background-color: #333333;
        color: white;
        border: none;
    }

    /* Scrollbar for better visibility */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background-color: #555555;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)





# import streamlit as st
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# from src.utils import load_jobs_from_file
# from src.embed_utils import embed_text, embed_texts
# from src.recommendation import generate_recommendations  # this still uses OpenAI or can be adapted later
# import PyPDF2
# import io


# st.title("Job Recommendation Demo with Resume Upload")
# if st.button("Get Recommendations") and user_input.strip() != "":
#     with st.spinner("Finding best jobs for you..."):
#         user_embedding = embed_text(user_input)
#         job_texts = [job["text"] for job in jobs]
#         job_embeddings = embed_texts(job_texts)

#         similarities = cosine_similarity([user_embedding], job_embeddings)[0]
#         top_indices = np.argsort(similarities)[::-1][:5]
#         top_jobs = [jobs[i] for i in top_indices]

#     st.subheader("Top Job Matches")
#     for job in top_jobs:
#         st.markdown(f"### {job.get('title', 'No Title')}")
#         st.markdown(f"**Company:** {job.get('company', 'Unknown')}")
#         st.markdown(f"**Location:** {job.get('location', 'Not Specified')}")
#         st.markdown(f"{job.get('text', '')[:300]}...")  # short description
#         st.markdown("---")



# uploaded_file = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])

# if uploaded_file is not None:
#     # Read PDF content
#     pdf_reader = PyPDF2.PdfReader(uploaded_file)
#     resume_text = ""
#     for page in pdf_reader.pages:
#         resume_text += page.extract_text() or ""

#     st.subheader("Extracted Resume Text")
#     st.text_area("Resume content extracted", resume_text, height=200)

#     # Use resume text as user input instead of manual text
#     user_input = resume_text
# else:
#     user_input = st.text_area("Or enter your skills or job preferences manually", height=150)


# # Load jobs once
# jobs = load_jobs_from_file("data/jobs.json")
# st.write(f"Loaded {len(jobs)} jobs from scraped data.")

# user_input = st.text_area("Enter your skills or job preferences", height=150)

# if st.button("Get Recommendations") and user_input.strip() != "":
#     # Embed user input and job descriptions locally
#     user_embedding = embed_text(user_input)
#     job_texts = [job["text"] for job in jobs]
#     job_embeddings = embed_texts(job_texts)

#     # Calculate cosine similarity and get top 5 jobs
#     similarities = cosine_similarity([user_embedding], job_embeddings)[0]
#     top_indices = np.argsort(similarities)[::-1][:5]
#     top_jobs = [jobs[i] for i in top_indices]

#     # Prepare job texts for recommendation prompt
#     matched_texts = [job["text"] for job in top_jobs]

#     # For demo, skip or mock this if you don't want OpenAI API here
#     try:
#         recommendation = generate_recommendations(user_input, matched_texts, None)
#     except Exception:
#         recommendation = "Personalized recommendations unavailable (API quota or key missing). Showing top matches:\n\n"
#         recommendation += "\n\n".join([f"- {job['text'][:200]}..." for job in top_jobs])

#     st.subheader("Personalized Job Recommendations")
#     st.write(recommendation)

# else:
#     st.info("Enter your skills/preferences and click 'Get Recommendations'")



