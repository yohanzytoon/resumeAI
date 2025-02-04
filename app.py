import streamlit as st
import openai
import fitz  # PyMuPDF
import docx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import tiktoken
from tenacity import retry, wait_random_exponential, stop_after_attempt

# Configuration
MAX_TOKENS_EMBEDDING = 8191  # text-embedding-ada-002 limit
MAX_TOKENS_GPT = 4096        # gpt-3.5-turbo limit
MODEL_NAME = "gpt-3.5-turbo"

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("OPENAI_API_KEY environment variable not found")
    st.stop()

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(3))
def process_with_retry(func, *args, **kwargs):
    """Generic retry wrapper for OpenAI API calls"""
    return func(*args, **kwargs)

def smart_truncate(text, max_tokens, model_name):
    """Intelligently truncate text preserving complete sentences"""
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    # Find the last sentence boundary within token limit
    decoded = encoding.decode(tokens[:max_tokens])
    last_period = decoded.rfind('. ')
    if last_period != -1:
        return decoded[:last_period+1] + " [truncated]"
    return decoded + " [truncated]"

def extract_text(file):
    """Enhanced text extraction with layout preservation"""
    try:
        if file.name.lower().endswith('.pdf'):
            doc = fitz.open(stream=file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text("text", sort=True)
            return text.strip()
        
        elif file.name.lower().endswith('.docx'):
            return "\n".join(
                [para.text for para in docx.Document(file).paragraphs if para.text.strip()]
            )
        
        elif file.name.lower().endswith('.txt'):
            return file.read().decode("utf-8").strip()
        
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        if file.name.lower().endswith('.pdf'):
            st.info("Tip: Make sure your PDF is text-based, not scanned")
        return ""

@st.cache_data(show_spinner=False)
def get_embedding(text):
    """Get embedding with smart truncation"""
    clean_text = text.strip()
    if not clean_text:
        return None
    
    truncated = smart_truncate(clean_text, MAX_TOKENS_EMBEDDING, "text-embedding-ada-002")
    try:
        response = process_with_retry(
            openai.embeddings.create,
            input=[truncated],
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Embedding failed: {str(e)}")
        return None

def analyze_compatibility(resume_embed, job_embed):
    """Generate comprehensive compatibility analysis"""
    similarity = cosine_similarity(
        np.array(resume_embed).reshape(1, -1),
        np.array(job_embed).reshape(1, -1)
    )[0][0]
    
    # Create visual feedback
    score_percent = min(max(similarity * 100, 0), 100)
    color = "red" if score_percent < 50 else "orange" if score_percent < 75 else "green"
    
    st.markdown(f"""
    <div style="text-align: center; border: 2px solid {color}; padding: 20px; border-radius: 10px;">
        <h3 style="color: {color};">Match Score: {score_percent:.1f}%</h3>
        <progress value="{score_percent}" max="100" style="width: 80%; height: 20px;"></progress>
        <p>{"Needs significant improvement" if score_percent < 50 else 
             "Good match but could be better" if score_percent < 75 else 
             "Excellent match!"}</p>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def generate_enhanced_suggestions(resume_text, job_text):
    """Generate structured suggestions with optimized prompting"""
    truncated_resume = smart_truncate(resume_text, MAX_TOKENS_GPT//2, MODEL_NAME)
    truncated_job = smart_truncate(job_text, MAX_TOKENS_GPT//2, MODEL_NAME)
    
    try:
        response = process_with_retry(
            openai.chat.completions.create,
            model=MODEL_NAME,
            messages=[{
                "role": "system",
                "content": """You are a professional resume optimization expert. Analyze this resume against the job 
                description and provide specific, actionable suggestions in this format:
                
                ### 🔍 Keyword Optimization
                - List missing keywords from job description
                - Suggest natural ways to incorporate them
                
                ### 🛠 Skills Alignment
                - Identify skills gaps
                - Recommend transferable skills to emphasize
                
                ### ✨ Experience Reframing
                - Highlight relevant experience to expand
                - Suggest quantifiable achievements to add
                
                Keep suggestions practical and specific. Use bullet points."""
            }, {
                "role": "user",
                "content": f"RESUME:\n{truncated_resume}\n\nJOB DESCRIPTION:\n{truncated_job}"
            }],
            temperature=0.3,
            max_tokens=800,
            top_p=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Suggestion generation failed: {str(e)}")
        return None

def main():
    st.title("Career Compass: AI Resume Optimizer")
    
    with st.sidebar:
        st.header("Upload Documents")
        resume_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
        job_file = st.file_uploader("Upload Job Description", type=["pdf", "docx", "txt"])
        st.markdown("---")
        st.caption("ℹ️ Supports text-based PDF, DOCX, and TXT files")

    if resume_file and job_file:
        with st.spinner("Analyzing documents..."):
            resume_text = extract_text(resume_file)
            job_text = extract_text(job_file)

            if not resume_text or not job_text:
                st.error("Failed to extract text from one or more files")
                return

            col1, col2 = st.columns(2)
            with col1:
                with st.expander("View Resume Text"):
                    st.write(resume_text[:3000] + "..." if len(resume_text) > 3000 else resume_text)
            with col2:
                with st.expander("View Job Description"):
                    st.write(job_text[:3000] + "..." if len(job_text) > 3000 else job_text)

            resume_embed = get_embedding(resume_text)
            job_embed = get_embedding(job_text)

            if resume_embed and job_embed:
                st.header("Analysis Results")
                analyze_compatibility(resume_embed, job_embed)
                
                st.markdown("---")
                st.subheader("Optimization Recommendations")
                suggestions = generate_enhanced_suggestions(resume_text, job_text)
                
                if suggestions:
                    st.markdown(suggestions)
                    st.markdown("---")
                    st.caption("💡 Apply these suggestions and re-upload your resume for improved results!")
                else:
                    st.warning("Could not generate suggestions. Please try again.")

if __name__ == "__main__":
    main()