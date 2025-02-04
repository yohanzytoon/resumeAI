# Career Compass: AI Resume Optimizer

Career Compass is an AI-powered resume optimization tool that helps job seekers analyze and enhance their resumes by comparing them with job descriptions. The application uses OpenAI's language models to generate actionable, structured suggestions and provides a visual match score based on semantic similarity between the resume and the job description.

## Features

- **Document Extraction:**  
  Supports text-based PDFs, DOCX, and TXT files for both resumes and job descriptions.

- **Smart Truncation:**  
  Intelligently truncates text to stay within token limits while preserving complete sentences.

- **Semantic Embeddings:**  
  Uses OpenAI's `text-embedding-ada-002` to generate high-dimensional embeddings of the documents.

- **Compatibility Analysis:**  
  Computes cosine similarity between resume and job description embeddings and visualizes the match score.

- **Enhanced Suggestions:**  
  Leverages GPT-3.5-turbo to generate structured, actionable suggestions for resume optimization in key areas such as keyword optimization, skills alignment, and experience reframing.

- **Retry Mechanism:**  
  Implements retry logic (using the `tenacity` package) to handle transient API errors.

## Requirements

- Python 3.8 or higher
- [Streamlit](https://streamlit.io/)
- [OpenAI Python Library](https://github.com/openai/openai-python) (latest version)
- [PyMuPDF](https://pymupdf.readthedocs.io) for PDF text extraction
- [python-docx](https://python-docx.readthedocs.io) for DOCX text extraction
- [NumPy](https://numpy.org/) and [scikit-learn](https://scikit-learn.org/) for similarity computation
- [tiktoken](https://github.com/openai/tiktoken) for token encoding
- [tenacity](https://tenacity.readthedocs.io/) for API call retries

## Setup & Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yohanzytoon/resumeAI.git
   cd resumeAI
   ```

2. **Create a Conda Environment:**

   ```bash
   conda create -n ai_resume_analyzer python=3.8
   conda activate ai_resume_analyzer
   ```

3. **Install Dependencies:**

   Make sure you have the `requirements.txt` file with the following content:

   ```txt
   streamlit
   openai
   PyMuPDF
   python-docx
   scikit-learn
   numpy
   tiktoken
   tenacity
   ```

   Then run:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Your OpenAI API Key:**

   Export your OpenAI API key as an environment variable. For example, on Linux/Mac:

   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   ```

   On Windows:

   ```cmd
   set OPENAI_API_KEY=your_openai_api_key_here
   ```

## Running the Application

Start the Streamlit app by running the following command in your terminal:

```bash
streamlit run app.py
```

Once the application is running, use the sidebar to upload your resume and job description files. The app will extract text, generate embeddings, calculate a match score, and display optimization recommendations.

## Project Structure

- **app.py:** Main application file containing the Streamlit UI, text extraction, embedding generation, compatibility analysis, and suggestion generation functions.
- **requirements.txt:** Lists all the required Python packages.
- **README.md:** This documentation file.

## Troubleshooting

- **Text Extraction Issues:**  
  Ensure your PDFs are text-based and not scanned images. For scanned documents, consider using OCR tools.

- **OpenAI API Errors:**  
  The app includes retry logic for transient errors. If you continue to experience issues, verify your API key and review your OpenAI account usage/quota.

## License

This project is open-source and available under the [MIT License](LICENSE).

## Acknowledgments

- OpenAI for providing powerful language models and APIs.
- The developers of Streamlit, PyMuPDF, python-docx, NumPy, scikit-learn, tiktoken, and tenacity for their excellent tools and libraries.

