import streamlit as st
import os
import requests
import tempfile
import re
import google.generativeai as genai
from pypdf import PdfReader
from fpdf import FPDF
from deep_translator import GoogleTranslator
import io # Required for handling PDF bytes

# --- ResearchAgent Class (Slightly Modified for Streamlit Context) ---

class ResearchAgent:
    def __init__(self, api_key):
        # Configure Gemini API Key securely
        try:
            genai.configure(api_key=api_key)
            # Test connection briefly (optional, but good practice)
            # _ = genai.GenerativeModel("gemini-1.5-flash") # Use a valid model
        except Exception as e:
            st.error(f"Failed to configure Gemini API: {e}")
            raise ConnectionError("Gemini API configuration failed.") from e

        self.semantic_scholar_api = "https://api.semanticscholar.org/graph/v1/paper/search"
        self.temp_files = [] # Keep track of temp files to clean up

    def _cleanup_temp_files(self):
        """Removes temporary files created during processing."""
        for f_path in self.temp_files:
            try:
                if os.path.exists(f_path):
                    os.remove(f_path)
                    # st.write(f"Cleaned up temp file: {f_path}") # Optional debug message
            except Exception as e:
                st.warning(f"Could not remove temporary file {f_path}: {e}")
        self.temp_files = []


    def download_arxiv_pdf(self, arxiv_url):
        """Download a paper from Arxiv given its URL."""
        try:
            arxiv_id_match = re.search(r'(\d{4}\.\d{4,5})', arxiv_url) # More robust regex
            if not arxiv_id_match:
                 # Try extracting last part if specific pattern fails
                 parts = arxiv_url.split("/")
                 if parts and parts[-1]:
                     arxiv_id = parts[-1].replace(".pdf", "")
                 else:
                    raise ValueError("Could not extract ArXiv ID from URL.")
            else:
                 arxiv_id = arxiv_id_match.group(1)

            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            response = requests.get(pdf_url, stream=True, timeout=30) # Added stream and timeout
            response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)

            # Create a temporary file safely
            temp_fd, temp_pdf_path = tempfile.mkstemp(suffix=".pdf")
            self.temp_files.append(temp_pdf_path) # Track for cleanup

            with os.fdopen(temp_fd, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            st.info(f"Downloaded Arxiv PDF: {os.path.basename(temp_pdf_path)}")
            return temp_pdf_path

        except requests.exceptions.RequestException as e:
            st.error(f"Failed to download Arxiv PDF from {pdf_url}: {e}")
            raise
        except ValueError as e:
             st.error(f"Invalid ArXiv URL format: {e}")
             raise
        except Exception as e:
            st.error(f"An unexpected error occurred during download: {e}")
            raise


    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a given PDF file."""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            total_pages = len(reader.pages)
            progress_bar = st.progress(0)
            for i, page in enumerate(reader.pages):
                extracted = page.extract_text()
                if extracted: # Append only if text was extracted
                    text += extracted + "\n"
                progress_bar.progress((i + 1) / total_pages)

            if not text.strip():
                st.warning("No text could be extracted from the PDF. It might be image-based or protected.")
                return ""
            return text.strip()
        except Exception as e:
            st.error(f"Failed to extract text from PDF: {e}")
            raise


    def generate_text_with_gemini(self, prompt):
        """Generate text using Google Gemini LLM."""
        try:
            # Ensure model name is current - check Gemini documentation if needed
            # model = genai.GenerativeModel("gemini-1.5-flash") # Example model
            # Trying gemini-pro as per previous examples, adjust if needed
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            # Handle potential lack of content or different response structure
            if response and hasattr(response, 'text'):
                 return response.text
            elif response and response.parts:
                 # Handle potential multi-part responses if necessary
                 return "".join(part.text for part in response.parts if hasattr(part, 'text'))
            else:
                 st.warning("Gemini response structure might have changed or response is empty.")
                 return "No valid response text received from Gemini."
        except Exception as e:
            st.error(f"Error interacting with Gemini API: {e}")
            # Consider more specific error handling based on Gemini API exceptions
            raise


    def summarize_text(self, text, language="en"):
        """Generate a summary of the extracted text using Gemini."""
        if not text: return "Cannot summarize empty text."

        # Adjust truncation based on model limits - check Gemini docs
        # Using a smaller chunk size for safety
        max_chars = 30000 # Rough estimate, depends on model (gemini-pro is ~32k tokens)
        truncated_text = text[:max_chars]
        if len(text) > max_chars:
             st.warning(f"Text truncated to {max_chars} characters for summarization.")

        prompt = f"Please provide a concise summary of the key points from the following research paper content:\n\n---\n{truncated_text}\n---"
        summary = self.generate_text_with_gemini(prompt)

        # Translate if needed
        if language.lower() not in ["en", "english", ""]:
            try:
                st.info(f"Translating summary to {language}...")
                summary = GoogleTranslator(source="auto", target=language).translate(summary)
            except Exception as e:
                st.error(f"Failed to translate summary: {e}")
                # Fallback to English summary
                st.warning("Displaying summary in English.")

        return summary


    def extract_key_insights(self, text):
        """Extract key insights and main findings from the paper using Gemini."""
        if not text: return "Cannot extract insights from empty text."

        max_chars = 30000 # Match truncation limit for consistency if needed
        truncated_text = text[:max_chars]
        if len(text) > max_chars:
             st.warning(f"Text truncated to {max_chars} characters for insight extraction.")

        prompt = f"Analyze the following research paper text and list the main findings, key insights, and conclusions as bullet points:\n\n---\n{truncated_text}\n---"
        return self.generate_text_with_gemini(prompt)


    def extract_statistics(self, text):
        """Extract key statistics and numerical data from the research paper."""
        if not text: return "No text provided for statistics extraction."
        # Regular expression to identify potential stats (can be improved)
        # Looks for numbers with decimals, percentages, integers near context words.
        stats = re.findall(
            r'\b\d{1,3}(?:,\d{3})*\.\d+%?' +  # Numbers with decimals and optional % (e.g., 12.34, 1,234.5%)
            r'|\b\d+%|\b\d{1,3}(?:,\d{3})*\b' + # Percentages (e.g., 50%) or whole numbers
            r'|p\s*<\s*0\.\d+' + # p-values like p < 0.05
            r'|(?<=[=<>±])\s*\d+\.?\d*', # Numbers after =, <, >, ±
            text
        )
        # Simple filtering to remove very common/simple numbers unless they are percentages
        filtered_stats = [s for s in stats if '%' in s or (len(s) > 2 and '.' in s) or len(s) > 4 or re.search(r'[=<>±p]', s, re.IGNORECASE)]

        unique_stats = sorted(list(set(filtered_stats)), key=len, reverse=True)

        return "\n".join(unique_stats) if unique_stats else "No specific key statistics found using pattern matching."


    def save_summary_as_pdf(self, title, summary, insights):
        """Save the summary, key insights, and statistics as a PDF file in memory."""
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Add Unicode font if needed for different languages
        try:
            # Try adding a common Unicode font (ensure it's installed or provide path)
            # pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True) # Example, requires font file
            # pdf.set_font('DejaVu', size=10)
            # If font isn't available, fall back to Arial, which may not support all chars
            pdf.set_font("Arial", size=10)
        except RuntimeError:
            st.warning("Unicode font not found. Falling back to Arial. Some characters might not display correctly in the PDF.")
            pdf.set_font("Arial", size=10)


        # Title
        pdf.set_font("Arial", style="B", size=14)
        # Handle potential encoding issues in title
        safe_title = title.encode('latin-1', 'replace').decode('latin-1')
        pdf.cell(0, 10, f"Summary: {safe_title}", ln=True, align="C")
        pdf.ln(10)

        # Summary Section
        pdf.set_font("Arial", style="B", size=11)
        pdf.cell(0, 10, "Summary:", ln=True)
        pdf.set_font("Arial", size=10)
        # Encode text safely for FPDF which often expects latin-1
        safe_summary = summary.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 7, safe_summary) # Use 0 for width to fill page
        pdf.ln(5)

        # Insights Section
        pdf.set_font("Arial", style="B", size=11)
        pdf.cell(0, 10, "Key Insights:", ln=True)
        pdf.set_font("Arial", size=10)
        safe_insights = insights.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 7, safe_insights)
        pdf.ln(5)

        # Return PDF content as bytes
        pdf_output_bytes = pdf.output(dest='S').encode('latin-1') # 'S' returns as string, encode it
        return pdf_output_bytes


    def find_related_papers(self, query_text):
        """Find related papers using Semantic Scholar API."""
        if not query_text: return []
        # Use a small, relevant part of the text for the query
        query = query_text[:500] # Limit query length
        params = {
            "query": query,
            "fields": "title,authors,url,year,abstract", # Added abstract
            "limit": 5
        }
        try:
            response = requests.get(self.semantic_scholar_api, params=params, timeout=20)
            response.raise_for_status()
            papers = response.json().get("data", [])
            return papers
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch related papers from Semantic Scholar: {e}")
            return []
        except Exception as e:
            st.error(f"An error occurred while fetching related papers: {e}")
            return []

# --- Streamlit App UI ---

st.set_page_config(page_title="Research Paper Summarizer", layout="wide")
st.title("📄 Research Paper Summarizer & Analyzer")

# --- Inputs Sidebar ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Google Gemini API Key:", type="password")
    st.markdown("[Get a Gemini API key](https://makersuite.google.com/app/apikey)") # Updated link

    st.header("Input")
    input_path_or_url = st.text_input("Enter local PDF path or ArXiv URL:", placeholder="e.g., https://arxiv.org/abs/2305.15334 or /path/to/paper.pdf")

    languages = {
        "English": "en", "Spanish": "es", "French": "fr",
        "German": "de", "Italian": "it", "Portuguese": "pt",
        "Hindi": "hi", "Chinese (Simplified)": "zh-cn"
        # Add more languages as needed
    }
    lang_name = st.selectbox("Select Summary Language:", options=languages.keys())
    language_code = languages[lang_name]

    process_button = st.button("Analyze Paper", type="primary")

# --- Main Processing Area ---

if process_button:
    if not api_key:
        st.error("Please enter your Google Gemini API Key in the sidebar.")
    elif not input_path_or_url:
        st.error("Please enter a PDF path or ArXiv URL in the sidebar.")
    else:
        agent = None # Initialize agent variable
        pdf_path = None
        try:
            # Instantiate agent only if API key is provided
            agent = ResearchAgent(api_key=api_key)

            # --- Step 1: Get PDF ---
            with st.spinner("Fetching and verifying PDF..."):
                if input_path_or_url.startswith("http"):
                    if "arxiv.org" in input_path_or_url:
                         # Assume it's an abstract URL, try to convert to PDF URL or let download handle it
                         if "/abs/" in input_path_or_url:
                              input_path_or_url = input_path_or_url.replace("/abs/", "/pdf/")
                         if not input_path_or_url.endswith(".pdf"):
                              input_path_or_url += ".pdf" # Append .pdf if missing
                         pdf_path = agent.download_arxiv_pdf(input_path_or_url)
                    else:
                         st.error("Currently, only ArXiv URLs are supported for direct download.")
                         st.stop() # Stop execution if not Arxiv

                elif os.path.exists(input_path_or_url) and input_path_or_url.lower().endswith(".pdf"):
                     pdf_path = input_path_or_url
                     st.info(f"Using local PDF: {os.path.basename(pdf_path)}")
                else:
                     st.error("Invalid input: Please provide a valid ArXiv URL or an existing local PDF path.")
                     st.stop() # Stop execution

            if pdf_path:
                # --- Step 2: Extract Text ---
                extracted_text = ""
                with st.spinner("Extracting text from PDF... (This may take a moment)"):
                    extracted_text = agent.extract_text_from_pdf(pdf_path)

                if extracted_text:
                    st.success("Text extracted successfully.")

                    col1, col2 = st.columns(2)

                    with col1:
                        # --- Step 3: Summarize ---
                        summary = ""
                        with st.spinner("Generating summary..."):
                             summary = agent.summarize_text(extracted_text, language_code)
                        st.subheader(f"📝 Summary ({lang_name})")
                        st.markdown(summary) # Use markdown for better formatting potential

                    with col2:
                         # --- Step 4: Extract Insights ---
                        insights = ""
                        with st.spinner("Extracting key insights..."):
                             insights = agent.extract_key_insights(extracted_text)
                        st.subheader("💡 Key Insights")
                        st.markdown(insights) # Use markdown

                    # --- Step 6: Generate PDF Report ---
                    pdf_bytes = None
                    with st.spinner("Generating PDF report..."):
                        # Extract a title from the PDF name or URL
                        pdf_basename = os.path.basename(input_path_or_url).replace('.pdf', '')
                        pdf_bytes = agent.save_summary_as_pdf(pdf_basename, summary, insights)

                    if pdf_bytes:
                        st.download_button(
                            label="Download Summary PDF",
                            data=pdf_bytes,
                            file_name=f"{pdf_basename}_summary.pdf",
                            mime="application/pdf"
                        )

                    # --- Step 7: Find Related Papers ---
                    related_papers = []
                    with st.spinner("Finding related papers..."):
                        # Use extracted text or abstract if available for better query
                        related_papers = agent.find_related_papers(extracted_text) # Pass actual text

                    if related_papers:
                        st.subheader("🔍 Related Papers (via Semantic Scholar)")
                        for i, paper in enumerate(related_papers):
                             title = paper.get('title', 'N/A')
                             authors = ", ".join(a['name'] for a in paper.get('authors', [])) if paper.get('authors') else 'N/A'
                             year = paper.get('year', 'N/A')
                             url = paper.get('url', '#')
                             abstract = paper.get('abstract', 'No abstract available.')
                             st.markdown(f"**{i+1}. {title}** ({year})")
                             st.markdown(f"*Authors: {authors}*")
                             if url != '#':
                                 st.markdown(f"[Link]({url})")
                             with st.expander("Show Abstract"):
                                  st.write(abstract)
                    else:
                        st.info("Could not find related papers based on the initial text.")

                else:
                     st.error("Text extraction failed or PDF contained no text. Cannot proceed.")

        except ConnectionError as e:
             st.error(f"API Configuration Error: {e}") # Handle Gemini config error
        except Exception as e:
            st.error(f"An unexpected error occurred during processing: {e}")
            # Optionally add more detailed logging here for debugging
            st.error("Processing stopped.")
        finally:
             # Ensure cleanup happens even if errors occur mid-process
             if agent:
                  agent._cleanup_temp_files()


else:
    st.info("Enter your API key, PDF path/URL, language, and click 'Analyze Paper' to begin.")

# Add a footer or instructions
st.markdown("---")
st.markdown("Built with Streamlit and Google Gemini.")
st.markdown("**Note:** Ensure the provided PDF path is accessible by the server running Streamlit, or use an ArXiv URL.")
st.markdown("**Security:** For deployed applications, use Streamlit Secrets to manage your API key securely.")