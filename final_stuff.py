import streamlit as st
from PIL import Image
import pytesseract
import os
import time
import torch
from io import BytesIO
import base64
from pathlib import Path
from dotenv import load_dotenv
from googletrans import Translator
from gtts import gTTS

# Try to import optional dependencies
try:
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    transformers_available = True
except ImportError:
    transformers_available = False

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.output_parsers import StrOutputParser
    langchain_available = True
except ImportError:
    langchain_available = False

try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from huggingface_hub import login
    from docling_core.types.doc import DoclingDocument
    from docling_core.types.doc.document import DocTagsDocument
    docling_available = True
except ImportError:
    docling_available = False

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory() if langchain_available else None
if 'ocr_results' not in st.session_state:
    st.session_state.ocr_results = None
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""

def check_dependencies():
    """Check for missing dependencies"""
    missing = []
    
    # Check for basic dependencies
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        missing.append("tesseract-ocr")
    
    # Check for optional dependencies
    if not transformers_available:
        missing.append("transformers huggingface_hub")
    if not langchain_available:
        missing.append("langchain_google_genai langchain_core langchain_community")
    if not docling_available:
        missing.append("docling-core")
    
    return missing

def initialize_chat():
    """Initialize chat components if available"""
    if not langchain_available or not GOOGLE_API_KEY:
        return None
        
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=GOOGLE_API_KEY,
            transport="rest"
        )
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful assistant that analyzes OCR results and helps users understand the extracted text."
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{user_input}"),
        ])
        return prompt | llm | StrOutputParser()
    except Exception as e:
        st.error(f"Error initializing chat: {str(e)}")
        return None

def process_image_basic(image):
    """Basic OCR using pytesseract"""
    try:
        text = pytesseract.image_to_string(image, lang='en')
        return [line.strip() for line in text.split('\n') if line.strip()]
    except Exception as e:
        st.error(f"Error in basic OCR: {str(e)}")
        return []

def process_image_docling(image, prompt_text="Convert this page to docling."):
    """Process image using SmolDocling"""
    if not docling_available:
        st.error("SmolDocling dependencies are not installed")
        return None, None
        
    if not HF_TOKEN:
        st.warning("HF_TOKEN not found in .env file. Authentication may fail.")
    else:
        try:
            login(token=HF_TOKEN)
        except Exception as e:
            st.error(f"Hugging Face login error: {str(e)}")
            return None, None
    
    # Check for CUDA and use CPU if not available
    device = "cpu"  # Force CPU to avoid CUDA memory errors
    
    try:
        start_time = time.time()
        
        # Load processor and model with error handling
        try:
            processor = AutoProcessor.from_pretrained("ds4sd/SmolDocling-256M-preview")
            model = AutoModelForVision2Seq.from_pretrained(
                "ds4sd/SmolDocling-256M-preview",
                torch_dtype=torch.float32,
            ).to(device)
        except Exception as e:
            st.error(f"Error loading SmolDocling model: {str(e)}")
            return None, None
        
        # Create input messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]
            },
        ]
        
        # Prepare inputs
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")
        inputs = inputs.to(device)
        
        # Generate outputs with optimized settings
        with torch.no_grad():  # Disable gradient calculation to save memory
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=800,  # Reduced for stability
                do_sample=False      # Deterministic generation
            )
        
        prompt_length = inputs.input_ids.shape[1]
        trimmed_generated_ids = generated_ids[:, prompt_length:]
        doctags = processor.batch_decode(
            trimmed_generated_ids,
            skip_special_tokens=False,
        )[0].lstrip()
        
        # Clean the output
        doctags = doctags.replace("<end_of_utterance>", "").strip()
        
        # Populate document
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
        
        # Create a docling document
        doc = DoclingDocument(name="Document")
        doc.load_from_doctags(doctags_doc)
        
        # Export as markdown
        md_content = doc.export_to_markdown()
        
        processing_time = time.time() - start_time
        
        return doctags, md_content, processing_time
    except Exception as e:
        st.error(f"Error in SmolDocling processing: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, 0

def analyze_image_content(image, question="What is shown in this image?"):
    """Analyze image content using Gemini if available"""
    if not langchain_available or not GOOGLE_API_KEY:
        st.warning("Image analysis requires langchain and Google API key")
        return None
        
    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            transport="rest",
            temperature=0.2,
            max_output_tokens=800
        )
        
        # Convert and resize image
        max_size = (800, 800)  # Reduced size for stability
        if image.width > max_size[0] or image.height > max_size[1]:
            image.thumbnail(max_size, Image.LANCZOS)
        
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{question} Please provide a concise description (under 200 words)."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}"
                    }
                }
            ]
        }
        
        response = model.invoke(input=[message])
        
        # Clean up response
        import re
        cleaned_text = re.sub(r'\n+', '\n', response.content)
        cleaned_text = re.sub(r'\s+\.', '.', cleaned_text)
        
        return cleaned_text
        
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def get_language_menu():
    """Return dictionary of supported languages"""
    return {
        'English': 'en',
        'Hindi': 'hi',
        'Spanish': 'es',
        'French': 'fr',
        'German': 'de',
        'Chinese': 'zh-CN',
        'Japanese': 'ja',
        'Korean': 'ko',
        'Arabic': 'ar',
        'Russian': 'ru',
        'Italian': 'it',
        'Portuguese': 'pt',
        'Bengali': 'bn',
        'Tamil': 'ta',
        'Telugu': 'te',
        'Marathi': 'mr',
        'Gujarati': 'gu'
    }

def translate_text(text, target_language_code):
    """Translate text with improved error handling"""
    if not text or text.strip() == "" or target_language_code == 'en':
        return text

    try:
        translator = Translator()
        # Split text into smaller chunks to avoid translation errors
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        translated_chunks = []

        for chunk in chunks:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        time.sleep(1.5)  # Wait between retries
                    
                    translation = translator.translate(chunk, dest=target_language_code, src='en')
                    
                    if hasattr(translation, 'text'):
                        translated_chunks.append(translation.text)
                        break
                    else:
                        # If translation failed, continue to retry
                        continue

                except Exception:
                    time.sleep(1)  # Wait after error

        return ' '.join(translated_chunks) if translated_chunks else text

    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text

def generate_audio(text, language_code='en'):
    """Generate audio from text"""
    try:
        if not text or text.strip() == "":
            return None
            
        audio_bytes_io = BytesIO()
        tts = gTTS(text=text[:3000], lang=language_code, slow=False)  # Limit text length
        tts.write_to_fp(audio_bytes_io)
        audio_bytes_io.seek(0)
        return audio_bytes_io.getvalue()
    except Exception as e:
        st.error(f"Audio generation error: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Unified OCR App", layout="wide")
    
    st.title("📝 Unified OCR Application")
    
    # Check for missing dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        st.warning(f"Missing dependencies: {', '.join(missing_deps)}. Some features may be limited.")
        st.info("Install with: pip install " + " ".join(missing_deps))
    
    # Initialize session state
    if 'chain' not in st.session_state and langchain_available:
        st.session_state.chain = initialize_chat()
        if st.session_state.chain is None:
            st.error("Failed to initialize chat. Check your Google API key.")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        ocr_method = st.radio(
            "Choose OCR Method",
            ["Basic OCR", "SmolDocling OCR"],
            help="Basic OCR uses Tesseract, SmolDocling uses advanced AI models"
        )
        
        if ocr_method == "SmolDocling OCR":
            task_type = st.selectbox(
                "Select task type",
                [
                    "Convert this page to docling.",
                    "Convert this table to OTSL.",
                    "Convert code to text.",
                    "Convert formula to latex.",
                    "Convert chart to OTSL.",
                    "Extract all section header elements on the page."
                ]
            )
        
        # Camera option
        st.markdown("---")
        use_camera = st.checkbox("Use Camera Input")
        
        # Language settings
        st.markdown("---")
        st.subheader("Language Settings")
        languages = get_language_menu()
        selected_language = st.selectbox(
            "Select Output Language",
            options=list(languages.keys()),
            index=list(languages.keys()).index('English')
        )
    
    # Main content area
    if use_camera:
        camera_image = st.camera_input("Take a picture")
        if camera_image is not None:
            try:
                image = Image.open(camera_image).convert("RGB")  # Ensure RGB mode
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="Camera Image", use_container_width=True)
                
                with col2:
                    question = st.text_input("Ask about the image:", "What is shown in this image?")
                    
                    if st.button("Analyze Image"):
                        with st.spinner("Analyzing image..."):
                            analysis = analyze_image_content(image, question)
                            if analysis:
                                st.markdown("### 🔍 Analysis (English):")
                                st.write(analysis)
                                
                                tts_text = analysis
                                
                                if selected_language != "English":
                                    with st.spinner(f"Translating to {selected_language}..."):
                                        translated_analysis = translate_text(analysis, languages[selected_language])
                                        if translated_analysis:
                                            st.markdown(f"### 🌐 Analysis in {selected_language}:")
                                            st.write(translated_analysis)
                                            tts_text = translated_analysis
                                
                                audio_data = generate_audio(tts_text, languages[selected_language])
                                if audio_data:
                                    st.markdown("### 🔊 Text-to-Speech")
                                    st.audio(audio_data, format='audio/mp3')
                                    
                                    st.download_button(
                                        label="💾 Download Audio",
                                        data=audio_data,
                                        file_name=f"analysis_{selected_language}.mp3",
                                        mime="audio/mp3"
                                    )
            except Exception as e:
                st.error(f"Error processing camera image: {str(e)}")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Or choose an image...", 
        type=["jpg", "jpeg", "png", "bmp"]
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")  # Ensure RGB mode
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                if ocr_method == "Basic OCR":
                    with st.spinner("Processing with Basic OCR..."):
                        ocr_results = process_image_basic(image)
                        
                        if ocr_results:
                            # Store in session state
                            st.session_state.ocr_results = ocr_results
                            text_content = "\n".join(ocr_results)
                            st.session_state.extracted_text = text_content
                            
                            st.markdown("### 📄 Extracted Text:")
                            st.text_area("Results", text_content, height=250)
                            
                            st.download_button(
                                label="💾 Download Text",
                                data=text_content,
                                file_name="ocr_results.txt",
                                mime="text/plain"
                            )
                else:  # SmolDocling OCR
                    with st.spinner("Processing with SmolDocling..."):
                        doctags, md_content, processing_time = process_image_docling(image, task_type)
                        if doctags and md_content:
                            # Store in session state
                            st.session_state.extracted_text = md_content
                            
                            st.markdown("### 📄 SmolDocling Results:")
                            st.markdown(md_content)
                            
                            st.download_button(
                                label="💾 Download Markdown",
                                data=md_content,
                                file_name="docling_results.md",
                                mime="text/markdown"
                            )
                            
                            st.success(f"Processing completed in {processing_time:.2f} seconds")
            
            # Text-to-speech and translation section
            if st.session_state.extracted_text:
                st.markdown("---")
                st.markdown("### 🔊 Text-to-Speech & Translation")
                
                text_to_process = st.session_state.extracted_text
                
                # Translate if needed
                if selected_language != "English":
                    with st.spinner(f"Translating to {selected_language}..."):
                        translated_text = translate_text(text_to_process, languages[selected_language])
                        if translated_text:
                            st.markdown(f"### 🌐 Text in {selected_language}:")
                            st.text_area("Translated Results", translated_text, height=250)
                            
                            st.download_button(
                                label=f"💾 Download {selected_language} Text",
                                data=translated_text,
                                file_name=f"translated_results_{selected_language}.txt",
                                mime="text/plain"
                            )
                            
                            # Use translated text for TTS
                            text_to_process = translated_text
                
                # Generate audio
                if st.button("Generate Audio"):
                    with st.spinner("Generating audio..."):
                        audio_data = generate_audio(
                            text_to_process, 
                            languages[selected_language]
                        )
                        if audio_data:
                            st.audio(audio_data, format='audio/mp3')
                            
                            st.download_button(
                                label="💾 Download Audio",
                                data=audio_data,
                                file_name=f"audio_{selected_language}.mp3",
                                mime="audio/mp3"
                            )
            
            # Chat interface for analysis - now works with any OCR method
            if langchain_available and st.session_state.chain and st.session_state.extracted_text:
                st.markdown("---")
                st.markdown("### 💬 Ask about the Results")
                
                # Display chat history
                if st.session_state.chat_history and st.session_state.chat_history.messages:
                    for message in st.session_state.chat_history.messages:
                        role = "🤖 Assistant" if "AI" in str(type(message)) else "👤 You"
                        st.write(f"**{role}:** {message.content}")
                
                # User input
                user_input = st.text_input("Ask about the extracted text:", key="chat_input")
                
                # Send button
                if st.button("Send", key="send_button"):
                    if user_input:
                        try:
                            # Add context to user input
                            context_input = f"Based on this extracted text:\n{st.session_state.extracted_text}\n\nUser question: {user_input}"
                            
                            # Add user message to history
                            st.session_state.chat_history.add_user_message(user_input)
                            
                            # Get AI response
                            response = st.session_state.chain.invoke({
                                "user_input": context_input,
                                "chat_history": st.session_state.chat_history.messages,
                            })
                            
                            # Add AI response to history
                            st.session_state.chat_history.add_ai_message(response)
                            
                            # Rerun to update chat display
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Chat error: {str(e)}")
                            import traceback
                            st.error(traceback.format_exc())
                
                # Clear button
                if st.button("Clear Chat", key="clear_button"):
                    st.session_state.chat_history = ChatMessageHistory()
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
    
    # Information section
    with st.expander("About this app"):
        st.write("""
        ## Unified OCR Application
        
        This app combines multiple OCR techniques:
        - **Basic OCR**: Using Tesseract for straightforward text extraction
        - **SmolDocling OCR**: Using advanced AI models for more complex document understanding
        
        Additional features:
        - Image analysis with AI
        - Translation to multiple languages
        - Text-to-speech capabilities
        - Interactive chat for analyzing results
        
        ### Requirements
        - Basic: Tesseract OCR engine
        - Advanced: Various AI libraries (transformers, docling-core)
        - API keys for certain features
        """)

if __name__ == "__main__":
    main()