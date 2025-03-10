import streamlit as st
from PIL import Image
import pytesseract
from transformers import SegformerImageProcessor, SegformerForImageClassification
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from gtts import gTTS
from googletrans import Translator
from IPython.display import Audio
from io import BytesIO
import os
import base64
import traceback

# Load environment variables
load_dotenv()

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

def initialize_chat():
    """Initialize chat components"""
    api_key=os.getenv("GOOGLE_API_KEY")
    #api calls are made using rest api instead of grpc
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",google_api_key=api_key,transport="rest")
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant that analyzes OCR results and helps users understand the extracted text."
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{user_input}"),
    ])
    return prompt | llm | StrOutputParser()

def verify_tesseract():
    """Verify Tesseract installation and configuration"""
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        st.error("""
        ### Tesseract OCR Required
        1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
        2. Install to default location
        3. Add to system PATH
        4. Restart the application
        """)
        return False

def process_image(image, method="basic"):
    """Process image using selected OCR method"""
    if not verify_tesseract():
        return None
        
    if method == "basic":
        with st.spinner("Extracting text..."):
            text = pytesseract.image_to_string(image, lang='en')
            return [line.strip() for line in text.split('\n') if line.strip()]
    else:
        with st.spinner("Loading Segformer model..."):
            processor, model = load_segformer()
            
        with st.spinner("Processing image..."):
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            text = pytesseract.image_to_string(image, lang='en')
            return [line.strip() for line in text.split('\n') if line.strip()]

def load_segformer():
    """Load Segformer model and processor"""
    try:
        processor = SegformerFeatureExtractor.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        model = SegformerForImageClassification.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        return processor, model
    except Exception as e:
        st.error(f"Failed to load Segformer model: {str(e)}")
        return None, None

def save_temp_file(uploaded_file):
    """Save uploaded file to temporary location"""
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / "temp_image.jpg"
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_path

def cleanup_temp_files():
    """Clean up temporary files and directory"""
    temp_dir = Path("temp")
    if temp_dir.exists():
        for file in temp_dir.glob("*"):
            file.unlink()
        temp_dir.rmdir()

def analyze_image_content(image, question="What is shown in this image?", target_language="English"):
    """Analyze image content using Gemini Pro Vision with improved error handling"""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("Google API key is missing. Please check your .env file.")
            return None
            
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            transport="rest",
            temperature=0.2,  # Lower temperature for more reliable results
            max_output_tokens=800  # Limit output to avoid TTS issues
        )
        
        # Convert and resize image
        max_size = (1024, 1024)
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
        
        st.info("Sending request to Gemini...")
        response = model.invoke(input=[message])
        
        # Clean up response
        import re
        cleaned_text = re.sub(r'\n+', '\n', response.content)
        cleaned_text = re.sub(r'\s+\.', '.', cleaned_text)
        
        return cleaned_text
        
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        st.error(traceback.format_exc())
        return None

def get_language_menu():
    """Return dictionary of supported languages"""
    return {
        'Bengali': 'bn',
        'Gujarati': 'gu',
        'Hindi': 'hi',
        'Kannada': 'kn',
        'Malayalam': 'ml',
        'Marathi': 'mr',
        'Nepali': 'ne',
        'Punjabi': 'pa',
        'Telugu': 'te',
        'Urdu': 'ur',
        'Arabic': 'ar',
        'English': 'en',
        'Tamil': 'ta',
        'French': 'fr',
        'German': 'de',
        'Italian': 'it',
        'Japanese': 'ja',
        'Korean': 'ko',
        'Russian': 'ru',
        'Spanish': 'es',
        'Chinese': 'zh-CN'
    }

def translate_text(text, target_language_code):
    """Translate text with improved error handling"""
    if not text or text.strip() == "" or target_language_code == 'en':
        return text
        
    try:
        import time
        time.sleep(0.5)
        
        translator = Translator()
        st.session_state.translation_attempts = st.session_state.get('translation_attempts', 0) + 1
        
        chunks = [text[i:i+300] for i in range(0, len(text), 300)]
        translated_chunks = []
        
        for i, chunk in enumerate(chunks):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        time.sleep(2)
                    
                    if len(chunks) > 1:
                        st.info(f"Translating chunk {i+1}/{len(chunks)} (attempt {attempt+1})")
                    
                    translation = translator.translate(
                        text=chunk,
                        dest=target_language_code,
                        src='en'
                    )
                    
                    if translation and translation.text:
                        translated_chunks.append(translation.text)
                        break
                    else:
                        if attempt == max_retries - 1:
                            st.warning(f"Failed to translate chunk after {max_retries} attempts")
                        time.sleep(1)
                except Exception as chunk_error:
                    if attempt == max_retries - 1:
                        st.warning(f"Translation error on attempt {attempt+1}: {str(chunk_error)}")
                    time.sleep(1.5)
        
        return ' '.join(translated_chunks) if translated_chunks else text
            
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text

def main():
    st.set_page_config(page_title="OCR Tool", page_icon="📝", layout="wide")
    
    st.title("📝 Advanced OCR Application")
    st.sidebar.title("⚙️ Settings")
    
    # Initialize chat chain if not in session state
    if 'chain' not in st.session_state:
        chain = initialize_chat()
        if chain is None:
            st.warning("Chat functionality disabled - check API key configuration")
        else:
            st.session_state.chain = chain
    
    ocr_method = st.sidebar.radio(
        "Choose OCR Method",
        ["Basic OCR", "Segformer + OCR"]
    )
    
    # Add camera input option
    st.sidebar.markdown("---")
    use_camera = st.sidebar.checkbox("Use Camera Input")
    
    if use_camera:
        camera_image = st.camera_input("Take a picture")
        if camera_image is not None:
            image = Image.open(camera_image)
            col1, col2 = st.columns(2)
            
            with col1:
                # Changed use_column_width to use_container_width
                st.image(image, caption="Camera Image", use_container_width=True)
            
            with col2:
                question = st.text_input("Ask about the image:", "What is shown in this image?")
                languages = get_language_menu()
                selected_language = st.selectbox(
                    "Select Language",
                    options=list(languages.keys()),
                    index=list(languages.keys()).index('English')
                )
                
                if st.button("Analyze Image"):
                    with st.spinner(f"Analyzing image..."):
                        analysis = analyze_image_content(image, question, "English")
                        if analysis:
                            st.markdown("### 🔍 Original Analysis (English):")
                            st.write(analysis)
                            
                            tts_text = analysis
                            
                            if selected_language != "English":
                                with st.spinner(f"Translating to {selected_language}..."):
                                    translated_analysis = translate_text(analysis, languages[selected_language])
                                    if translated_analysis:
                                        st.markdown(f"### 🌐 Analysis in {selected_language}:")
                                        st.write(translated_analysis)
                                        tts_text = translated_analysis
                            
                            st.markdown("### 🔊 Text-to-Speech")
                            
                            if 'analysis_audio' not in st.session_state:
                                st.session_state.analysis_audio = {}
                            
                            audio_key = f"analysis_{selected_language}"
                            
                            try:
                                with st.spinner("Generating audio..."):
                                    audio_bytes_io = BytesIO()
                                    tts = gTTS(
                                        text=tts_text,
                                        lang=languages[selected_language],
                                        slow=False
                                    )
                                    tts.write_to_fp(audio_bytes_io)
                                    audio_bytes_io.seek(0)
                                    audio_data = audio_bytes_io.getvalue()
                                    
                                    st.session_state.analysis_audio[audio_key] = audio_data
                                    st.audio(audio_data, format='audio/mp3')
                                    
                                    st.download_button(
                                        label="💾 Download Audio",
                                        data=audio_data,
                                        file_name=f"analysis_{selected_language}.mp3",
                                        mime="audio/mp3",
                                        key=f"download_{audio_key}"
                                    )
                            except Exception as e:
                                st.error(f"Audio generation error: {str(e)}")
                                st.error(f"Language code used: {languages[selected_language]}")
                                st.error(f"Text length: {len(tts_text)} characters")
                                st.info("Troubleshooting: Try with a shorter text or a different language")
    
    # Continue with existing file upload logic
    uploaded_file = st.file_uploader(
        "Or choose an image...", 
        type=["jpg", "jpeg", "png", "bmp"]
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                temp_path = save_temp_file(uploaded_file)
                result = process_image(
                    image, 
                    "basic" if ocr_method == "Basic OCR" else "segformer"
                )
                
                if result:
                    st.markdown("### 📄 Extracted Text:")
                    for line in result:
                        st.write(line)
                    
                    text_content = "\n".join(result)
                    
                    # Add text-to-speech option
                    st.markdown("---")
                    st.markdown("### 🔊 Text-to-Speech")
                    enable_tts = st.checkbox("Enable Text-to-Speech")
                    
                    if enable_tts:
                        languages = get_language_menu()
                        selected_language = st.selectbox(
                            "Select Language",
                            options=list(languages.keys()),
                            index=list(languages.keys()).index('English')
                        )
                        
                        if st.button("Generate Speech"):
                            # Replace the translation section in both camera input and file upload sections with this updated version:
                            if selected_language == 'English':
                                translated_text = text_content  # or text_content for file upload section
                            else:
                                with st.spinner("Translating text..."):
                                    try:
                                        translator = Translator()
                                        # Add delay between translation attempts
                                        import time
                                        
                                        # Split text into smaller chunks
                                        chunks = [text_content[i:i+1000] for i in range(0, len(text_content), 1000)]
                                        translated_chunks = []
                                        
                                        for chunk in chunks:
                                            try:
                                                time.sleep(1)  # Add delay to avoid rate limiting
                                                translation = translator.translate(
                                                    text=chunk,
                                                    dest=languages[selected_language],
                                                    src='en'
                                                )
                                                if translation and translation.text:
                                                    translated_chunks.append(translation.text)
                                                else:
                                                    st.error(f"Translation failed for chunk: {chunk[:100]}...")
                                                    return
                                            except Exception as chunk_error:
                                                st.error(f"Chunk translation error: {str(chunk_error)}")
                                                return
                                        
                                        if translated_chunks:
                                            translated_text = ' '.join(translated_chunks)
                                            st.success("Translation completed!")
                                        else:
                                            st.error("Translation failed - no text was translated")
                                            return
                                            
                                    except Exception as trans_error:
                                        st.error(f"Translation error: {str(trans_error)}")
                                        return

                            # Then continue with the audio generation part:
                            if 'translated_text' in locals():
                                try:
                                    with st.spinner("Generating audio..."):
                                        tts = gTTS(text=translated_text, lang=languages[selected_language])
                                        audio_bytes_io = BytesIO()
                                        tts.write_to_fp(audio_bytes_io)
                                        audio_bytes = audio_bytes_io.getvalue()
                                        
                                        st.audio(audio_bytes, format="audio/mp3")
                                        st.download_button(
                                            label="💾 Download Audio",
                                            data=audio_bytes,
                                            file_name=f"translated_audio_{selected_language}.mp3",
                                            mime="audio/mp3",
                                            key=f"download_{selected_language}"  # Unique key for each language
                                        )
                                except Exception as e:
                                    st.error(f"Audio generation error: {str(e)}")
                    
                    st.download_button(
                        label="💾 Download Results",
                        data=text_content,
                        file_name="extracted_text.txt",
                        mime="text/plain"
                    )
                    
                    # Replace the chat history display section with this updated version:
                    # Add chat interface for analysis
                    st.markdown("---")
                    st.markdown("### 💬 Ask about the Results")

                    # Add language selection for chat translation
                    enable_chat_translation = st.checkbox("Enable Chat Translation")
                    if enable_chat_translation:
                        chat_language = st.selectbox(
                            "Select Chat Language",
                            options=list(languages.keys()),
                            index=list(languages.keys()).index('English'),
                            key="chat_translation_language"
                        )

                    # Display chat history with translation
                    if st.session_state.chat_history.messages:
                        for idx, message in enumerate(st.session_state.chat_history.messages):
                            role = "🤖 Assistant" if "AI" in str(type(message)) else "👤 You"
                            content = message.content
                            
                            # Only translate if enabled and not English
                            if enable_chat_translation and chat_language != 'English':
                                try:
                                    translated_content = translate_text(content, languages[chat_language])
                                    if translated_content:
                                        content = translated_content
                                except Exception as e:
                                    st.error(f"Translation error: {str(e)}")
                            
                            # Display message with TTS option
                            st.write(f"**{role}:** {content}")
                            
                            # Add TTS button for each message
                            col1, col2 = st.columns([6, 1])
                            with col2:
                                if st.button("🔊", key=f"tts_{idx}"):
                                    try:
                                        with st.spinner("Generating audio..."):
                                            audio_bytes_io = BytesIO()
                                            tts = gTTS(
                                                text=content,
                                                lang=languages[chat_language] if enable_chat_translation else 'en',
                                                slow=False
                                            )
                                            tts.write_to_fp(audio_bytes_io)
                                            audio_bytes_io.seek(0)
                                            st.audio(audio_bytes_io.getvalue(), format='audio/mp3')
                                    except Exception as e:
                                        st.error(f"Audio generation error: {str(e)}")

                    # User input section
                    user_input = st.text_input("Ask about the extracted text:", key="chat_input")

                    # Send and Clear buttons
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if st.button("Send", key="send_button"):
                            if user_input:
                                try:
                                    # Add context to user input
                                    context_input = f"Based on this extracted text:\n{text_content}\n\nUser question: {user_input}"
                                    
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

                    with col2:
                        if st.button("Clear Chat", key="clear_button"):
                            st.session_state.chat_history = ChatMessageHistory()
                            st.rerun()
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
        finally:
            cleanup_temp_files()
            
if __name__ == "__main__":
    main()
