from PIL import Image
import pytesseract
from transformers import SegformerFeatureExtractor,SegformerForImageClassification
import torch

def get_image_text(image_path, language=["en"]):
    """
    Extracts text from an image using OCR.
    """
    try:
        # Open the image file
        image = Image.open(image_path)
        
        # Use pytesseract for basic OCR
        text = pytesseract.image_to_string(image, lang='+'.join(language))
        
        # Split into lines and remove empty lines
        return [line.strip() for line in text.split('\n') if line.strip()]
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return []

def load_segformer():
    """
    Load Segformer model and processor
    """
    processor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    model = SegformerForImageClassification.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    return processor, model

if __name__ == "__main__":
    # Test the function
    image_path = r"C:\AI_PROJECTS\Vitstuff\WhatsApp Image 2025-03-07 at 23.26.04_1e48c753.jpg"
    result = get_image_text(image_path)
    print(result)