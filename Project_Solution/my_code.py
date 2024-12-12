import os
import re
import requests
import easyocr
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Tuple
from src.utils import download_images, common_mistake, parse_string
import src.constants
from src.sanity import sanity_check

# Initialize the EasyOCR reader for English
reader = easyocr.Reader(['en'])

def download_image(image_link: str) -> Image.Image:
    try:
        response = requests.get(image_link, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img.verify()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error downloading image from {image_link}: {e}")
        return None

def extract_text_from_image(image: Image.Image) -> str:
    try:
        image_bytes = BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)
        result = reader.readtext(image_bytes.read(), detail=0)
        return " ".join(result)
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""

def standardize_unit(unit: str) -> str:
    unit = unit.lower()
    mapping = {
        'g': 'gram', 'gram': 'gram', 'grams': 'gram',
        'kg': 'kilogram', 'kilogram': 'kilogram', 'kilograms': 'kilogram', 'kgs': 'kilogram',
        'lbs': 'pound', 'lb': 'pound', 'pound': 'pound', 'pounds': 'pound',
        'cm': 'centimetre', 'centimeter': 'centimetre', 'centimetre': 'centimetre', 'cms': 'centimetre',
        'mm': 'millimetre', 'millimeter': 'millimetre', 'millimetre': 'millimetre',
        'm': 'metre', 'meter': 'metre', 'metre': 'metre',
        'ft': 'foot', 'feet': 'foot', 'foot': 'foot',
        'v': 'volt', 'volt': 'volt', 'volts': 'volt',
        'w': 'watt', 'watts': 'watt', 'watt': 'watt',
        'oz': 'ounce', 'ounce': 'ounce',
        'yard': 'yard', 'yards': 'yard'
    }
    return mapping.get(unit, unit)

def parse_value_from_text(text: str, entity_name: str) -> List[Tuple[float, str]]:
    patterns = {
        'item_weight': [
            r'net\s*wt\.?\s*(\d+\.?\d*)\s*(kg|g|lbs?|oz|gram|pound|kilogram|ton)',
            r'total\s*weight:?\s*(\d+\.?\d*)\s*(kg|g|lbs?|oz|gram|pound|kilogram|ton)',
            r'package\s*weight:?\s*(\d+\.?\d*)\s*(kg|g|lbs?|oz|gram|pound|kilogram|ton)',
            r'(\d+\.?\d*)\s*(kg|g|lbs?|oz|gram|pound|kilogram|ton)\b'
        ],
        'maximum_weight_recommendation': [
            r'max\s*weight:?\.?\s*(\d+\.?\d*)\s*(kg|g|lbs?|oz|gram|pound|kilogram|ton)',
            r'weight\s*capacity:?\.?\s*(\d+\.?\d*)\s*(kg|g|lbs?|oz|gram|pound|kilogram|ton)',
            r'recommended\s*weight:?\.?\s*(\d+\.?\d*)\s*(kg|g|lbs?|oz|gram|pound|kilogram|ton)',
            r'(\d+\.?\d*)\s*(kg|g|lbs?|oz|gram|pound|kilogram|ton)\b'
        ],
        'voltage': [
            r'(\d+\.?\d*)\s*(v|volts?)\b',
            r'voltage:?\.?\s*(\d+\.?\d*)\s*(v|volts?)\b'
        ],
        'wattage': [
            r'(\d+\.?\d*)\s*(w|watts?)\b',
            r'wattage:?\.?\s*(\d+\.?\d*)\s*(w|watts?)\b'
        ],
        'width': [
            r'width:?\.?\s*(\d+\.?\d*)\s*(cm|mm|ft|inch|metre|yard)\b',
            r'(\d+\.?\d*)\s*(cm|mm|ft|inch|metre|yard)\s*width'
        ],
        'depth': [
            r'depth:?\.?\s*(\d+\.?\d*)\s*(cm|mm|ft|inch|metre|yard)\b',
            r'(\d+\.?\d*)\s*(cm|mm|ft|inch|metre|yard)\s*depth'
        ],
        'height': [
            r'height:?\.?\s*(\d+\.?\d*)\s*(cm|mm|ft|inch|metre|yard)\b',
            r'(\d+\.?\d*)\s*(cm|mm|ft|inch|metre|yard)\s*height'
        ],
        'item_volume': [
            r'(\d+\.?\d*)\s*(ml|l|litre|gallon|pint|quart|fluid ounce|cup)\b',
            r'volume:?\.?\s*(\d+\.?\d*)\s*(ml|l|litre|gallon|pint|quart|fluid ounce|cup)\b'
        ]
    }
    
    results = []
    for pattern in patterns.get(entity_name, []):
        matches = re.findall(pattern, text.lower())
        for match in matches:
            value, unit = match
            results.append((float(value), standardize_unit(unit)))
    
    return results

def select_best_value(values: List[Tuple[float, str]], entity_name: str) -> str:
    if not values:
        return ""
    
    values.sort(key=lambda x: x[0], reverse=True)
    
    if entity_name == 'item_weight':
        for value, unit in values:
            if unit in src.constants.allowed_units and 0.1 <= value <= 50:
                return f"{value} {unit}"
    
    return f"{values[0][0]} {values[0][1]}"

def predictor(image_link: str, entity_name: str) -> str:
    image = download_image(image_link)
    if image is None:
        return ""
    
    width, height = image.size
    segments = [
        image.crop((0, 0, width, height/3)),
        image.crop((0, height/3, width, 2*height/3)),
        image.crop((0, 2*height/3, width, height))
    ]
    
    all_values = []
    for segment in segments:
        extracted_text = extract_text_from_image(segment)
        values = parse_value_from_text(extracted_text, entity_name)
        all_values.extend(values)
    
    return select_best_value(all_values, entity_name)

def calculate_metrics(y_true: List[str], y_pred: List[str]) -> Tuple[float, float, float]:
    precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    return precision, recall, f1

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv(r'dataset/sample_test.csv', dtype=str)
    
    # Predict values for all records
    df['prediction'] = df.apply(
        lambda row: predictor(row['image_link'], row['entity_name']), axis=1)
    
    # Save the predictions in the required format
    output_filename = r'dataset\sample_test_out_predicted.csv'
    df[['index', 'prediction']].to_csv(output_filename, index=False)
    
    # Run sanity check
    try:
        sanity_check(r'dataset\sample_test.csv', output_filename)
    except Exception as e:
        print(f"Sanity check failed: {e}")
    
    # Calculate evaluation metrics
    y_true = df['entity_value'].tolist()
    y_pred = df['prediction'].tolist()
    
    precision, recall, f1 = calculate_metrics(y_true, y_pred)
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
