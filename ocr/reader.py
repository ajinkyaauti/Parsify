import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_DISABLE_MKLDNN"] = "True"

from paddleocr import PaddleOCR
from PIL import Image

# Create the OCR model (only loads once)
ocr_model = PaddleOCR(lang='en', enable_mkldnn=False)


def read_image(image_path):
    """
    Give this function an image path.
    It returns a list of all text found in the image.
    Each item in the list has: text, position, and confidence.
    """

    raw_results = ocr_model.predict(image_path)

    extracted = []
    # print(raw_results)

    for page in raw_results:
        boxes = page.get("dt_polys") or page.get("dt_boxes") or page.get("rec_boxes", [])
        texts = page.get("rec_texts", [])
        scores = page.get("rec_scores", [])

        for position, text, confidence in zip(boxes, texts, scores):
            extracted.append({
                "text": text,
                "position": position,
                "confidence": round(confidence, 2)
            })

    return extracted


# ---- TEST IT ----
# This block only runs when you run this file directly
if __name__ == "__main__":

    # Change this to your sample invoice image path
    test_image = "samples/iv.png"

    print("Running OCR on:", test_image)
    results = read_image(test_image)

    print("\n--- Results ---")
    for item in results:
        print(f"Text: {item['text']}  |  Confidence: {item['confidence']}")
