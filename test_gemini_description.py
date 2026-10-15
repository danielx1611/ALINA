import os
from pathlib import Path

import cv2
from google import genai
from google.genai import types

# Load the same image used in our experiments
image_path = Path("data/Raw_Data/vidd_1/00001.jpg")
img = cv2.imread(str(image_path))

if img is None:
    raise FileNotFoundError(image_path)

# Resize exactly as ALINA does for manual selection
height, width = img.shape[:2]
scale = min(1.0, 1080 / max(width, height))

resized = cv2.resize(
    img, None, fx=scale, fy=scale
)

# Encode the image for Gemini
success, encoded = cv2.imencode(".jpg", resized)

if not success:
    raise RuntimeError("Failed to encode image")

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

prompt = """
You are analyzing an image captured from the cockpit
of an aircraft at an airport.

Examine the image carefully and identify the visible
YELLOW PAINTED TAXIWAY LINE MARKINGS.

Answer the following questions:

1. Where are the yellow taxiway markings located?
   Are they primarily in the upper, middle, or lower
   portion of the image?

2. Describe the position of the markings relative
   to the aircraft nose and propeller.

3. Describe the direction in which the yellow markings
   extend across the image.

4. Are there multiple yellow markings visible?
   If so, describe their approximate locations.

5. Which marking would be the best candidate for
   selecting a region of interest for automated
   taxiway line detection, and why?

IMPORTANT:
- Do not return pixel coordinates.
- Do not construct a bounding box or trapezoid.
- Do not assume markings are at the bottom.
- Base your answers only on what is actually
  visible in the image.
"""

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[
        types.Part.from_bytes(
            data=encoded.tobytes(),
            mime_type="image/jpeg"
        ),
        prompt
    ],
    config=types.GenerateContentConfig(
        temperature=0
    )
)

print("\nGEMINI IMAGE DESCRIPTION\n")
print(response.text)