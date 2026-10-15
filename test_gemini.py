import os
from pathlib import Path

from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

image_path = Path("data/Raw_Data/vidd_1/00001.jpg")
image_bytes = image_path.read_bytes()

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg"
        ),
        """This is an image taken from an aircraft taxiing
        on an airport taxiway.

        Identify the yellow taxiway markings.

        Return four pixel coordinates defining a trapezoidal
        region of interest around the visible taxiway markings.

        The coordinates must be ordered:
        bottom-left, top-left, top-right, bottom-right.

        Return only JSON in this format:
        {
            "points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        }
        """
    ],
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0
    )
)

print(response.text)