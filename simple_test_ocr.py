from mistralai import Mistral, DocumentURLChunk, TextChunk
from pathlib import Path
import json
import os

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

pdf_file = Path("recu_physio.pdf")

uploaded_file = client.files.upload(
    file={
        "file_name": pdf_file.stem,
        "content": pdf_file.read_bytes(),
    },
    purpose="ocr",
)

signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

pdf_response = client.ocr.process(document=DocumentURLChunk(document_url=signed_url.url), model="mistral-ocr-latest",
                                  include_image_base64=False)

pdf_ocr_markdown = pdf_response.pages[0].markdown

chat_response = client.chat.complete(
    model="ministral-8b-latest",
    messages=[
        {
            "role": "user",
            "content": [
                TextChunk(text="This is image's OCR in markdown:\n"
                               f"<BEGIN_IMAGE_OCR>\n{pdf_ocr_markdown}\n<END_IMAGE_OCR>.\n"
                                "Convert this into a sensible structured json response."
                               "Keep privacy, do not extract personnal infos, names, tel numbers, adresses, only items, prices and taxes"
                                "The output should be strictly be json with no extra commentary")
            ],
        },
    ],
    response_format={"type": "json_object"},
    temperature=0
)

response_dict = json.loads(chat_response.choices[0].message.content)
json_string = json.dumps(response_dict, indent=4)
print(json_string)
