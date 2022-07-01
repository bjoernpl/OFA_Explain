import io
import os
import random
import string

from PIL import Image
from fastapi import FastAPI, UploadFile, Form
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from explanation_generator import ExplanationGenerator

app = FastAPI()

origins = [
    "127.0.0.1",
    "http://localhost:3000",
    "http://127.0.0.1",
    "https://127.0.0.1",
    "http://127.0.0.1:8000",
    "https://127.0.0.1:8000",
    "https://explainable-ofa.ml"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def session_string(str_size):
    chars = string.ascii_letters
    return ''.join(random.choice(chars) for x in range(str_size))


base_dir = os.path.join(os.getcwd(), "results")


@app.on_event('startup')
def init_data():
    global explanation_generator
    explanation_generator = ExplanationGenerator()


@app.post("/process_image")
async def ProcessImage(file: UploadFile, question: str = Form()):
    request_code = session_string(16)

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    encoder_path = os.path.join(base_dir, request_code, "encoder")
    os.makedirs(encoder_path, exist_ok=True)
    decoder_path = os.path.join(base_dir, request_code, "decoder")
    os.makedirs(decoder_path, exist_ok=True)

    answer, encoder_indices, decoder_indices = explanation_generator.explain(
        image, question, encoder_path, decoder_path
    )
    response = {
        "answer": answer,
        "encoder_indices": encoder_indices,
        "decoder_indices": decoder_indices,
        "request_code": request_code
    }

    return response


@app.get("/response/{enc_or_dec}/{idx_token}")
async def ResultsImage(enc_or_dec: str, idx_token: int, request_code: str):
    if not request_code:
        raise HTTPException(status_code=404, detail="No request code")
    if enc_or_dec not in ["encoder", "decoder"]:
        raise HTTPException(status_code=404, detail="Item not found")
    path = os.path.join(base_dir, request_code, enc_or_dec, str(idx_token) + ".jpg")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Item not found")
    return FileResponse(path=path)
