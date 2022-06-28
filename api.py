from typing import Optional

from fastapi import FastAPI, File, UploadFile, Response, Cookie
from fastapi.responses import FileResponse
from fastapi.exceptions import HTTPException
from explanation_generator import ExplanationGenerator
from PIL import Image
import io
import os
import string
import random
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "127.0.0.1",
    "http://localhost",
    "http://127.0.0.1",
    "https://127.0.0.1",
    "http://127.0.0.1:8000",
    "https://127.0.0.1:8000",
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
    explanation_generator= ExplanationGenerator()


@app.post("/process_image")
async def ProcessImage(file: UploadFile, question: str, response: Response):
    session = session_string(16)
    response.set_cookie("session", session)

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    answer, encoder_explanations, decoder_explanations = explanation_generator.explain(image, question, session)
    response = {
        "answer": answer,
        "encoder_paths": [],
        "decoder_paths" : [],
        "session": session
    }
    encoder_path = os.path.join(base_dir, session, "encoder")
    os.makedirs(encoder_path, exist_ok=True)
    decoder_path = os.path.join(base_dir, session, "decoder")
    os.makedirs(decoder_path, exist_ok=True)
    for i, img in enumerate(encoder_explanations):
        path = os.path.join(encoder_path, str(i)+".png")
        img.save(path)
        response["encoder_paths"] += [str(i)+".png"]
    for i, img in enumerate(decoder_explanations):
        path = os.path.join(decoder_path, str(i)+".png")
        #img.save(path)
        response["decoder_paths"] += [str(i)+".png"]

    return response


@app.get("/response/{enc_or_dec}/{idx_token}.png")
async def ResultsImage(enc_or_dec: str, idx_token: int, session: str):
    if not session:
        raise HTTPException(status_code=404, detail="No session cookie found")
    if enc_or_dec not in ["encoder", "decoder"]:
        raise HTTPException(status_code=404, detail="Item not found")
    path = os.path.join(base_dir, session, enc_or_dec, str(idx_token)+".png")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Item not found")
    return FileResponse(path=path)
