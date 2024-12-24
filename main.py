import os
from http import HTTPStatus
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import INPUT_FILE_FORMAT, PORT, UPLOAD_FOLDER, VECTOR_FOLDER
from inference import rag_chain
from ingest import process_uploaded_docs

app = FastAPI(title="d-chat")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
templates = Jinja2Templates(directory="templates")

# Mount static files for serving CSS
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    if not files:
        return {"error": "No files selected"}

    for file in files:
        if file.filename.endswith(INPUT_FILE_FORMAT):
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())
        else:
            raise HTTPException(
                status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                detail="Not a valid file.",
            )
    has_header= True
    has_footer= True
    process_uploaded_docs(file_path, INPUT_FILE_FORMAT, has_header, has_footer)
    return {"message": "Files uploaded successfully!"}


@app.post("/ask")
async def ask_question(question: str, file_name: str):
    try:
        answer = rag_chain(question, file_name)
        return {"question": question, "answer": answer}
    except FileNotFoundError as e:
        print(repr(e))
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, detail="Index file not found"
        ) from e
    except Exception as e:
        print(repr(e))
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e


@app.get("/uploaded-files")
async def list_faiss_files():
    try:
        if not os.path.exists(VECTOR_FOLDER):
            raise HTTPException(status_code=404, detail="Vectors folder not found")

        files = [
            os.path.splitext(f)[0]
            for f in os.listdir(VECTOR_FOLDER)
            if f.endswith(".faiss")
        ]
        return {"files": files}
    except Exception as e:
        print(repr(e))
        raise HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
