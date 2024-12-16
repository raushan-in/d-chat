import os
from typing import List

from fastapi import FastAPI, File, UploadFile
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import INPUT_FILE_FORMAT, UPLOAD_FOLDER
from operations import process_uploaded_docs

app = FastAPI(title="d-chat")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
templates = Jinja2Templates(directory="templates")

# Mount static files for serving CSS
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def landing_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


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
            return {"error": f"File {file.filename} is not a {INPUT_FILE_FORMAT}"}

    process_uploaded_docs(file_path, INPUT_FILE_FORMAT)
    return {"message": "Files uploaded successfully!"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
