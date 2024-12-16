from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from typing import List
import os

app = FastAPI(title="d-chat")

# Set up upload folder and templates
UPLOAD_FOLDER = 'pdf_uploads'
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
        if file.filename.endswith('.pdf'):
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())
        else:
            return {"error": f"File {file.filename} is not a PDF"}

    return {"message": "Files uploaded successfully!"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
