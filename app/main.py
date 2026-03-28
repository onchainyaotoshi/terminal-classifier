import asyncio
from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from app.classifier import TerminalClassifier
from app.config import settings

classifier: TerminalClassifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    classifier = TerminalClassifier()
    yield


app = FastAPI(title="Terminal Classifier", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    classification: str
    confidence: float
    scores: dict[str, float]


def verify_api_key(x_api_key: str = Header()):
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.post("/classify", response_model=ClassifyResponse)
async def classify(
    request: ClassifyRequest,
    x_api_key: str = Header(),
):
    verify_api_key(x_api_key)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, partial(classifier.classify, request.text))
    return result
