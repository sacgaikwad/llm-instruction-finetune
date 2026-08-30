from fastapi import FastAPI
from src.api.classifier_service import ClassifierService
from src.api.schemas import (
    ClassificationRequest,
    ClassificationResponse,
)

# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="GPT Spam Classifier API",
    description="FastAPI interface for the GPT-style SCAM/LEGIT classifier.",
    version="1.0.0",
)


# ============================================================
# CLASSIFIER
# ============================================================

classifier = ClassifierService()


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model": "classification_controlled_v2",
    }


# ============================================================
# CLASSIFICATION
# ============================================================

@app.post(
    "/classify",
    response_model=ClassificationResponse,
)
def classify(
    request: ClassificationRequest,
):

    result = classifier.classify(
        request.message
    )

    return result