from pydantic import BaseModel, Field


class ClassificationRequest(BaseModel):

    message: str = Field(
        ...,
        min_length=1,
        description="Message to classify"
    )


class ClassificationResponse(BaseModel):

    prediction: str

    scam_probability: float

    legit_probability: float

    confidence: float

    confidence_level: str