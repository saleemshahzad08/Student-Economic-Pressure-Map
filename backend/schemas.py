from pydantic import BaseModel, Field
from typing import Literal

# 1. Input Model: Strictly validates incoming student profile parameters

class StudentProfileRequest(BaseModel):
    internet_cost: Literal["Not at all", "A little", "Moderately", "A lot"] = Field(
        ..., description="Interet Cost Burden")

    device_access: Literal["Not at all", "A little", "Moderately", "A lot"] = Field(
        ..., description="Device accessibility strain")

    work_reason: Literal['Education Cost', 'Family Support', 'Future savings / Other', 'Personal expenses', 'Not Applicable'] = Field(
        ..., description="Reason for working (if any)")

    education_improvement: Literal['Better career guidance', 'Better learning resources', 'Greater financial stability', 'More flexible study arrangements'] = Field(
        ..., description="Institutional improvement area")

# 2. Sub-Model: Validates sensory confidence scores bounded strictly between 0.0 and 1.0

class ProbabilityDistribution(BaseModel):
    Low: float = Field(..., ge=0.0, le=1.0)
    Moderate: float = Field(..., ge=0.0, le=1.0)
    High: float = Field(..., ge=0.0, le=1.0)

# 3. Output Model: Packages the primary decision, score distribution, and audit flags

class PredictionResponse(BaseModel):
    predicted_category: Literal['Low', 'Moderate', 'High']
    probabilities: ProbabilityDistribution
    rule_trace: str
    is_uncertain: bool