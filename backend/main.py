import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from schemas import StudentProfileRequest, PredictionResponse, ProbabilityDistribution

# 1. Importing Model
try:
    pipeline = joblib.load("student_economic_pressure_pipeline.joblib")
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
except Exception as e:
    raise RuntimeError(
        f"Critical: Model artifact could not be loaded: {e}"
    ) from e

# 2. Defining Integer-to-String Mapping for Label
LABEL_MAP = {0: "Low", 1: "Moderate", 2: "High"}

# 3. Defining Leaf Rules for Transparent Decision Making
LEAF_RULES = {
    4: "Low internet cost burden (<= A little) with minimal device access strain (<= Moderately). Associated with Low Economic Pressure cohort (86.0%).",
    5: "Low internet cost burden (<= A little) with emerging device access strain (>= Moderately). Associated with Low Economic Pressure cohort (78.0%).",
    7: "Moderate internet cost burden with low device access strain (<= A little). Dominant alignment with Low Economic Pressure (69.0%).",
    8: "Moderate internet cost burden with moderate-to-high device access strain (> A little). Baseline alignment with Low Economic Pressure (52.0%).",
    10: "Low-to-moderate internet cost burden with low device access strain, accompanied by prioritization of institutional financial stability. Leans Low (52.0%).",
    11: "Low-to-moderate internet cost burden with elevated device access strain, accompanied by prioritization of institutional financial stability. Leans Moderate (50.0%).",
    15: "High internet cost burden (>= Moderately) with low-to-moderate device access strain (<= Moderately). Associated with Moderate Economic Pressure (48.0%).",
    16: "High internet cost burden (>= Moderately) with elevated device access strain (> Moderately), where primary work motivation is personal expense management. Leans Moderate (45.0%).",
    17: "High internet cost burden (>= Moderately) with severe device access strain (A lot), with no prioritization of institutional financial stability. High Contextual Uncertainty cohort across Low (33%), Moderate (33%), and High (35%).",
    20: "High internet cost burden (A lot) and moderate device access strain, with institutional financial stability priorities (non-working cohort). Leans Moderate (45.0%).",
    21: "High internet cost burden (A lot) and moderate device access strain, with institutional financial stability priorities (working cohort). Leans Moderate (52.0%).",
    22: "High internet cost burden (A lot) combined with compound device accessibility strain (A lot) and financial stability needs. Strong alignment with High Economic Pressure (66.1%)."
}

# 4. Instantiating the Main App
app = FastAPI(
    title='Student Economic Pressure Map API',
    description='Stateless decision-support service for student socioeconomic pressure modeling',
    version='1.0.0'
)

# 5. Handling CORS-related Issues
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://localhost:5500", "http://127.0.0.1:5500", "https://saleemshahzad08.github.io"],
    allow_credentials = True,
    allow_methods = ["POST", "GET", "OPTIONS"],
    allow_headers = ["*"]
)

# 5. Adding a Lightweight Health Checkup Endpoint
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_checkup():
    return {
        "status": "healthy", 
        "service": "student-economic-pressure-api"
    }

# 6. Adding Main Prediction Route
@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_economic_pressure(profile:StudentProfileRequest):
    try:
        # 6(A). Converting validated Pydantic model to single-row DataFrame
        input_df = pd.DataFrame([profile.model_dump()])

        # 6(B). Extracting probability distribution
        raw_probs = pipeline.predict_proba(input_df)[0]
        prob_dist = ProbabilityDistribution(
            Low=round(float(raw_probs[0]), 4),
            Moderate=round(float(raw_probs[1]), 4),
            High=round(float(raw_probs[2]), 4)
        )

        # 6(C). Determining majority label
        pred_int = int(np.argmax(raw_probs))
        predicted_category = LABEL_MAP[pred_int]

        # 6(D). Uncertainty check ( top margin <= 0.05 or maximum probability <0.45)
        sorted_probs = sorted(raw_probs, reverse=True)
        margin = round(sorted_probs[0] - sorted_probs[1], 4)
        is_uncertain = bool(margin <= 0.05 or round(sorted_probs[0], 2) < 0.45)

        # 6(E). Extracting tree leaf ID and retrieving plain-language explanation
        input_df_transformed = preprocessor.transform(input_df)
        leaf_id = int(classifier.apply(input_df_transformed)[0])
        rule_trace = LEAF_RULES.get(leaf_id, f"Leaf Node #{leaf_id}: General pattern matching {predicted_category} pressure profile.")

        # 6(F). Returning response payload
        return PredictionResponse(
            predicted_category = predicted_category,
            probabilities = prob_dist,
            rule_trace = rule_trace,
            is_uncertain = is_uncertain
        )
    except Exception as e:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail = f"Inference execution failed:{str(e)}"
        )

@app.get("/")
def welcome():
    return "Welcome to API Testing."