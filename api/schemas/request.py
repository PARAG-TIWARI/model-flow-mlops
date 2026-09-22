"""Pydantic request schemas for inference validation."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class CustomerFeatures(BaseModel):
    """Customer profile and subscription features for churn risk prediction."""

    gender: Literal["Female", "Male"] = Field(
        ..., description="Customer biological gender", examples=["Female"]
    )
    SeniorCitizen: Literal["0", "1", 0, 1] = Field(
        ..., description="Whether customer is senior citizen (0 or 1)", examples=[0]
    )
    Partner: Literal["Yes", "No"] = Field(
        ..., description="Whether customer has a domestic partner", examples=["Yes"]
    )
    Dependents: Literal["Yes", "No"] = Field(
        ..., description="Whether customer has dependents", examples=["No"]
    )
    tenure: int = Field(
        ..., ge=0, le=120, description="Number of months customer has stayed with company", examples=[24]
    )
    PhoneService: Literal["Yes", "No"] = Field(
        ..., description="Whether customer has a landline phone service", examples=["Yes"]
    )
    MultipleLines: Literal["No phone service", "No", "Yes"] = Field(
        ..., description="Whether customer has multiple telephone lines", examples=["No"]
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Customer's internet service provider type", examples=["Fiber optic"]
    )
    OnlineSecurity: Literal["No internet service", "No", "Yes"] = Field(
        ..., description="Whether customer has online security add-on", examples=["No"]
    )
    OnlineBackup: Literal["No internet service", "No", "Yes"] = Field(
        ..., description="Whether customer has cloud backup add-on", examples=["Yes"]
    )
    DeviceProtection: Literal["No internet service", "No", "Yes"] = Field(
        ..., description="Whether customer has device protection plan", examples=["No"]
    )
    TechSupport: Literal["No internet service", "No", "Yes"] = Field(
        ..., description="Whether customer has dedicated premium tech support", examples=["No"]
    )
    StreamingTV: Literal["No internet service", "No", "Yes"] = Field(
        ..., description="Whether customer uses streaming TV services", examples=["Yes"]
    )
    StreamingMovies: Literal["No internet service", "No", "Yes"] = Field(
        ..., description="Whether customer uses streaming movie services", examples=["Yes"]
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="Billing contract term length", examples=["Month-to-month"]
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ..., description="Whether paperless digital billing is activated", examples=["Yes"]
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(
        ..., description="Customer payment channel", examples=["Electronic check"]
    )
    MonthlyCharges: float = Field(
        ..., ge=0.0, description="Current monthly billed charge amount", examples=[89.85]
    )
    TotalCharges: Optional[float] = Field(
        None, ge=0.0, description="Cumulative total charges billed to customer", examples=[2156.40]
    )

    @model_validator(mode="after")
    def populate_total_charges_if_missing(self) -> "CustomerFeatures":
        """Auto-populate TotalCharges if absent."""
        if self.TotalCharges is None:
            self.TotalCharges = round(self.MonthlyCharges * max(1, self.tenure), 2)
        return self

    def to_dict(self):
        """Convert to flat dictionary matching pipeline expectation."""
        d = self.model_dump()
        d["SeniorCitizen"] = str(d["SeniorCitizen"])
        return d


class BatchInferenceRequest(BaseModel):
    """Batch inference request payload containing multiple customer records."""

    customers: List[CustomerFeatures] = Field(
        ..., min_length=1, max_length=1000, description="List of customer feature records"
    )
