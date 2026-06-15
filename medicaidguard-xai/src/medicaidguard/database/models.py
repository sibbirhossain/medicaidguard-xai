"""SQLAlchemy schema. SQLite for the local demo, PostgreSQL-compatible by design.

Type choices avoid SQLite-only constructs: String rather than TEXT-with-affinity,
Numeric for money, explicit indexes rather than relying on implicit ones. The
same metadata creates the schema on either backend.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Provider(Base):
    __tablename__ = "providers"
    provider_id = Column(String(24), primary_key=True)
    provider_type = Column(String(32))
    specialty = Column(String(48))
    enrollment_date = Column(Date)
    service_region = Column(String(16), index=True)
    active_status = Column(Boolean, default=True)
    historical_claim_volume = Column(Integer)
    caregivers = relationship("Caregiver", back_populates="provider")


class Caregiver(Base):
    __tablename__ = "caregivers"
    caregiver_id = Column(String(24), primary_key=True)
    provider_id = Column(String(24), ForeignKey("providers.provider_id"), index=True)
    employment_start_date = Column(Date)
    service_region = Column(String(16), index=True)
    qualification_status = Column(String(24))
    historical_visit_volume = Column(Integer)
    provider = relationship("Provider", back_populates="caregivers")


class Patient(Base):
    __tablename__ = "patients"
    patient_id = Column(String(24), primary_key=True)
    service_region = Column(String(16), index=True)
    demographic_group = Column(String(24))
    age_band = Column(String(16))
    authorized_hours_weekly = Column(Float)
    service_type = Column(String(32))
    eligibility_status = Column(String(24))


class Authorization(Base):
    __tablename__ = "authorizations"
    authorization_id = Column(String(24), primary_key=True)
    patient_id = Column(String(24), ForeignKey("patients.patient_id"), index=True)
    authorized_service_code = Column(String(16))
    authorized_start_date = Column(Date)
    authorized_end_date = Column(Date)
    authorized_hours_weekly = Column(Float)
    authorization_status = Column(String(24))


class Claim(Base):
    __tablename__ = "claims"
    claim_id = Column(String(24), primary_key=True)
    provider_id = Column(String(24), ForeignKey("providers.provider_id"), index=True)
    caregiver_id = Column(String(24), ForeignKey("caregivers.caregiver_id"), index=True)
    patient_id = Column(String(24), ForeignKey("patients.patient_id"), index=True)
    authorization_id = Column(String(24), ForeignKey("authorizations.authorization_id"))
    service_date = Column(Date, index=True)
    billing_start_time = Column(DateTime)
    billing_end_time = Column(DateTime)
    billed_hours = Column(Float)
    authorized_hours = Column(Float)
    service_code = Column(String(16), index=True)
    billed_amount = Column(Numeric(12, 2))
    claim_status = Column(String(16))
    submission_timestamp = Column(DateTime)

    # Composite index matching the dominant access pattern: a caregiver's
    # activity within a date window, used by every rolling feature and by the
    # overlap/travel checks.
    __table_args__ = (Index("ix_claims_caregiver_date", "caregiver_id", "service_date"),)


class EVVEvent(Base):
    __tablename__ = "evv_events"
    evv_event_id = Column(String(24), primary_key=True)
    claim_id = Column(String(24), ForeignKey("claims.claim_id"), index=True, unique=True)
    caregiver_id = Column(String(24), index=True)
    patient_id = Column(String(24), index=True)
    check_in_time = Column(DateTime)
    check_out_time = Column(DateTime)
    check_in_latitude = Column(Float)
    check_in_longitude = Column(Float)
    check_out_latitude = Column(Float)
    check_out_longitude = Column(Float)
    device_type = Column(String(24))
    verification_method = Column(String(24))
    location_accuracy_m = Column(Float)
    event_status = Column(String(16))


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    claim_id = Column(String(24), ForeignKey("claims.claim_id"), index=True)
    model_version = Column(String(48), index=True)
    risk_score = Column(Float)
    model_probability = Column(Float)
    review_priority = Column(String(16), index=True)
    scored_at = Column(DateTime)
    __table_args__ = (UniqueConstraint("claim_id", "model_version",
                                       name="uq_prediction_claim_model"),)


class AlertEvidence(Base):
    __tablename__ = "alert_evidence"
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String(32), index=True)
    claim_id = Column(String(24), ForeignKey("claims.claim_id"), index=True)
    rule_evidence = Column(JSON)
    feature_evidence = Column(JSON)
    explanation_limitations = Column(String(512))
    review_status = Column(String(24), default="pending")
    investigator_notes = Column(String(2000))


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_name = Column(String(64), index=True)
    git_commit = Column(String(40))
    seed = Column(Integer)
    settings = Column(JSON)
    results = Column(JSON)
    created_at = Column(DateTime)
