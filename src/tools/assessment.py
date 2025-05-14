# src/tools/assessment.py
"""
Assessment tools for evaluating supply chain situation criticality.
"""

from typing import Dict, List, Optional # Added Optional

def criticality_assessment_tool(
    drug_info: Dict,
    stockout_history: List[Dict],
    unfulfilled_demand: float,
    total_demand: float,
    # New argument:
    regional_patient_impact_score: Optional[float] = None # Patient impact for the region/drug context
) -> Dict:
    """
    Assesses the criticality of a drug supply situation based on stockout history
    and unfulfilled demand, providing recommendations for action. Optionally considers patient impact.
    
    Args:
        drug_info: Information about the drug being assessed
        stockout_history: Record of stockout events
        unfulfilled_demand: Amount of unfulfilled demand
        total_demand: Total demand amount for comparison
        regional_patient_impact_score: Optional. Accumulated patient impact score for the context.
        
    Returns:
        Dict: Assessment results including criticality score, category, and recommendations
    """
    drug_criticality = drug_info.get("criticality_value", 1)
    drug_name = drug_info.get("name", "Unknown Drug") # More descriptive default

    stockout_days = len(set(s['day'] for s in stockout_history if isinstance(s, dict) and 'day' in s))

    if total_demand > 1e-6: # Avoid division by zero
        unfulfilled_percentage = (unfulfilled_demand / total_demand) * 100
    else:
        unfulfilled_percentage = 0 if unfulfilled_demand < 1e-6 else 100

    base_score = drug_criticality * 10
    stockout_penalty = min(30, stockout_days * 5)
    unfulfilled_penalty = min(30, unfulfilled_percentage * 0.6)
    
    impact_penalty = 0.0
    if regional_patient_impact_score is not None and regional_patient_impact_score > 0:
        # Example scaling: Assume impact score relates to unfulfilled units * criticality.
        # A score of 1000 might be significant. Add up to +10 to criticality score.
        # This scaling needs tuning based on typical patient_impact_score magnitudes.
        scaled_impact_for_crit_score = min(regional_patient_impact_score / 100.0, 10.0) 
        impact_penalty = scaled_impact_for_crit_score

    criticality_score = base_score + stockout_penalty + unfulfilled_penalty + impact_penalty # Added impact_penalty
    criticality_score = min(100, max(0, criticality_score))

    # Determine category
    if criticality_score >= 80: category = "Critical Emergency"
    elif criticality_score >= 60: category = "Severe Shortage"
    elif criticality_score >= 40: category = "Moderate Concern"
    elif criticality_score >= 20: category = "Potential Issue"
    else: category = "Normal Operations"

    recommendations = []
    if category == "Critical Emergency":
        recommendations.extend([
            "PRIORITY 1: Request IMMEDIATE emergency allocation/resupply.",
            "Activate strict rationing protocols NOW.",
            "Urgently seek therapeutic alternatives.",
            "Escalate issue to regional/central command."
        ])
    elif category == "Severe Shortage":
        recommendations.extend([
            "Significantly increase order quantities (e.g., 2x-3x normal).",
            "Request expedited delivery of pending orders.",
            "Implement patient prioritization criteria.",
            "Notify regional coordinator of severe shortage."
        ])
    elif category == "Moderate Concern":
        recommendations.extend([
            "Increase safety stock targets.",
            "Place supplementary order (e.g., 1.5x normal).",
            "Review usage patterns for potential optimization.",
            "Monitor inbound shipments closely."
        ])
    elif category == "Potential Issue": # pragma: no cover (harder to hit this exactly in tests without specific setup)
        recommendations.extend([
            "Monitor inventory and demand trends very closely.",
            "Consider a small increase in next order.",
            "Verify accuracy of demand forecast."
        ])
    else: 
        recommendations.append("Maintain standard ordering procedures based on forecast.")

    assessment = {
        "drug_name": drug_name,
        "criticality_score": round(criticality_score, 1),
        "category": category,
        "stockout_days_recent": stockout_days,
        "unfulfilled_percentage_recent": round(unfulfilled_percentage, 1),
        "recommendations": recommendations,
        "regional_patient_impact_score_considered": regional_patient_impact_score if regional_patient_impact_score is not None else "N/A"
    }
    return assessment