import os
import json
import numpy as np

def analyze_customer_support_call(timeline_segments: list, transcript_data: dict, duration: float) -> dict:
    """
    Dedicated workflow for analyzing customer service conversations:
    - Speaker turn segmentation (Agent vs Customer assignment)
    - Speaking time distribution & call dynamics
    - Customer Frustration Index & Escalation Detection
    - Agent Empathy & Composure Adherence
    - Human Review Flags
    - Decision support call summary
    """
    if not timeline_segments:
        return {
            "error": "No timeline segments provided for call analysis."
        }

    turns = []
    customer_dur = 0.0
    agent_dur = 0.0

    customer_frustrations = []
    agent_calmness = []
    uncertain_segments_for_review = []

    # Assign alternating speaker turns or use speaker tags
    for i, seg in enumerate(timeline_segments):
        start = seg.get("start_time", 0.0)
        end = seg.get("end_time", 0.0)
        seg_dur = max(0.1, end - start)
        emotion = seg.get("predicted_emotion", "neutral")
        confidence = seg.get("model_score", 0.8)
        pitch = seg.get("pitch_mean", 160.0)
        energy = seg.get("energy_mean", 0.05)

        # In typical two-speaker customer service, turns alternate or cluster
        # Even indices: Customer, Odd indices: Support Agent
        speaker = "Customer" if i % 2 == 0 else "Support Agent"

        # Matching transcript if available
        matched_text = ""
        if transcript_data and "segments" in transcript_data:
            for t_seg in transcript_data["segments"]:
                if (t_seg["start"] <= end and t_seg["end"] >= start):
                    matched_text += t_seg["text"] + " "
        matched_text = matched_text.strip() or f"Conversation turn {i + 1}"

        # Uncertainty check: confidence < 0.50
        needs_review = confidence < 0.55
        if needs_review:
            uncertain_segments_for_review.append({
                "turn_index": i + 1,
                "speaker": speaker,
                "timestamp": f"{start:.1f}s - {end:.1f}s",
                "predicted_emotion": emotion,
                "confidence": round(confidence, 2),
                "reason": "Borderline acoustic probability distribution. Human auditory verification recommended.",
            })

        # Calculate metrics per speaker
        if speaker == "Customer":
            customer_dur += seg_dur
            frustration_weight = 1.0 if emotion == "angry" else 0.8 if emotion in ["fearful", "disgust"] else 0.6 if emotion == "sad" else 0.1
            customer_frustrations.append(frustration_weight)
        else:
            agent_dur += seg_dur
            calm_weight = 1.0 if emotion in ["calm", "neutral"] else 0.9 if emotion == "happy" else 0.2
            agent_calmness.append(calm_weight)

        turns.append({
            "turn_index": i + 1,
            "speaker": speaker,
            "start_time": start,
            "end_time": end,
            "duration": round(seg_dur, 2),
            "emotion": emotion,
            "confidence": round(confidence, 2),
            "pitch_hz": round(pitch, 1),
            "energy": round(energy, 4),
            "text": matched_text,
            "needs_review": needs_review,
        })

    # Aggregates
    total_active_talk = customer_dur + agent_dur
    customer_talk_pct = round((customer_dur / max(total_active_talk, 1e-5)) * 100, 1)
    agent_talk_pct = round((agent_dur / max(total_active_talk, 1e-5)) * 100, 1)

    customer_frustration_index = round(float(np.mean(customer_frustrations) * 100), 1) if customer_frustrations else 15.0
    agent_empathy_score = round(float(np.mean(agent_calmness) * 100), 1) if agent_calmness else 88.0

    # Escalation risk classification
    if customer_frustration_index > 65.0:
        escalation_risk = "HIGH"
        risk_color = "#ef4444"
        recommendation = "Immediate supervisor follow-up recommended. Customer vocal markers exhibited acute stress or agitation."
    elif customer_frustration_index > 40.0:
        escalation_risk = "MODERATE"
        risk_color = "#f59e0b"
        recommendation = "Standard follow-up survey recommended. Minor friction detected during specific interaction turns."
    else:
        escalation_risk = "LOW"
        risk_color = "#10b981"
        recommendation = "Interaction remained stable and constructive throughout."

    return {
        "call_duration_sec": round(duration, 2),
        "total_turns": len(turns),
        "customer_talk_pct": customer_talk_pct,
        "agent_talk_pct": agent_talk_pct,
        "customer_frustration_index": customer_frustration_index,
        "agent_empathy_score": agent_empathy_score,
        "escalation_risk": escalation_risk,
        "risk_color": risk_color,
        "recommendation": recommendation,
        "turns": turns,
        "review_flags": uncertain_segments_for_review,
        "ethical_disclaimer": (
            "Ethical Use Notice: Analyzed data constitutes automated acoustic decision-support "
            "metrics. They must never be treated as definitive or punitive measures of a customer's attitude "
            "or an employee's professional performance."
        ),
    }
