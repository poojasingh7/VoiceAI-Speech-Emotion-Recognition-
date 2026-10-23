import os
import sys
import asyncio

backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from database import SessionLocal
from main import analyze_audio, list_demos

async def run_tests():
    db = SessionLocal()
    demos = list_demos()
    print(f"Loaded {len(demos)} demo files.")

    print("\n--- TESTING EMOTION ANALYZE ENDPOINT ---")
    for d in demos:
        dname = d["filename"]
        res = await analyze_audio(
            file=None,
            demo_name=dname,
            user=None,
            db=db
        )
        primary = res["primary_prediction"]["predicted_emotion"]
        conf = res["primary_prediction"]["confidence"]
        ensemble = res["ensemble_prediction"]["predicted_emotion"]
        ens_conf = res["ensemble_prediction"]["confidence"]
        agreement = res["model_agreement"]["status"]
        transcript = res["transcript"]["text"]
        print(f"Demo: {dname:<26} -> Ensemble: {ensemble:<8} ({ens_conf*100:.1f}%) | Primary: {primary:<8} ({conf*100:.1f}%) | {agreement}")
        print(f"      Text: {transcript[:55]}...")
    
    db.close()

if __name__ == "__main__":
    asyncio.run(run_tests())
