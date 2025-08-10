import json

def load_jobs_from_file(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        jobs_data = json.load(f)
    jobs = []
    for job in jobs_data:
        title = job.get("title", "")
        company = job.get("companyName", "")
        desc = job.get("description", "")
        combined_text = f"{title} at {company}\n{desc}"
        jobs.append({"id": job.get("id", ""), "text": combined_text})
    return jobs
