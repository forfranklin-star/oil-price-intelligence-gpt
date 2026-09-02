from src.pipeline import run_pipeline

try:
    report = run_pipeline()
    print("REPORT_OK")
    print(report["report_date"])
    print(report["latest"])
except Exception as exc:
    print("REPORT_FAILED")
    print(type(exc).__name__)
    print(str(exc))
    raise
