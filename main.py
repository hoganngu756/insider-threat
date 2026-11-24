import argparse
import sys

from src import pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the insider threat pipeline end to end.")
    parser.add_argument("--data", default="data/mock_data.csv", help="raw SIEM export")
    parser.add_argument("--db", default="data/security.db", help="SQLite database to write")
    parser.add_argument("--processed-dir", default="data/processed", help="where the hand-off CSVs go")
    parser.add_argument("--powerbi-dir", default="data/powerbi", help="where the Power BI CSVs go")
    parser.add_argument("--skip-llm", action="store_true", help="don't call the LLM API")
    parser.add_argument("--placeholders", action="store_true",
                        help="fill missing risk scoring and AI output with marked stand-in rows")
    parser.add_argument("--strict", action="store_true", help="stop on the first failing stage")
    args = parser.parse_args(argv)

    results = pipeline.run(
        data=args.data, db=args.db, processed_dir=args.processed_dir, powerbi_dir=args.powerbi_dir,
        use_llm=not args.skip_llm, use_placeholders=args.placeholders, strict=args.strict,
    )
    print()
    print(pipeline.format_results(results))
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
