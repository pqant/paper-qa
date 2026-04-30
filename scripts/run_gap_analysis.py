"""Run the Phase 2 gap analysis pipeline.

Usage:
    PYTHONPATH=src python3 scripts/run_gap_analysis.py
    PYTHONPATH=src python3 scripts/run_gap_analysis.py --output-dir ./data/gap_analysis_custom/
"""

import argparse
import asyncio
import logging
import sys

sys.path.insert(0, "src")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


async def main(output_dir: str, llm_api_base: str) -> None:
    from paperqa.stores.gap_analysis import run_gap_analysis

    report = await run_gap_analysis(
        qdrant_collection="paperbridge_glm_v2",
        output_dir=output_dir,
        llm_api_base=llm_api_base,
    )

    print(f"\n{'='*60}")
    print(f"Gap Analysis Complete")
    print(f"{'='*60}")
    print(f"Gaps identified: {len(report.gaps)}")
    print(f"Summary: {report.summary}")
    print(f"\nTop gaps:")
    for gap in report.gaps[:10]:
        print(f"  {gap.id}. [{gap.confidence:.2f}] {gap.title}")
        print(f"     {gap.description[:150]}...")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 2 gap analysis")
    parser.add_argument("--output-dir", default="./data/gap_analysis/")
    parser.add_argument("--llm-api-base", default="http://192.168.0.28:8005/v1")
    args = parser.parse_args()
    asyncio.run(main(args.output_dir, args.llm_api_base))
