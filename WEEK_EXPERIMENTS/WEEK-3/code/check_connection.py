import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from pymongo.errors import ConfigurationError, PyMongoError

from mongo_logger import MongoLogger


def check_connection(write: bool = False) -> None:
    load_dotenv(Path(__file__).with_name(".env"))
    logger = MongoLogger()
    try:
        print("PASS: MongoDB ping successful.")
        if write:
            inserted_id = logger.log_call({
                "record_type": "connectivity_check",
                "status": "success",
                "message": "Week 3 MongoDB insert/read verification",
            })
            if logger.collection.find_one({"_id": inserted_id}) is None:
                raise RuntimeError(
                    f"Inserted record {inserted_id} could not be read back."
                )
            print(f"PASS: Inserted and read back document {inserted_id}.")
            print("Refresh llm_monitoring > call_logs in Compass or Atlas Data Explorer.")
        else:
            print("Next: python check_connection.py --write")
    finally:
        logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify MongoDB without an LLM call.")
    parser.add_argument(
        "--write", action="store_true",
        help="Insert/read a marked connectivity-check document (kept as evidence).",
    )
    args = parser.parse_args()
    try:
        check_connection(write=args.write)
    except (ConfigurationError, PyMongoError):
        print(
            "FAIL: MongoDB connection/write failed. Check the URI, database user "
            "permissions, server status and network/TLS. "
            "For Atlas, also check the IP access list. "
            "Do not share your connection string.",
            file=sys.stderr,
        )
        sys.exit(1)
    except (ValueError, RuntimeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
