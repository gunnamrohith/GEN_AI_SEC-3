import os
from datetime import datetime, timezone

import certifi
from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import PyMongoError


class MongoLogger:
    def __init__(self) -> None:
        uri = os.getenv("MONGODB_URI", "").strip()
        if not uri.startswith(("mongodb://", "mongodb+srv://")):
            raise ValueError(
                "Set MONGODB_URI to a mongodb:// or mongodb+srv:// URI."
            )
        if "YOUR_" in uri or "<" in uri or ">" in uri:
            raise ValueError("Replace all placeholders in MONGODB_URI.")
        database = os.getenv("MONGODB_DATABASE", "llm_monitoring").strip()
        collection = os.getenv("MONGODB_COLLECTION", "call_logs").strip()
        if not database or not collection:
            raise ValueError("MongoDB database and collection must not be empty.")

        tls_options = {"tlsCAFile": certifi.where()} if uri.startswith("mongodb+srv://") else {}
        self.client: MongoClient = MongoClient(
            uri,
            **tls_options,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
            socketTimeoutMS=20000,
            appname="week3-llm-monitoring",
        )
        try:
            self.client.admin.command("ping")
        except PyMongoError:
            self.client.close()
            raise
        self.db = self.client[database]
        self.collection = self.db[collection]

    def log_call(self, log_data: dict[str, object]) -> ObjectId:
        document = {**log_data, "timestamp": datetime.now(timezone.utc)}
        return self.collection.insert_one(document).inserted_id

    def close(self) -> None:
        self.client.close()
