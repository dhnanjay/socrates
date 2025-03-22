import sqlite3
import os
import time
import uuid
import asyncio
import logging

# Define the threshold at which summarization is triggered
MEMORY_LIMIT = 5


class SQLiteMemory:
    """
    A lightweight memory manager using SQLite.
    Stores messages with the following fields:
      - msg_id (TEXT PRIMARY KEY)
      - user_id (TEXT)
      - role (TEXT): 'user' or 'agent'
      - text (TEXT)
      - timestamp (INTEGER)
    Also supports a summaries table to roll older messages.
    """

    def __init__(self, db_path="memory.db"):
        """
        :param db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        """Create the necessary tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Table for individual messages
        c.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                msg_id TEXT PRIMARY KEY,
                user_id TEXT,
                role TEXT,
                text TEXT,
                timestamp INTEGER
            )
        """)

        # Table for summaries
        c.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                summary_id TEXT PRIMARY KEY,
                user_id TEXT,
                text TEXT,
                timestamp INTEGER
            )
        """)

        conn.commit()
        conn.close()

    def store_message(self, user_id: str, text: str, role: str = "user"):
        """
        Insert a new message into the database. After insertion,
        check if the number of messages has reached MEMORY_LIMIT,
        and if so, trigger summarization.
        """
        msg_id = str(uuid.uuid4())
        timestamp = int(time.time())

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            INSERT INTO messages (msg_id, user_id, role, text, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (msg_id, user_id, role, text, timestamp))

        c.execute("SELECT COUNT(*) FROM messages WHERE user_id = ?", (user_id,))
        count = c.fetchone()[0]
        conn.commit()
        conn.close()

        # Trigger summarization if we have enough messages
        if count >= MEMORY_LIMIT:
            asyncio.create_task(self.manage_summarization(user_id))

    async def manage_summarization(self, user_id: str):
        """
        Summarize older messages for a user:
         - Keep the last (MEMORY_LIMIT - 1) messages.
         - Summarize the older messages (optionally include existing summary).
         - Delete the summarized messages and store the new summary.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            SELECT msg_id, role, text, timestamp
            FROM messages
            WHERE user_id = ?
            ORDER BY timestamp ASC
        """, (user_id,))
        rows = c.fetchall()
        total_messages = len(rows)

        if total_messages < MEMORY_LIMIT:
            conn.close()
            return  # Not enough messages

        # Keep the last (MEMORY_LIMIT - 1) messages; summarize the rest.
        keep_count = MEMORY_LIMIT - 1
        to_summarize = rows[: total_messages - keep_count]

        # Fetch existing summary (if any)
        c.execute("""
            SELECT summary_id, text, timestamp 
            FROM summaries 
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (user_id,))
        existing_summary = c.fetchone()
        existing_summary_text = existing_summary[1] if existing_summary else ""

        # Build text for summarization; include role labels.
        conversation_texts = []
        if existing_summary_text:
            conversation_texts.append(f"SUMMARY: {existing_summary_text}")
        for row in to_summarize:
            _, role, text, _ = row
            conversation_texts.append(f"{role.upper()}: {text}")

        # For demonstration, create a "fake" summary.
        new_summary_text = "FAKE SUMMARY: " + " | ".join(conversation_texts[:3]) + "..."

        new_summary_id = str(uuid.uuid4())
        new_timestamp = int(time.time())

        try:
            c.execute("BEGIN")
            # Remove any old summary for this user
            c.execute("DELETE FROM summaries WHERE user_id = ?", (user_id,))
            # Insert the new summary
            c.execute("""
                INSERT INTO summaries (summary_id, user_id, text, timestamp)
                VALUES (?, ?, ?, ?)
            """, (new_summary_id, user_id, new_summary_text, new_timestamp))
            # Delete the messages that were summarized
            to_summarize_ids = tuple(row[0] for row in to_summarize)
            placeholders = ",".join("?" * len(to_summarize_ids))
            c.execute(f"DELETE FROM messages WHERE msg_id IN ({placeholders})", to_summarize_ids)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logging.error(f"Error during summarization: {e}")
        finally:
            conn.close()

    def retrieve_memory(self, user_id: str) -> str:
        """
        Retrieve the current rolling memory for a user:
        Returns the latest summary (if exists) plus any remaining messages.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute("""
            SELECT text FROM summaries
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (user_id,))
        row = c.fetchone()
        summary_text = row[0] if row else ""

        c.execute("""
            SELECT role, text, timestamp
            FROM messages
            WHERE user_id = ?
            ORDER BY timestamp ASC
        """, (user_id,))
        messages = c.fetchall()
        conn.close()

        lines = []
        if summary_text:
            lines.append(f"SUMMARY: {summary_text}")
        for role, text, _ in messages:
            lines.append(f"{role.upper()}: {text}")
        return "\n".join(lines) if lines else "No memory found for this user."
