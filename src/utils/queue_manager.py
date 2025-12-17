"""
Queue manager for producer-consumer pattern.
Uses SQLite database to track unprocessed outputs between processes.
"""

import sqlite3
import time
from pathlib import Path
from typing import Optional, List, Tuple
from contextlib import contextmanager
import threading


class QueueManager:
    """
    Manages a queue of unprocessed output files using SQLite.
    Thread-safe and process-safe for inter-process communication.
    Uses connection reuse to reduce overhead.
    """
    
    def __init__(self, db_path: str = "output_queue.db", max_unprocessed: int = 10):
        """
        Initialize the queue manager.
        
        Args:
            db_path: Path to SQLite database file
            max_unprocessed: Maximum number of unprocessed items allowed
        """
        self.db_path = Path(db_path)
        self.max_unprocessed = max_unprocessed
        self._lock = threading.Lock()
        self._connection = None  # Reuse connection for better performance
        self._init_database()
    
    def _init_database(self):
        """Initialize the database schema."""
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS output_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL UNIQUE,
                created_at REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                processed_at REAL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON output_queue(status)
        """)
        conn.commit()
    
    def _get_connection(self):
        """
        Get a database connection, reusing existing one if available.
        Creates new connection if needed or if existing one is closed.
        """
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.db_path),
                timeout=30.0,  # 30 second timeout for concurrent access
                check_same_thread=False
            )
            self._connection.row_factory = sqlite3.Row
        return self._connection
    
    @contextmanager
    def _get_connection_context(self):
        """
        Get a database connection context manager for operations that need isolation.
        Use this for write operations that need to be atomic.
        """
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _close_connection(self):
        """Close the persistent connection if it exists."""
        if self._connection is not None:
            try:
                self._connection.close()
            except:
                pass
            finally:
                self._connection = None
    
    def add_output(self, file_path: str) -> bool:
        """
        Add an output file to the queue.
        
        Args:
            file_path: Path to the output file
            
        Returns:
            True if added successfully, False if already exists
        """
        file_path = str(Path(file_path).resolve())
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("""
                    INSERT INTO output_queue (file_path, created_at, status)
                    VALUES (?, ?, 'pending')
                """, (file_path, time.time()))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                # File already in queue
                return False
            except sqlite3.OperationalError:
                # Connection may be closed, recreate it
                self._close_connection()
                conn = self._get_connection()
                try:
                    conn.execute("""
                        INSERT INTO output_queue (file_path, created_at, status)
                        VALUES (?, ?, 'pending')
                    """, (file_path, time.time()))
                    conn.commit()
                    return True
                except sqlite3.IntegrityError:
                    return False
    
    def count_unprocessed(self) -> int:
        """Count the number of unprocessed items in the queue."""
        try:
            conn = self._get_connection()
            cursor = conn.execute("""
                SELECT COUNT(*) as count
                FROM output_queue
                WHERE status = 'pending'
            """)
            result = cursor.fetchone()
            return result['count'] if result else 0
        except sqlite3.OperationalError:
            # Connection may be closed, recreate it
            self._close_connection()
            conn = self._get_connection()
            cursor = conn.execute("""
                SELECT COUNT(*) as count
                FROM output_queue
                WHERE status = 'pending'
            """)
            result = cursor.fetchone()
            return result['count'] if result else 0
    
    def can_produce(self) -> bool:
        """
        Check if producer can generate more outputs.
        
        Returns:
            True if there are less than max_unprocessed items, False otherwise
        """
        return self.count_unprocessed() < self.max_unprocessed
    
    def wait_until_can_produce(self, check_interval: float = 1.0):
        """
        Wait until producer can generate more outputs.
        
        Args:
            check_interval: Seconds to wait between checks
        """
        while not self.can_produce():
            time.sleep(check_interval)
    
    def get_next_pending(self) -> Optional[str]:
        """
        Get the next pending file path to process (FIFO order).
        
        Returns:
            File path or None if no pending items
        """
        try:
            conn = self._get_connection()
            cursor = conn.execute("""
                SELECT file_path
                FROM output_queue
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT 1
            """)
            result = cursor.fetchone()
            return result['file_path'] if result else None
        except sqlite3.OperationalError:
            # Connection may be closed, recreate it
            self._close_connection()
            conn = self._get_connection()
            cursor = conn.execute("""
                SELECT file_path
                FROM output_queue
                WHERE status = 'pending'
                ORDER BY created_at ASC
                LIMIT 1
            """)
            result = cursor.fetchone()
            return result['file_path'] if result else None
    
    def mark_processing(self, file_path: str) -> bool:
        """
        Mark a file as being processed.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if marked successfully, False if not found
        """
        file_path = str(Path(file_path).resolve())
        with self._lock:
            try:
                conn = self._get_connection()
                cursor = conn.execute("""
                    UPDATE output_queue
                    SET status = 'processing'
                    WHERE file_path = ? AND status = 'pending'
                """, (file_path,))
                conn.commit()
                return cursor.rowcount > 0
            except sqlite3.OperationalError:
                # Connection may be closed, recreate it
                self._close_connection()
                conn = self._get_connection()
                cursor = conn.execute("""
                    UPDATE output_queue
                    SET status = 'processing'
                    WHERE file_path = ? AND status = 'pending'
                """, (file_path,))
                conn.commit()
                return cursor.rowcount > 0
    
    def mark_completed(self, file_path: str):
        """
        Mark a file as completed and remove it from the queue.
        
        Args:
            file_path: Path to the file
        """
        file_path = str(Path(file_path).resolve())
        try:
            conn = self._get_connection()
            conn.execute("""
                DELETE FROM output_queue
                WHERE file_path = ?
            """, (file_path,))
            conn.commit()
        except sqlite3.OperationalError:
            # Connection may be closed, recreate it
            self._close_connection()
            conn = self._get_connection()
            conn.execute("""
                DELETE FROM output_queue
                WHERE file_path = ?
            """, (file_path,))
            conn.commit()
    
    def get_all_pending(self) -> List[Tuple[str, float]]:
        """
        Get all pending file paths with their creation times.
        
        Returns:
            List of (file_path, created_at) tuples
        """
        try:
            conn = self._get_connection()
            cursor = conn.execute("""
                SELECT file_path, created_at
                FROM output_queue
                WHERE status = 'pending'
                ORDER BY created_at ASC
            """)
            return [(row['file_path'], row['created_at']) for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            # Connection may be closed, recreate it
            self._close_connection()
            conn = self._get_connection()
            cursor = conn.execute("""
                SELECT file_path, created_at
                FROM output_queue
                WHERE status = 'pending'
                ORDER BY created_at ASC
            """)
            return [(row['file_path'], row['created_at']) for row in cursor.fetchall()]
    
    def cleanup_missing_files(self):
        """
        Remove entries from queue for files that no longer exist.
        """
        try:
            conn = self._get_connection()
            cursor = conn.execute("SELECT file_path FROM output_queue")
            for row in cursor.fetchall():
                file_path = row['file_path']
                if not Path(file_path).exists():
                    conn.execute("DELETE FROM output_queue WHERE file_path = ?", (file_path,))
            conn.commit()
        except sqlite3.OperationalError:
            # Connection may be closed, recreate it
            self._close_connection()
            conn = self._get_connection()
            cursor = conn.execute("SELECT file_path FROM output_queue")
            for row in cursor.fetchall():
                file_path = row['file_path']
                if not Path(file_path).exists():
                    conn.execute("DELETE FROM output_queue WHERE file_path = ?", (file_path,))
            conn.commit()
    
    def __del__(self):
        """Clean up connection on deletion."""
        self._close_connection()

