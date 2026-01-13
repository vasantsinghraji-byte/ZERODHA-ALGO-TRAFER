import pytest
import psycopg2
from time import sleep
from infrastructure.db import (
    init_db, close_db_pool, get_db_connection, execute_query,
    execute_batch_query, DatabaseError, pool
)

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Setup test database environment"""
    init_db()
    yield
    close_db_pool()

@pytest.fixture(autouse=True)
def setup_test_table():
    """Create and cleanup test table for each test"""
    execute_query("""
        CREATE TEMP TABLE IF NOT EXISTS test_data (
            id SERIAL PRIMARY KEY,
            value TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    yield
    execute_query("DROP TABLE IF EXISTS test_data")

def test_connection_pool():
    """Test connection pool initialization and management"""
    assert pool._pool is not None
    
    # Test multiple connections
    conns = []
    for _ in range(3):
        conn = pool.get_connection()
        assert conn is not None
        conns.append(conn)
    
    # Return connections
    for conn in conns:
        pool.return_connection(conn)

def test_transaction_commit():
    """Test successful transaction commit"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO test_data (value) VALUES (%s)", ("test_value",))
    
    result = execute_query("SELECT value FROM test_data")
    assert len(result) == 1
    assert result[0]["value"] == "test_value"

def test_transaction_rollback():
    """Test transaction rollback on error"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO test_data (value) VALUES (%s)", ("before_error",))
                cur.execute("SELECT * FROM nonexistent_table")
    except DatabaseError:
        pass
    
    result = execute_query("SELECT value FROM test_data")
    assert len(result) == 0  # Should be rolled back

def test_batch_operations():
    """Test batch query execution"""
    test_data = [
        ("value1",),
        ("value2",),
        ("value3",)
    ]
    
    execute_batch_query(
        "INSERT INTO test_data (value) VALUES (%s)",
        test_data
    )
    
    result = execute_query("SELECT value FROM test_data ORDER BY id")
    assert len(result) == 3
    assert [r["value"] for r in result] == ["value1", "value2", "value3"]

def test_query_retry():
    """Test query retry mechanism"""
    def failing_query():
        # Simulate temporary connection issue
        pool.close_all()
        sleep(0.1)
        return execute_query("SELECT 1")
    
    result = failing_query()
    assert len(result) == 1
    assert result[0]["?column?"] == 1

def test_error_handling():
    """Test various error conditions"""
    # Test invalid SQL
    with pytest.raises(DatabaseError):
        execute_query("INVALID SQL")
    
    # Test constraint violation
    execute_query("""
        CREATE TEMP TABLE unique_test (
            id SERIAL PRIMARY KEY,
            value TEXT UNIQUE
        )
    """)
    execute_query("INSERT INTO unique_test (value) VALUES (%s)", ("unique",))
    
    with pytest.raises(DatabaseError):
        execute_query("INSERT INTO unique_test (value) VALUES (%s)", ("unique",))

def test_query_parameters():
    """Test query parameter handling"""
    # Test different parameter types
    execute_query(
        "INSERT INTO test_data (value) VALUES (%s)",
        ("test",)
    )
    
    result = execute_query(
        "SELECT value FROM test_data WHERE value = %s",
        ("test",)
    )
    assert len(result) == 1
    assert result[0]["value"] == "test"
