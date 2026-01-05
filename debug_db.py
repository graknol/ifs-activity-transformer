"""
Debug script to test database connection and queries.
Run this script to diagnose database issues.
"""
import sys
sys.path.insert(0, '.')

from app.database import OracleDBConnection
from app.config import DatabaseConfig, DisciplineConfig

def main():
    print("="*60)
    print("DEBUG: Database Connection Test")
    print("="*60)
    
    # Load configs
    print("\n1. Loading configuration...")
    try:
        db_config = DatabaseConfig.from_env()
        print(f"   Host: {db_config.host}")
        print(f"   Port: {db_config.port}")
        print(f"   Service: {db_config.service}")
        print(f"   User: {db_config.user}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return
    
    # Load discipline config
    print("\n2. Loading discipline config...")
    try:
        disc_config = DisciplineConfig.load()
        sql_queries = disc_config._data.get('sql_queries', {})
        print(f"   Found {len(sql_queries)} queries in config")
        print(f"   Available queries: {list(sql_queries.keys())}")
    except Exception as e:
        print(f"   ERROR: {e}")
        
    # Create connection
    print("\n3. Creating database connection...")
    db = OracleDBConnection(config=db_config, discipline_config=disc_config)
    
    # Connect
    print("\n4. Connecting to database...")
    if db.connect():
        print("   SUCCESS: Connected!")
    else:
        print("   FAILED: Could not connect")
        return
    
    # Test simple query
    print("\n5. Testing simple count query...")
    try:
        test_query = "SELECT COUNT(*) as cnt FROM IFSAPP.activity_tab WHERE ROWNUM = 1"
        print(f"   Query: {test_query}")
        df = db.execute_custom_query(test_query)
        print(f"   Result: {df}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Get training query
    print("\n6. Getting training query (sample)...")
    training_query = db.get_training_query(query_type='sample')
    print(f"   Query length: {len(training_query)} chars")
    print(f"   First 500 chars:")
    print(f"   {training_query[:500]}")
    
    # Check if query contains correct schema
    print("\n7. Validating query syntax...")
    if 'IFSAPP.' in training_query:
        print("   ✓ Contains IFSAPP. schema prefix")
    else:
        print("   ✗ MISSING IFSAPP. schema prefix!")
        
    if '@IFS10PRD' in training_query:
        print("   ✗ Contains @IFS10PRD db link (should be removed!)")
    else:
        print("   ✓ No @IFS10PRD db link")
        
    if 'category1_id' in training_query:
        print("   ✓ Uses category1_id column")
    elif 'project_category_id' in training_query:
        print("   ✗ Uses wrong column project_category_id!")
    else:
        print("   ? No category column found")
    
    # Execute training query
    print("\n8. Executing training query...")
    try:
        df = db.fetch_training_data(query_type='sample')
        print(f"   Rows returned: {len(df)}")
        if len(df) > 0:
            print(f"   Columns: {list(df.columns)}")
            print(f"   First row:")
            print(f"   {df.iloc[0].to_dict()}")
        else:
            print("   WARNING: Empty result!")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Try an even simpler query
    print("\n9. Testing direct activity query (no joins)...")
    try:
        simple_query = """
        SELECT activity_seq, project_id, sub_project_id, description
        FROM IFSAPP.activity_tab 
        WHERE rowstate IN ('Closed', 'Completed') 
          AND description IS NOT NULL
          AND ROWNUM <= 5
        """
        print(f"   Query: {simple_query.strip()}")
        df = db.execute_custom_query(simple_query)
        print(f"   Rows: {len(df)}")
        if len(df) > 0:
            print(df)
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    # Disconnect
    db.disconnect()
    print("\n10. Disconnected.")
    print("="*60)

if __name__ == '__main__':
    main()
