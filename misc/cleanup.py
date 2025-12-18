#!/usr/bin/env python3
"""
Cleanup script for Constellation project
Removes old completed training jobs and registered models
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timedelta

def get_database_path():
    """Find the database file"""
    db_paths = [
        Path("constellation.db"),  # Server uses this by default
        Path("server/constellation.db"),
        Path("./constellation.db")
    ]
    
    for path in db_paths:
        if path.exists():
            return path
    
    return None

def cleanup_completed_jobs(keep_recent=5, days_old=None):
    """Clean up completed jobs, keeping the most recent N or jobs newer than X days"""
    db_path = get_database_path()
    
    if not db_path:
        print(f"❌ Database not found in expected locations")
        return False
    
    print(f"📁 Using database: {db_path.absolute()}")
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        # Get all completed jobs
        cursor.execute("""
            SELECT id, name, status, created_at, completed_at 
            FROM training_jobs 
            WHERE status = 'completed'
            ORDER BY completed_at DESC
        """)
        completed_jobs = cursor.fetchall()
        
        print(f"\n📊 Found {len(completed_jobs)} completed job(s)")
        
        if len(completed_jobs) == 0:
            print("✅ No completed jobs to clean up")
            return True
        
        # Determine which jobs to delete
        jobs_to_delete = []
        
        if days_old:
            # Delete jobs older than X days
            cutoff_date = datetime.now() - timedelta(days=days_old)
            for job in completed_jobs:
                if job[4]:  # completed_at exists
                    try:
                        completed_date = datetime.fromisoformat(job[4].replace('Z', '+00:00'))
                        if completed_date.replace(tzinfo=None) < cutoff_date:
                            jobs_to_delete.append(job)
                    except:
                        pass
        else:
            # Keep most recent N jobs
            if len(completed_jobs) > keep_recent:
                jobs_to_delete = completed_jobs[keep_recent:]
        
        if not jobs_to_delete:
            print(f"✅ No jobs to delete (keeping all {len(completed_jobs)} completed jobs)")
            return True
        
        print(f"\n🗑️  Will delete {len(jobs_to_delete)} job(s):")
        for job in jobs_to_delete[:10]:
            print(f"   - {job[1]} ({job[0][:8]}...)")
        if len(jobs_to_delete) > 10:
            print(f"   ... and {len(jobs_to_delete) - 10} more")
        
        # Delete jobs
        deleted_count = 0
        for job in jobs_to_delete:
            job_id = job[0]
            
            # Delete related device_training records first
            cursor.execute("DELETE FROM device_training WHERE job_id = ?", (job_id,))
            
            # Delete the job
            cursor.execute("DELETE FROM training_jobs WHERE id = ?", (job_id,))
            deleted_count += 1
        
        conn.commit()
        
        print(f"\n✅ Successfully deleted {deleted_count} completed job(s)")
        print(f"   Kept {len(completed_jobs) - deleted_count} most recent completed job(s)")
        return True
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        conn.rollback()
        import traceback
        traceback.print_exc()
        return False
    finally:
        conn.close()

def cleanup_models(keep_recent=5, days_old=None, delete_all=False):
    """Clean up old models, keeping the most recent N or models newer than X days"""
    db_path = get_database_path()
    
    if not db_path:
        print(f"❌ Database not found in expected locations")
        return False
    
    print(f"📁 Using database: {db_path.absolute()}")
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        # Get all models
        cursor.execute("""
            SELECT id, name, model_type, status, created_at, checkpoint_path
            FROM models 
            ORDER BY created_at DESC
        """)
        all_models = cursor.fetchall()
        
        print(f"\n📊 Found {len(all_models)} model(s)")
        
        if len(all_models) == 0:
            print("✅ No models to clean up")
            return True
        
        # Determine which models to delete
        models_to_delete = []
        
        if delete_all:
            models_to_delete = all_models
        elif days_old:
            # Delete models older than X days
            cutoff_date = datetime.now() - timedelta(days=days_old)
            for model in all_models:
                if model[4]:  # created_at exists
                    try:
                        created_date = datetime.fromisoformat(model[4].replace('Z', '+00:00'))
                        if created_date.replace(tzinfo=None) < cutoff_date:
                            models_to_delete.append(model)
                    except:
                        pass
        else:
            # Keep most recent N models
            if len(all_models) > keep_recent:
                models_to_delete = all_models[keep_recent:]
        
        if not models_to_delete:
            print(f"✅ No models to delete (keeping all {len(all_models)} models)")
            return True
        
        print(f"\n🗑️  Will delete {len(models_to_delete)} model(s):")
        for model in models_to_delete[:10]:
            print(f"   - {model[1]} ({model[0][:8]}...)")
        if len(models_to_delete) > 10:
            print(f"   ... and {len(models_to_delete) - 10} more")
        
        # Delete models
        deleted_count = 0
        for model in models_to_delete:
            model_id = model[0]
            cursor.execute("DELETE FROM models WHERE id = ?", (model_id,))
            deleted_count += 1
        
        conn.commit()
        
        print(f"\n✅ Successfully deleted {deleted_count} model(s)")
        print(f"   Kept {len(all_models) - deleted_count} most recent model(s)")
        return True
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        conn.rollback()
        import traceback
        traceback.print_exc()
        return False
    finally:
        conn.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Cleanup old completed training jobs and registered models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Keep 5 most recent completed jobs
  python3 cleanup.py --jobs --keep 5
  
  # Keep 3 most recent models
  python3 cleanup.py --models --keep 3
  
  # Cleanup both, keeping 5 of each
  python3 cleanup.py --jobs --models --keep 5
  
  # Delete jobs older than 30 days
  python3 cleanup.py --jobs --days 30
  
  # Delete all models
  python3 cleanup.py --models --all
        """
    )
    
    parser.add_argument("--jobs", action="store_true", help="Cleanup completed training jobs")
    parser.add_argument("--models", action="store_true", help="Cleanup registered models")
    parser.add_argument("--keep", type=int, default=5, help="Number of recent items to keep (default: 5)")
    parser.add_argument("--days", type=int, help="Delete items older than X days")
    parser.add_argument("--all", action="store_true", help="Delete all items (models only)")
    
    args = parser.parse_args()
    
    if not args.jobs and not args.models:
        parser.print_help()
        print("\n⚠️  Please specify --jobs and/or --models")
        return
    
    success = True
    
    if args.jobs:
        print("="*60)
        print("🧹 Cleaning Up Completed Jobs")
        print("="*60)
        if args.days:
            print(f"   Deleting jobs older than {args.days} days...")
            success = cleanup_completed_jobs(days_old=args.days) and success
        else:
            print(f"   Keeping {args.keep} most recent completed jobs...")
            success = cleanup_completed_jobs(keep_recent=args.keep) and success
    
    if args.models:
        print("\n" + "="*60)
        print("🧹 Cleaning Up Registered Models")
        print("="*60)
        if args.all:
            print("   Deleting ALL models...")
            success = cleanup_models(delete_all=True) and success
        elif args.days:
            print(f"   Deleting models older than {args.days} days...")
            success = cleanup_models(days_old=args.days) and success
        else:
            print(f"   Keeping {args.keep} most recent models...")
            success = cleanup_models(keep_recent=args.keep) and success
    
    if success:
        print("\n" + "="*60)
        print("✅ Cleanup Complete!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("⚠️  Cleanup completed with errors")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main()

