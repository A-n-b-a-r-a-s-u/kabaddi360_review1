#!/usr/bin/env python
"""
Utility to view and search pipeline logs.
Usage:
    python view_logs.py              # List all logs
    python view_logs.py latest       # View latest log
    python view_logs.py search ERROR  # Search for ERROR in latest
    python view_logs.py search ERROR <timestamp>  # Search specific log
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def list_logs(output_dir="outputs"):
    """List all available logs."""
    logs_dir = Path(output_dir) / "logs"
    if not logs_dir.exists():
        print(f"No logs directory found in {output_dir}")
        return
    
    index_file = logs_dir / "logs_index.json"
    if not index_file.exists():
        print(f"No logs index found")
        return
    
    with open(index_file, 'r') as f:
        logs = json.load(f)
    
    print("\n" + "="*80)
    print("AVAILABLE PIPELINE LOGS")
    print("="*80)
    print(f"{'Timestamp':<20} {'Log File':<50} {'Status':<10}")
    print("-"*80)
    
    for timestamp, info in sorted(logs.items(), reverse=True):
        log_file = Path(info['log_file']).name
        status = info.get('status', 'unknown')
        print(f"{timestamp:<20} {log_file:<50} {status:<10}")
    
    print("="*80 + "\n")

def view_log(output_dir="outputs", timestamp=None, lines=50):
    """View a specific log file."""
    logs_dir = Path(output_dir) / "logs"
    
    if not logs_dir.exists():
        print(f"No logs directory found in {output_dir}")
        return
    
    # If no timestamp, use latest
    if not timestamp:
        log_files = sorted(logs_dir.glob("pipeline_*.log"), reverse=True)
        if not log_files:
            print("No log files found")
            return
        log_file = log_files[0]
    else:
        log_file = logs_dir / f"pipeline_{timestamp}.log"
    
    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return
    
    with open(log_file, 'r') as f:
        content = f.readlines()
    
    print(f"\n{'='*80}")
    print(f"LOG FILE: {log_file.name}")
    print(f"Total lines: {len(content)}")
    print(f"Showing last {min(lines, len(content))} lines")
    print("="*80 + "\n")
    
    for line in content[-lines:]:
        print(line.rstrip())
    
    print("\n" + "="*80 + "\n")

def search_logs(query, output_dir="outputs", timestamp=None):
    """Search for a query in log files."""
    logs_dir = Path(output_dir) / "logs"
    
    if not logs_dir.exists():
        print(f"No logs directory found in {output_dir}")
        return
    
    # If timestamp specified, search only that file
    if timestamp:
        log_files = [logs_dir / f"pipeline_{timestamp}.log"]
    else:
        # Search all logs, latest first
        log_files = sorted(logs_dir.glob("pipeline_*.log"), reverse=True)
    
    results = []
    for log_file in log_files:
        if not log_file.exists():
            continue
        
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            if query.lower() in line.lower():
                results.append((log_file.name, line_num, line.rstrip()))
    
    if not results:
        print(f"\nNo matches found for: {query}")
        return
    
    print(f"\n{'='*80}")
    print(f"SEARCH RESULTS: {query} ({len(results)} matches)")
    print("="*80 + "\n")
    
    current_file = None
    for log_file, line_num, line in results:
        if log_file != current_file:
            print(f"\n--- {log_file} ---")
            current_file = log_file
        
        print(f"[Line {line_num}] {line}")
    
    print("\n" + "="*80 + "\n")

def main():
    if len(sys.argv) < 2:
        # List all logs
        list_logs()
        print("Usage:")
        print("  python view_logs.py              # List all logs")
        print("  python view_logs.py latest       # View latest log (last 50 lines)")
        print("  python view_logs.py latest 200   # View latest log (last 200 lines)")
        print("  python view_logs.py search ERROR # Search for ERROR in latest")
        print("  python view_logs.py search ERROR <timestamp>  # Search specific log")
        print("  python view_logs.py <timestamp>  # View specific log by timestamp")
        return
    
    command = sys.argv[1].lower()
    
    if command == "latest":
        lines = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        view_log(lines=lines)
    
    elif command == "search":
        query = sys.argv[2] if len(sys.argv) > 2 else "ERROR"
        timestamp = sys.argv[3] if len(sys.argv) > 3 else None
        search_logs(query, timestamp=timestamp)
    
    else:
        # Assume it's a timestamp
        view_log(timestamp=command)

if __name__ == "__main__":
    main()
