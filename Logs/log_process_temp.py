import json


def trim_json_data(input_file, output_file):
    # Load the JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Process log_entries
    if 'log_entries' in data:
        log_entries = data['log_entries']
        trimmed_log = []
        
        # Find the cutoff point where iteration >= 260
        for entry in log_entries:
            if entry.get('iteration', 0) >= 260:
                break
            trimmed_log.append(entry)
        
        data['log_entries'] = trimmed_log
    
    # Process trajectory (Python list slicing is end-exclusive)
    if 'trajectory' in data:
        data['trajectory'] = data['trajectory'][:260+1]  # 0-260 inclusive
    
    # Save the modified data
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Trimmed JSON saved to {output_file}")

# Usage example
trim_json_data(
    "Logs/dwa_log_details_20250306_155027/log_details_old.json",
    "Logs/dwa_log_details_20250306_155027/log_details.json"
)
