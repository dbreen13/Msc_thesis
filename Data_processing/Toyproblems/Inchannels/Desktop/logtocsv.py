import csv

# Open the log file for reading
with open('Inch_nograd.log', 'r') as log_file:
    # Open a CSV file for writing
    with open('testen.csv', 'w', newline='') as csv_file:
        # Define the CSV writer
        csv_writer = csv.writer(csv_file)
        
        # Write header to CSV file
        csv_writer.writerow(['Datetime', 'Info'])
        
        # Iterate over each line in the log file
        for line in log_file:
            # Split the line into parts based on ' - '
            parts = line.strip().split(' - ')
            
            # Check if the line has at least two parts
            if len(parts) >= 2:
                datetime_str = parts[0]  # Get the datetime string
                info = parts[-1]  # Get the info
                
                # Split the datetime string into date and time
                datetime_parts = datetime_str.split(' ')
                
                # Check if the datetime string has expected format
                if len(datetime_parts) == 2:
                    date, time = datetime_parts
                    # Split the time part by comma and take the first part (without milliseconds)
                    time = time.split(',')[0]
                    
                    # Combine date and time into a single datetime string
                    datetime_combined = f"{date} {time}"
                    
                    # Write datetime and info to the CSV file
                    csv_writer.writerow([datetime_combined, info])
                else:
                    print(f"Ignoring line: {line.strip()}")  # Print warning for unexpected line
            else:
                print(f"Ignoring line: {line.strip()}")  # Print warning for unexpected line
