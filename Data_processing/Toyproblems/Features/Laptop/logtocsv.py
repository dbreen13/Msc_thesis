import csv

# Open the logger log file
with open('Feat.log', 'r') as log_file:
    # Open a new csv file, which will be filled with data
    with open('testen.csv', 'w', newline='') as csv_file:
        # Define the CSV writer
        csv_writer = csv.writer(csv_file)
        
        # Give headers to the CSV file, with the Datetime stamp and the experiment name
        csv_writer.writerow(['Datetime', 'Name'])
        
        # Iterate over each line in the log file
        for line in log_file:
            # Based on the ' - ' split the lof file lines
            parts = line.strip().split(' - ')
            
            # Check if the line has at least two parts
            if len(parts) >= 2:
                datetime_str = parts[0]  # Get the datetime string
                name = parts[-1]  # Get the name of the experiment
                
                # Split the datetime string into seperate date and time to remove the milliseconds
                datetime_parts = datetime_str.split(' ')
                
                # Check if the datetime string has two parts
                if len(datetime_parts) == 2:
                    date, time = datetime_parts
                    # Split the time part by comma and remove the milliseconds
                    time = time.split(',')[0]
                    
                    # Combine date and time into a single datetime stamp
                    datetime_combined = f"{date} {time}"
                    
                    # Write datetime and info to the CSV file
                    csv_writer.writerow([datetime_combined, name])
                else:
                    print(f"Ignoring line: {line.strip()}")  # Print warning for unexpected line
            else:
                print(f"Ignoring line: {line.strip()}")  # Print warning for unexpected line
