import pandas as pd
from notion_client import Client
import tkinter as tk
from tkinter import messagebox


class DataUploader:
    """Class to handle uploading game data to Notion"""
    
    def __init__(self):
        self.notion = Client(auth="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")
        self.database_id = "23d2a82eac558020ab77c53fce4e1795"
        self.upload_popup = None
    
    def get_next_run_id(self):
        """Get the next run ID by finding the largest existing run ID and adding 1"""
        try:
            print("Fetching largest run ID from database...")
            
            # Query the database for all entries
            response = self.notion.databases.query(
                database_id=self.database_id,
                sorts=[{
                    "property": "local_run_id",
                    "direction": "descending"
                }],
                page_size=1  # Only get the first result (highest run ID)
            )
            
            if response["results"]:
                # Get the largest run ID from the first result
                largest_run_id = response["results"][0]['properties'].get('local_run_id', {}).get('rich_text', [{}])[0].get('text', {}).get('content', '0')
                try:
                    largest_run_id = int(largest_run_id)
                    next_run_id = largest_run_id + 1
                    print(f"Largest run ID in database: {largest_run_id}")
                    print(f"Next run ID will be: {next_run_id}")
                    return next_run_id
                except ValueError:
                    print("Error parsing run ID, defaulting to 0")
                    return 0
            else:
                print("No entries found in database, starting with run ID 0")
                return 0
                
        except Exception as e:
            print(f"Error getting next run ID: {e}")
            print("Defaulting to run ID 0")
            return 0
    
    def assign_run_id_to_log(self, run_id):
        """Assign the given run ID to all entries in log.csv"""
        try:
            print(f"Assigning run ID {run_id} to all log entries...")
            
            # Read the log.csv file
            df = pd.read_csv("log.csv")
            
            # Check if there are any entries (not just headers)
            if len(df) > 0:
                # Assign the run ID to all rows
                df['local_run_id'] = run_id
                
                # Write back to log.csv
                df.to_csv("log.csv", index=False)
                print(f"Successfully assigned run ID {run_id} to {len(df)} log entries")
                return True
            else:
                print("No log entries found to assign run ID to")
                return False
                
        except Exception as e:
            print(f"Error assigning run ID to log: {e}")
            return False
    
    def upload_data(self):
        """Upload all data from log.csv to Notion"""
        try:
            print("Starting the upload process...")
            
            # Create popup window
            self.upload_popup = tk.Toplevel()
            self.upload_popup.title("Uploading Data")
            self.upload_popup.geometry("500x200")
            self.upload_popup.configure(bg="black")
            
            # Center the popup
            self.upload_popup.transient()  # Make it a top-level window
            self.upload_popup.grab_set()   # Make it modal
            
            # Create message label
            message = "Please don't close the game yet, we are uploading data to the database 😀.\n\nYou will experience (heavy) amounts of lag while it uploads, which could take around 30 seconds to finish. (You are going to get that annoying 'Page not responding' thing, but that's fine.)\n\nThis window will close automatically when the data uploading has finished."
            label = tk.Label(self.upload_popup, text=message, font=("Arial", 12), 
                           fg="white", bg="black", wraplength=450, justify="center")
            label.pack(expand=True, fill="both", padx=20, pady=20)
            
            # Update the popup to show it
            self.upload_popup.update()
            
            # Get the next run ID and assign it to log entries
            next_run_id = self.get_next_run_id()
            if not self.assign_run_id_to_log(next_run_id):
                print("Failed to assign run ID, aborting upload")
                if self.upload_popup:
                    self.upload_popup.destroy()
                    self.upload_popup = None
                return False
            
            # Load CSV with the assigned run ID
            df = pd.read_csv("log.csv")
            
            for index, row in df.iterrows():
                properties = {}
                
                # Add all columns as rich_text properties
                for col in df.columns:
                    value = row[col]
                    properties[col] = {
                        "rich_text": [{"text": {"content": str(value)}}]
                    }

                try:
                    self.notion.pages.create(
                        parent={"database_id": self.database_id},
                        properties=properties
                    )
                    print(f"Uploaded row {index + 1} with run ID {next_run_id}")
                except Exception as e:
                    print(f"Failed to upload row {index + 1}: {e}")
                    print(f"Properties: {properties}")
                    # Close popup on error
                    if self.upload_popup:
                        self.upload_popup.destroy()
                        self.upload_popup = None
                    return False  # Stop on first error
            
            print(f"Done uploading run ID {next_run_id}.")
            
            # Close popup when upload is complete
            if self.upload_popup:
                self.upload_popup.destroy()
                self.upload_popup = None
            
            return True
            
        except Exception as e:
            print(f"Error in upload_data: {e}")
            # Close popup on error
            if self.upload_popup:
                self.upload_popup.destroy()
                self.upload_popup = None
            return False


# For backward compatibility - if run directly as a script
if __name__ == "__main__":
    uploader = DataUploader()
    uploader.upload_data()