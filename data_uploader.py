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
            message = "Please don't close the game yet, we are uploading data to the database 😀.\n\nYou might experience (heavy) amounts of lag while it uploads, which could take around 30 seconds to finish.\n\nThis window will close automatically when the data uploading has finished."
            label = tk.Label(self.upload_popup, text=message, font=("Arial", 12), 
                           fg="white", bg="black", wraplength=450, justify="center")
            label.pack(expand=True, fill="both", padx=20, pady=20)
            
            # Update the popup to show it
            self.upload_popup.update()
            
            # Load CSV
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
                    print(f"Uploaded row {index + 1}")
                except Exception as e:
                    print(f"Failed to upload row {index + 1}: {e}")
                    print(f"Properties: {properties}")
                    # Close popup on error
                    if self.upload_popup:
                        self.upload_popup.destroy()
                        self.upload_popup = None
                    return False  # Stop on first error
            
            print("Done uploading.")
            
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