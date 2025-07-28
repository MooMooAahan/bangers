import pandas as pd
from notion_client import Client


class DataUploader:
    """Class to handle uploading game data to Notion"""
    
    def __init__(self):
        self.notion = Client(auth="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")
        self.database_id = "23d2a82eac558020ab77c53fce4e1795"
    
    def upload_data(self):
        """Upload all data from log.csv to Notion"""
        try:
            print("Starting the upload process...")
            
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
                    return False  # Stop on first error
            
            print("Done uploading.")
            return True
            
        except Exception as e:
            print(f"Error in upload_data: {e}")
            return False


# For backward compatibility - if run directly as a script
if __name__ == "__main__":
    uploader = DataUploader()
    uploader.upload_data()