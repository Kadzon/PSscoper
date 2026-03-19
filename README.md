## BOSS PS Scoper
Install Chainlit to run the application
Set up API Keys for the following
AI models
TAVILY Key

### Authenticating to Google Drive
Join the BOSS Project: Click the project dropdown in the top left and select BOSS.

Enable the API: In the search bar at the top, type "Google Drive API" and click Enable.

Configure OAuth Consent Screen:
Go to APIs & Services > OAuth consent screen.
Create Credentials:
Go to APIs & Services > Credentials.
Click + Create Credentials > OAuth client ID.
Select Desktop App as the application type to run locally.
Click Create, then download the JSON file. Rename it to credentials.json and place it in your project folder.

### Possible Errors
Incase of trouble reading the documents, you should check the diagnostic script on connect_drive.py that is commented out and run it to ensure you understand the correct folder specifications and mimetypes to be exported.