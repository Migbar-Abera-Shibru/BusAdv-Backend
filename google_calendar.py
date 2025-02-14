import os
from datetime import datetime, timedelta
from flask import Flask, redirect, request, url_for
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# Initialize Flask app
app = Flask(__name__)

# Google OAuth settings
SCOPES = ["https://www.googleapis.com/auth/calendar"]
CLIENT_SECRETS_FILE = "./config/credentials.json"

# Temporary storage for credentials (use a database in production)
credentials = None

@app.route("/auth/google")
def auth_google():
    """Redirect the user to Google's OAuth consent page."""
    flow = InstalledAppFlow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=url_for("auth_google_callback", _external=True)
    )
    auth_url, _ = flow.authorization_url(prompt="consent")
    return redirect(auth_url)

@app.route("/auth/google/callback")
def auth_google_callback():
    """Handle the OAuth callback and store credentials."""
    global credentials
    flow = InstalledAppFlow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=url_for("auth_google_callback", _external=True)
    )
    flow.fetch_token(authorization_response=request.url)
    credentials = flow.credentials

    # Redirect to the events page
    return redirect(url_for("get_events"))

@app.route("/events")
def get_events():
    """Fetch events from Google Calendar and return them in a textual format."""
    global credentials
    if not credentials or not credentials.valid:
        return "Not authenticated. Please sign in first.", 401

    try:
        # Fetch events from Google Calendar
        service = build("calendar", "v3", credentials=credentials)

        # Define the time range (now to next 7 days)
        now = datetime.utcnow().isoformat() + "Z"  # 'Z' indicates UTC time
        next_week = (datetime.utcnow() + timedelta(days=7)).isoformat() + "Z"

        # Fetch events
        events_result = (
            service.events()
            .list(
                calendarId="primary",  # Use the user's primary calendar
                timeMin=now,          # Start time (now)
                timeMax=next_week,    # End time (7 days from now)
                maxResults=10,        # Limit the number of events
                singleEvents=True,    # Expand recurring events
                orderBy="startTime",  # Order by start time
            )
            .execute()
        )

        # Extract events from the response
        events = events_result.get("items", [])

        # Format events as plain text
        if not events:
            return "No events found."

        events_text = "Google Calendar Events:\n\n"
        for event in events:
            events_text += f"Event: {event.get('summary', 'No title')}\n"
            events_text += f"Time: {event['start'].get('dateTime', event['start'].get('date'))}\n"
            events_text += f"Description: {event.get('description', 'No description available')}\n"
            events_text += "-" * 40 + "\n"

        return events_text

    except Exception as e:
        return f"Error fetching events: {str(e)}", 500

if __name__ == "__main__":
    # Allow HTTP for local testing (remove in production)
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    # Run the Flask app
    app.run(port=5000)