from flask import Blueprint, redirect, request, url_for, jsonify
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from datetime import datetime, timedelta
import os

# Create a Blueprint for Google Calendar
google_calendar_bp = Blueprint("google_calendar", __name__)

SCOPES = ["https://www.googleapis.com/auth/calendar"]
CLIENT_SECRETS_FILE = "./config/credentials.json"

credentials = None

@google_calendar_bp.route("/auth/google")
def auth_google():
    flow = InstalledAppFlow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=url_for("google_calendar.auth_google_callback", _external=True)
    )
    auth_url, _ = flow.authorization_url(prompt="consent")
    return redirect(auth_url)

@google_calendar_bp.route("/auth/google/callback")
def auth_google_callback():
    global credentials
    flow = InstalledAppFlow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=url_for("google_calendar.auth_google_callback", _external=True)
    )
    flow.fetch_token(authorization_response=request.url)
    credentials = flow.credentials
    return redirect("http://localhost:3000/calendar?auth_success=true")

@google_calendar_bp.route("/api/events")
def get_events():
    global credentials
    if not credentials or not credentials.valid:
        return jsonify({"error": "Not authenticated. Please sign in first."}), 401

    try:
        service = build("calendar", "v3", credentials=credentials)
        now = datetime.utcnow().isoformat() + "Z"
        next_week = (datetime.utcnow() + timedelta(days=7)).isoformat() + "Z"

        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=now,
                timeMax=next_week,
                maxResults=10,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )

        events = events_result.get("items", [])
        events_json = [
            {
                "id": event.get("id"),
                "summary": event.get("summary", "No title"),
                "start": event["start"].get("dateTime", event["start"].get("date")),
                "end": event["end"].get("dateTime", event["end"].get("date")),
                "description": event.get("description", "No description available"),
            }
            for event in events
        ]

        return jsonify({"events": events_json})

    except Exception as e:
        return jsonify({"error": f"Error fetching events: {str(e)}"}), 500

def init_google_calendar(app):
    """Register Google Calendar routes with the main Flask app."""
    app.register_blueprint(google_calendar_bp)
