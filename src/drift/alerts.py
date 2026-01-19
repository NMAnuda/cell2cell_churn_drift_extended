import smtplib
from email.mime.text import MIMEText
from datetime import datetime

def send_drift_alert(drift_type, psi, email_to='alert@company.com'):
    """Send email alert on drift (optional — skip if no creds)."""
    # FIXED: Print only (disable SMTP for demo)
    print(f"Drift Alert: {drift_type} (PSI {psi:.3f}) at {datetime.now()}")
    print("Email skipped — configure Gmail app password for production.")
    

    # msg = MIMEText(f" Drift Detected: {drift_type} (PSI {psi:.3f}) at {datetime.now()}")
    # msg['Subject'] = "Churn Model Drift Alert"
    # msg['From'] = 'your-real@gmail.com'  # Replace with real
    # msg['To'] = email_to
    # with smtplib.SMTP('smtp.gmail.com', 587) as server:
    #     server.starttls()
    #     server.login('your-real@gmail.com', 'your-app-password')  # 16-char app pass
    #     server.send_message(msg)
    # print(" Drift alert sent via email!")

# Slack webhook example (add if needed)
def send_slack_alert(webhook_url, message):
    import requests
    payload = {"text": f" {message}"}
    requests.post(webhook_url, json=payload)
    print(" Drift alert sent via Slack!")