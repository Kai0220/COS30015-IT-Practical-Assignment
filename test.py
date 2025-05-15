import os
import re
import email
import pandas as pd
import requests
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup

def parse_email(file_path):
    with open(file_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)
    headers = dict(msg.items())
    body = msg.get_body(preferencelist=('plain', 'html'))
    return {
        "subject": headers.get("Subject", ""),
        "from": headers.get("From", ""),
        "to": headers.get("To", ""),
        "received": headers.get("Received", ""),
        "date": headers.get("Date", ""),
        "body": body.get_content() if body else ""
    }

def clean_email_body(raw_html):
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(separator=" ", strip=True)

def extract_info_from_received(received_header):
    ip_match = re.search(r'\[?(\d{1,3}(?:\.\d{1,3}){3})\]?', received_header)
    ip = ip_match.group(1) if ip_match else None

    domain_match = re.search(r'from\s+([\w\.-]+)', received_header, re.IGNORECASE)
    domain = domain_match.group(1) if domain_match else None

    return ip, domain

def load_emails_from_folder(folder_path, label):
    emails = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        try:
            parsed = parse_email(file_path)
            ip, domain = extract_info_from_received(parsed["received"])
            emails.append({
                "from": parsed["from"],
                "subject": parsed["subject"],
                "date": parsed["date"],
                "text": clean_email_body(parsed["body"]),
                "ip": ip,
                "domain": domain,
                "label": label
            })
        except Exception as e:
            continue
    return emails

def enrich_ip_metadata(ip):
    if not ip or ip.startswith("192.") or ip.startswith("10.") or ip.startswith("127."):
        return {"country": None, "org": None}
    try:
        r = requests.get(f"http://ip-api.com/json/{ip}", timeout=5)
        if r.status_code == 200:
            data = r.json()
            return {
                "country": data.get("country"),
                "org": data.get("org")
            }
    except:
        pass
    return {"country": None, "org": None}

# Load folders and combine into a single DataFrame
ham_folder = "easy_ham"
spam_folder = "spam"
spam2_folder = "spam_2"

ham_emails = load_emails_from_folder(ham_folder, 0)
spam_emails = load_emails_from_folder(spam_folder, 1)
spam2_emails = load_emails_from_folder(spam2_folder, 1)

all_emails = pd.DataFrame(ham_emails + spam_emails + spam2_emails)
all_emails = all_emails[all_emails['text'].notnull()].sample(frac=1).reset_index(drop=True)

# === Get top 50 most frequent spam IPs ===
top_ips = (
    all_emails[all_emails['label'] == 1]
    .dropna(subset=["ip"])
    .groupby('ip')
    .size()
    .sort_values(ascending=False)
    .head(50)
    .index
)

# Filter all spam emails with these IPs
sampled = all_emails[
    (all_emails['label'] == 1) & (all_emails['ip'].isin(top_ips))
].copy()

# Enrich IP metadata (lookup only unique IPs to avoid rate limits)
unique_sampled_ips = sampled['ip'].dropna().unique()
ip_metadata = {ip: enrich_ip_metadata(ip) for ip in unique_sampled_ips}
sampled['country'] = sampled['ip'].map(lambda ip: ip_metadata.get(ip, {}).get('country', None))
sampled['org'] = sampled['ip'].map(lambda ip: ip_metadata.get(ip, {}).get('org', None))
sampled['text_preview'] = sampled['text'].str.replace('\n', ' ').str.replace('\r', ' ')

# Export to CSV
csv_path = "top_50_spam_ips.csv"
sampled.to_csv(csv_path, index=False)

# Export to TXT (formatted)
txt_path = "top_50_spam_ips.txt"
with open(txt_path, "w", encoding="utf-8") as f:
    for _, row in sampled.iterrows():
        f.write(f"IP: {row['ip']}\n")
        f.write(f"Domain: {row['domain']}\n")
        f.write(f"Country: {row['country']}\n")
        f.write(f"Org: {row['org']}\n")
        f.write(f"From: {row['from']}\n")
        f.write(f"Subject: {row['subject']}\n")
        f.write(f"Date: {row['date']}\n")
        f.write(f"Label: {'Spam' if row['label'] == 1 else 'Ham'}\n")
        f.write(f"Preview: {row['text_preview']}\n")
        f.write("-" * 60 + "\n")

print(f"\n✅ Files saved:\nCSV: {csv_path}\nTXT: {txt_path}")

# === Minimal CSV for ML Model Training ===
ml_data = all_emails[['from', 'subject', 'text', 'label']]
ml_csv_path = "emails_for_ml.csv"
ml_data.to_csv(ml_csv_path, index=False)
print(f"✅ ML training data saved to: {ml_csv_path}")
