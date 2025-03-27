import requests
import pandas as pd
import os

def generate_access_token(client_id, client_secret, refresh_token):
    params = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    try:
        response = requests.post("https://accounts.zoho.com/oauth/v2/token", data=params)
        response.raise_for_status()
        return response.json().get("access_token")
    except requests.RequestException as e:
        print(f"Error refreshing access token: {e}")
        return None


def save_csv_from_response(response, report_name, client_name):
    output_dir = "csvdata"
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, f"{report_name}_details_{client_name}.csv")

    with open(csv_filename, "wb") as f:
        f.write(response.content)  # Save CSV directly from response

    print(f"CSV file saved: {csv_filename}")

def fetch_report(report_name, url_template, client_data, date_filter):
    access_token = generate_access_token(
        client_data["CLIENT_ID"], 
        client_data["CLIENT_SECRET"], 
        client_data["REFRESH_TOKEN"]
    )
    if not access_token:
        print("Failed to obtain access token.")
        return
    
    url = url_template.format(ORG_ID=client_data["ORG_ID"], value=date_filter)
    headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        # Save the response content as CSV directly
        save_csv_from_response(response, report_name, client_data["Client"])
    except requests.RequestException as e:
        print(f"Error fetching {report_name}: {e}")


def fetch_all_reports(date_filter):
    CREDENTIALS = [
        {
            "CLIENT_ID": "1000.35RGAMWJI3DP9TQLIYM6S2P3L9P0DX",
            "CLIENT_SECRET": "5273f0c15951cc6141a0e208d8f55d84809c13b03a",
            "REFRESH_TOKEN": "1000.73bb1e5ba1420722376937d7b90fac0e.f82401fb0ad44a23f4ccc97a447b2d57",
            "ORG_ID": "642273083",
            "Client": "smcs"
        },
        {
            "CLIENT_ID": "1000.F9IOE3YYE2DAIRWJTK3D4FRGXMS5TB",
            "CLIENT_SECRET": "f8046687a3aae7ef2617be818eca9bf7f4d18bf68c",
            "REFRESH_TOKEN": "1000.1ca7f0567e31f01a27f482bf90c6fc26.a9977be797a3e3ee5436c8982e530a4a",
            "ORG_ID": "693033731",
            "Client": "nvb"
        }
    ]
    
    REPORTS = {
        "invoice_aging": {
            "url": "https://www.zohoapis.com/books/v3/reports/aragingdetails?accept=csv&organization_id={ORG_ID}&page=1&per_page=100000&sort_order=A&sort_column=date&interval_range=15&number_of_columns=4&interval_type=days&group_by=none&filter_by=InvoiceDueDate.{value}&entity_list=invoice&is_new_flow=true&response_option=1",
            "key": "invoiceaging"
        },
        "customer_balance_summary": {
            "url": "https://www.zohoapis.com/books/v3/reports/customerbalancesummary?accept=csv&organization_id={ORG_ID}&page=1&per_page=100000&sort_order=A&filter_by=TransactionDate.{value}&select_columns=%5B%7B%22field%22%3A%22customer_name%22%2C%22group%22%3A%22report%22%7D%2C%7B%22field%22%3A%22invoiced_amount%22%2C%22group%22%3A%22report%22%7D%2C%7B%22field%22%3A%22amount_received%22%2C%22group%22%3A%22report%22%7D%2C%7B%22field%22%3A%22closing_balance%22%2C%22group%22%3A%22report%22%7D%5D&is_for_date_range=true&usestate=true&group_by=%5B%7B%22field%22%3A%22none%22%2C%22group%22%3A%22report%22%7D%5D&sort_column=customer_name&response_option=0",
            "key": "customerbalancesummary"
        }
    }
    
    for client_data in CREDENTIALS:
        for report_name, report_data in REPORTS.items():
            fetch_report(report_name, report_data["url"], client_data, date_filter)