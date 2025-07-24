def get_name_by_caller_id(caller_id):
    records = [
    {"Name": "Teena Soni", "CallerID": "1001"},
    {"Name": "Tushar Thakare", "CallerID": "1002"},
    {"Name": "Shourya Jain", "CallerID": "1003"},
    {"Name": "Soumya Saxena", "CallerID": "1003"},
    {"Name": "Tejdeep singh", "CallerID": "1004"},
    {"Name": "Nikhil Vaidya", "CallerID": "1005"},
    {"Name": "Shrikant Khupat", "CallerID": "1006"},
    {"Name": "Manoj Kabule", "CallerID": "1007"},
    {"Name": "Nasreen Mohammad", "CallerID": "1008"},
    {"Name": "Pramod Moondra", "CallerID": "1009"}
    ]
    for entry in records:
        if entry.get("CallerID") == caller_id:
            return entry.get("Name")
    return None

def get_status(application):
    records = [
    {"Application": "Wait", "Status": "Waiting In Queue"},
    {"Application": "BackGround", "Status": "On IVR Options"},
    {"Application": "Record", "Status": "Recorded"}
    ]
    for entry in records:
        if entry.get("Application") == application:
            return entry.get("Status")
    return None
