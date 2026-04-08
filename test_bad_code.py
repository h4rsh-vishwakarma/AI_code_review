"""Test file with intentional security and code quality issues."""

import os
import subprocess

# Hardcoded secret — should be caught
API_KEY = "sk-proj-abc123secretkey456xyz"
DATABASE_PASSWORD = "admin@12345"


def get_user(user_input):
    """SQL injection vulnerability."""
    query = f"SELECT * FROM users WHERE id = {user_input}"
    return query


def run_command(user_input):
    """Command injection via shell=True."""
    result = subprocess.run(f"echo {user_input}", shell=True, capture_output=True)
    return result.stdout


def process_data(data):
    """Using eval on untrusted input."""
    result = eval(data)
    return result


def fetch_all_users(db):
    """N+1 query problem."""
    users = db.execute("SELECT * FROM users").fetchall()
    for user in users:
        orders = db.execute(f"SELECT * FROM orders WHERE user_id = {user['id']}").fetchall()
        user["orders"] = orders
    return users


def calculate(items):
    """O(n^2) when O(n) is possible."""
    result = []
    for item in items:
        if item not in result:
            result.append(item)
    return result


def login(username, password):
    """Multiple issues: prints sensitive data, no hashing."""
    print(f"Login attempt: {username} / {password}")
    if password == DATABASE_PASSWORD:
        return True
    return False


# TODO fix this later
def broken_error_handling():
    """Swallowed exception."""
    try:
        data = open("important_file.txt").read()
        return data
    except:
        pass


console_log_test = "console.log('debug info')"
