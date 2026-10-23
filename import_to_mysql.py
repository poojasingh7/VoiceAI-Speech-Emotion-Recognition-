import pymysql
import os
import sys

sql_file = "voice_analysis_workbench.sql"
if not os.path.exists(sql_file):
    print("voice_analysis_workbench.sql not found.")
    exit(1)

with open(sql_file, "r", encoding="utf-8") as f:
    sql_content = f.read()

# Try common passwords or take command line argument
passwords_to_try = []
if len(sys.argv) > 1:
    passwords_to_try.append(sys.argv[1])
else:
    passwords_to_try.extend(["", "root", "password", "123456", "1234", "admin", "mysql"])

connected = False
for pwd in passwords_to_try:
    try:
        conn = pymysql.connect(
            host="localhost",
            user="root",
            password=pwd,
            port=3306,
            client_flag=pymysql.constants.CLIENT.MULTI_STATEMENTS
        )
        print(f"Connected to MySQL successfully with password: '{pwd}'")
        connected = True
        with conn.cursor() as cursor:
            cursor.execute(sql_content)
        conn.commit()
        print("Imported all tables and data into MySQL database 'voice_analysis' successfully!")
        conn.close()
        break
    except pymysql.err.OperationalError as e:
        continue
    except Exception as e:
        print("Execution notice:", e)
        break

if not connected:
    print("Could not auto-connect with default passwords.")
    print("You can run: python import_to_mysql.py <YOUR_MYSQL_PASSWORD>")
    print("OR open 'voice_analysis_workbench.sql' directly in MySQL Workbench and click the ⚡ Execute button!")
