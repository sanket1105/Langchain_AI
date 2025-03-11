print("The inserted records are")
data = cursor.execute("""Select * from STUDENT""")
for row in data:
    print(row)
