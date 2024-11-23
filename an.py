import csv

list = [0, 0, 0, 0, 0, 0]
third = 0
total = 0

with open('daily.csv', 'r', newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:

        list.append(float(row["change"]))

        list.pop(0)

        if list[0] < 0 and list[1] > 0 and list[2] > 0 and list[3] > 0 and list[4] > 0:
            total += 1
            if list[5] > 0:
                print("buy")
                third += 1
            else:
                print("no")


print("good", third)
print("total", total)
print(third/total)
