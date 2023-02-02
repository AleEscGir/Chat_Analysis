import json
from pathlib import Path
from sys import path




def take_data_from_whatsapp(data, text):
    for line in text:
        for i in range(len(line)):
            if line[i] == "]":
                line = line[i+1:len(line)]
                break
        for i in range(len(line)):
            if line[i] == ":":
                user = line[1:i]
                message = line[i+2:len(line) - 1]
                if user in data:
                    data[user]["text"].append(message)
                else:
                    data[user] = {}
                    data[user]["text"] = [message]
                break


def take_data_from_telegram(data, text):
    user = ""
    for line in text:
        line = line[:len(line) - 1]
        if line == '':
            continue
        if line[len(line) - 1] == "]":
            if line[0] == "[":
                continue
            else:
                for i in range(len(line)):
                    if line[i] == ",":
                        user = line[:i]
                        break
        else:
            if user in data:
                data[user]["text"].append(line)
            else:
                data[user] = {}
                data[user]["text"] = [line]
            



social_red = "telegram"

path = Path(__file__).parent.absolute()

with open('data/data.json', 'r+') as file, open(social_red + '/text_4.txt', 'r+') as text:
    data = json.load(file)

    
    if social_red == "whatsapp":
        take_data_from_whatsapp(data, text)
    if social_red == "telegram":
        take_data_from_telegram(data, text)
    


    file.seek(0)
    json.dump(data, file, indent=4)
    file.truncate()


