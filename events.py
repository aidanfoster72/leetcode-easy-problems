import threading
import time

path = "text.txt"
text = ""

def readFile():
    global path, text
    while True:
        with open("text.txt", "r") as f:
            text = f.read()
        time.sleep(3)

def printloop():
    for x in range(30):
        print(text)
        time.sleep(1)


t1 = threading.Thread(target=readFile, daemon=True)
t2 = threading.Thread(target=printloop)

t1.start()
t2.start()








# event = threading.Event()

# def myfunc():
#     print("Waiting for event to trigger")
#     event.wait()
#     print("playing now...")


# t1 = threading.Thread(target=myfunc)
# t1.start()

# x = input("Do you want to trigger the event? (y/n) ")
# if x == "y":
#     event.set()

