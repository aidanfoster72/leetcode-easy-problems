import threading
import time

semaphore = threading.BoundedSemaphore(value=5)

def access(thread_number):
    print("{} is trying to access!".format(thread_number))
    semaphore.acquire()
    print("{} was granted access".format(thread_number))
    time.sleep(10)
    print("{} is now releasing!".format(thread_number))
    semaphore.release()


for thread_number in range(1,11):
    t = threading.Thread(target=access, args=(thread_number,))
    t.start()
    time.sleep(1)





# x = 8192

# lock = threading.Lock()

# def double():
#     global x, lock
#     lock.acquire()
#     while x < 16384:
#         x *= 2
#         print(x)
#         time.sleep(1)
#     print("At max")
#     lock.release()


# def halve():
#     global x, lock
#     lock.acquire()
#     while x > 1:
#         x /= 2
#         print(x)
#         time.sleep(1)
#     print("At min")
#     lock.release()



# t1 = threading.Thread(target=halve)
# t2 = threading.Thread(target=double)

# t1.start()
# t2.start()


# def func1():
#     for x in range(1000):
#         print("one")

# def func2():
#     for x in range(1000):
#         print("two")


# t1 = threading.Thread(target=func1)
# t2 = threading.Thread(target=func2)

# t1.start()
# t2.start()


# def hello():
#     for i in range(50):
#         print("Hello")

# t1 = threading.Thread(target=hello)
# t1.start()

# t1.join()

# print("all done")




