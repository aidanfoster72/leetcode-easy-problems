import queue

q = queue.PriorityQueue()

q.put((2, "Joe"))
q.put((1, "Steve"))
q.put((0, "Bob"))
q.put((100, 100))
q.put((69, False))

while not q.empty():
    print(q.get()[1])



# q = queue.LifoQueue()

# nums = [1,2,3,4,5,6,7,8,9,10]

# for x in nums:
#     q.put(x)

# print(q.get())
# print(q.get())
# print(q.get())
# print(q.get())




# q = queue.Queue()

# numbers = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# for number in numbers:
#     q.put(number)

# print(q.get())
# print(q.get())
# print(q.get())
# print(q.get())
