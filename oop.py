class Person:

    def __init__ (self, name, age, height, weight):
        self.name = name
        self.age = age
        self.height = height
        self.weight = weight

    def __str__(self):
        return "Name: {}, Age: {}, Height: {}, Weight: {}".format(self.name, self.age, self.height, self.weight)
    
    def get_older(years):
        self.age += years


class Worker(Person):

    def __init__(self, name, age, height, weight, hourly, hours):
        super(Worker, self).__init__(name, age, height, weight)
        self.hourly = hourly
        self.hours = hours

    def __str__(self):
        text = super(Worker, self).__str__()
        text += ", Hourly: {}, Hours: {}, Pay: {}".format(self.hourly, self.hours, self.pay())
        return text

    def pay(self):
        return self.hourly * self.hours

worker1 = Worker("Aidan",22,184,170,23.17,70)
print(worker1)
