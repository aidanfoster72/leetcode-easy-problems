class Solution(object):
    def isPalindrome(self, x):
        x = str(x)
        if x == x[::-1]:
            return True
        else:
            return False

# print(Solution().isPalindrome(2022))




class Solution(object):
    def longestCommonPrefix(self, strs):
        if not strs:
            return ""
            
        prefix = strs[0]
        
        for word in strs[1:]:
            while word.find(prefix) != 0: 
                prefix = prefix[:-1]      
                if prefix == "":
                    return ""
        
        return prefix

# class Solution(object):
#     def longestCommonPrefix(self, strs):
#         first = strs[0]
#         for i in range(len(first)):
#             c = first[i]
#             for s in strs[1:]:
#                 if i == len(s) or s[i] != c:
#                     return first[:i]
#         return first

# print(Solution().longestCommonPrefix(["flower","flow","flight"]))



class Solution(object):
    def isValid(self, s):
        stack = []
        for char in s:
            if char in "({[":
                stack.append(char)
            elif char in ")}]":
                if not stack:
                    return False
                top = stack.pop()
                if (char == ")" and top != "(") or (char == "}" and top != "{") or (char == "]" and top != "["):
                    return False
        return not stack


# print(Solution().isValid("()()()"))      



class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution(object):
    def mergeTwoLists(self, list1, list2):
        dummy = node = ListNode()

        while list1 and list2:
            if list1.val < list2.val:
                node.next = list1
                list1 = list1.next
            else:
                node.next = list2
                list2 = list2.next
            node = node.next
        node.next = list1 or list2
        return dummy.next

def build_linked_list(lst):
    dummy = ListNode()
    curr = dummy
    for val in lst:
        curr.next = ListNode(val)
        curr = curr.next
    return dummy.next

def print_linked_list(node):
    result = []
    while node:
        result.append(node.val)
        node = node.next
    print(result)


# l1 = build_linked_list([1,2,4])
# l2 = build_linked_list([1,3,4])
# merged = Solution().mergeTwoLists(l1, l2)
# print_linked_list(merged)


class Solution(object):
    def removeDuplicates(self, nums):
        if not nums:
            return 0
        write_index = 1
        for read_index in range(1, len(nums)):
            if nums[read_index] != nums[read_index - 1]:
                nums[write_index] = nums[read_index]
                write_index += 1
        return write_index

# print(Solution().removeDuplicates([1,1,2,2,3,3,4,4,5,5]))


class Solution(object):
    def removeElement(self, nums, val):
        write_index = 0
        for read_index in range(len(nums)):
            if nums[read_index] != val:
                nums[write_index] = nums[read_index]
                write_index += 1
        return write_index
        
#print(Solution().removeElement([3,2,3,0,3,4,5,3],3))



class Solution(object):
    def strStr(self, haystack, needle):
        if needle in haystack:
            return haystack.index(needle)
        else:
            return -1
        
# print(Solution().strStr("Hello, World!","o"))



class Solution(object):
    def searchInsert(self, nums, target):
        for i in range(len(nums)):
            if nums[i] == target:
                return i
            elif nums[i] > target:
                return i
        return len(nums)

#print(Solution().searchInsert([1,3,5,6],2))




class Solution(object):
    def lengthOfLastWord(self, s):
        return len(s.split()[-1])
    
#print(Solution().lengthOfLastWord("Hello World"))



class Solution(object):
    def plusOne(self, digits):
        digits = [str(i) for i in digits]
        digits = int("".join(digits)) + 1
        digits = [int(i) for i in str(digits)]
        return digits
        
# print(Solution().plusOne([9]))



class Solution(object):
    def addBinary(self, a, b):
        return bin(int(a, 2) + int(b, 2))[2:]
    
# print(Solution().addBinary("11","1001"))




import math

class Solution:
    def mySqrt(self, x):
        y = int(math.sqrt(x))
        return y


#print(Solution().mySqrt(1))



class Solution(object):
    def climbStairs(self, n):
        if n == 1:
            return 1
        if n == 2:
            return 2
        return self.climbStairs(n-1) + self.climbStairs(n-2)

# print(Solution().climbStairs(44))
        



class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution(object):
    def deleteDuplicates(self, head):
        current = head

        while current and current.next:
            if current.val == current.next.val:
                current.next = current.next.next
            else:
                current = current.next

        return head

# print(Solution().deleteDuplicates([1,1,2,3,3]))





class ListNode(object):

    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None


class LRUCache(object):

    def __init__(self, capacity):
        self.head = ListNode(-1, -1)
        self.tail = ListNode(-1, -1)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.size = 0
        self.capacity = capacity
        self.nodeMap = {}


    def get(self, key):
        if key in self.nodeMap:
            node = self.nodeMap[key]
            self.remove(node)
            self.add(node)
            return node.val
        return -1 


    def put(self, key, value):
        if key in self.nodeMap:
            node = self.nodeMap[key]
            node.val = value
            self.remove(node)
            self.add(node)
        else:
            newNode = ListNode(key, value)
            self.nodeMap[key] = newNode
            self.add(newNode)
            self.size += 1
            if self.size > self.capacity:
                nodeToRemove = self.tail.prev
                del self.nodeMap[nodeToRemove.key]
                self.remove(nodeToRemove)
                self.size -= 1


    def add(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

# obj = LRUCache(2)
# obj.put(1,1)
# obj.put(2,2)
# print(obj.get(1))
# obj.put(3,3)
# print(obj.get(2))
# obj.put(4,4)
# print(obj.get(1))
# print(obj.get(3))
# print(obj.get(4))







class UndergroundSystem(object):

    def __init__(self):
        self.in_map = {} 
        self.stats = {}  

    def checkIn(self, id, stationName, t):
        self.in_map[id] = (stationName, t)

    def checkOut(self, id, stationName, t):
        start_station, t_in = self.in_map.pop(id)
        key = (start_station, stationName)
        total, cnt = self.stats.get(key, (0, 0))
        self.stats[key] = (total + (t - t_in), cnt + 1)

    def getAverageTime(self, startStation, endStation):
        total, cnt = self.stats[(startStation, endStation)]
        return float(total) / cnt


# obj = UndergroundSystem()
# obj.checkIn(45,"Leyton",3)
# obj.checkIn(32,"Paradise",8)
# obj.checkIn(27,"Leyton",10)
# obj.checkOut(45,"Waterloo",15)
# obj.checkOut(27,"Waterloo",20)
# obj.checkOut(32,"Cambridge",22)
# print(obj.getAverageTime("Paradise","Cambridge"))
# print(obj.getAverageTime("Leyton","Waterloo"))
# obj.checkIn(10,"Leyton",24)
# print(obj.getAverageTime("Leyton","Waterloo"))

