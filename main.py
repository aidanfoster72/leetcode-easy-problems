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




