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




# 2. add two numbers

class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        dummy = ListNode(0)
        current = dummy
        carry = 0
        while l1 or l2 or carry:
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0
            carry, total = divmod(val1 + val2 + carry, 10)
            current.next = ListNode(total)
            current = current.next
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        return dummy.next


# l1 = build_linked_list([2, 4, 3])
# l2 = build_linked_list([5, 6, 4])
# res1 = Solution().addTwoNumbers(l1, l2)
# print_linked_list(res1)

# l3 = build_linked_list([9, 9, 9, 9, 9, 9, 9])
# l4 = build_linked_list([9, 9, 9, 9])
# res2 = Solution().addTwoNumbers(l3, l4)
# print_linked_list(res2)

# 3. longest substring without repeating characters

class Solution(object):
    def lengthOfLongestSubstring(self, s):
        char_set = set()
        left = 0
        max_length = 0
        for right in range(len(s)):
            while s[right] in char_set:
                char_set.remove(s[left])
                left += 1
            char_set.add(s[right])
            max_length = max(max_length, right - left + 1)
        return max_length

# print(Solution().lengthOfLongestSubstring("abcabcbb"))
# print(Solution().lengthOfLongestSubstring("bbbbb"))
# print(Solution().lengthOfLongestSubstring("pwwkew"))
# print(Solution().lengthOfLongestSubstring(""))
# print(Solution().lengthOfLongestSubstring(" "))
# print(Solution().lengthOfLongestSubstring("au"))
# print(Solution().lengthOfLongestSubstring("dvdf"))
# print(Solution().lengthOfLongestSubstring("abba"))
# print(Solution().lengthOfLongestSubstring("abcabcbb"))

# 4. median of two sorted arrays

# class Solution(object):
#     def findMedianSortedArrays(self, nums1, nums2):
#         nums1.extend(nums2)
#         nums1.sort()
#         print(nums1)
#         print(len(nums1))
#         if len(nums1) % 2 == 0:
#             return float(nums1[len(nums1)//2 - 1] + nums1[len(nums1)//2]) / 2
#         else:
#             return float(nums1[len(nums1)//2])


class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        # Ensure nums1 is the smaller array
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1

        m, n = len(nums1), len(nums2)
        imin, imax, half_len = 0, m, (m + n + 1) // 2

        while imin <= imax:
            i = (imin + imax) // 2
            j = half_len - i

            if i < m and nums2[j-1] > nums1[i]:
                # i is too small, must increase it
                imin = i + 1
            elif i > 0 and nums1[i-1] > nums2[j]:
                # i is too big, must decrease it
                imax = i - 1
            else:
                # i is perfect
                if i == 0: max_of_left = nums2[j-1]
                elif j == 0: max_of_left = nums1[i-1]
                else: max_of_left = max(nums1[i-1], nums2[j-1])

                if (m + n) % 2 == 1:
                    return max_of_left  # Odd length

                if i == m: min_of_right = nums2[j]
                elif j == n: min_of_right = nums1[i]
                else: min_of_right = min(nums1[i], nums2[j])

                return (max_of_left + min_of_right) / 2.0
        
# print(Solution().findMedianSortedArrays([1,2],[3,4]))


# 5. longest palindromic substring

class Solution(object):
    def longestPalindrome(self, s):
        start = 0
        end = 0
        for i in range(len(s)):
            len1 = self.expandAroundCenter(s, i, i)
            len2 = self.expandAroundCenter(s, i, i+1)
            max_len = max(len1, len2)
            if max_len > end - start:
                start = i - (max_len - 1) // 2
                end = i + max_len // 2
        return s[start:end+1]
        
    
    def expandAroundCenter(self, s, left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1

# print(Solution().longestPalindrome("babad"))
# print(Solution().longestPalindrome("cbbd"))   
# print(Solution().longestPalindrome("a"))



#  6. zigzag conversion

class Solution(object):
    def convert(self, s, numRows):
        if numRows == 1 or numRows >= len(s):
            return s
        rows = [""] * numRows
        index, step = 0, 1
        for char in s:
            rows[index] += char
            if index == 0:
                step = 1
            elif index == numRows - 1:
                step = -1
            index += step
        return "".join(rows)

# print(Solution().convert("PAYPALISHIRING", 3))
# print(Solution().convert("PAYPALISHIRING", 4))
# print(Solution().convert("A", 1))


#  7. reverse integer

class Solution(object):
    def reverse(self, x):
        x = str(x)
        if x[0] == "-":
            x = x[1:]
            x = x[::-1]
            x = "-" + x
        else:
            x = x[::-1]
        if int(x) > 2**31 - 1 or int(x) < -2**31:
            return 0
        return int(x)

# print(Solution().reverse(123))
# print(Solution().reverse(-123))
# print(Solution().reverse(120))
# print(Solution().reverse(0))
# print(Solution().reverse(1534236469))




# 8. string to integer (atoi)

class Solution(object):
    def myAtoi(self, s):
        s = s.strip()
        if not s:
            return 0
        sign = 1
        if s[0] == "-":
            sign = -1
            s = s[1:]
        elif s[0] == "+":
            s = s[1:]
        num = 0
        for char in s:
            if char.isdigit():
                num = num * 10 + int(char)
            else:
                break
        num = sign * num
        if num > 2**31 - 1:
            return 2**31 - 1
        elif num < -2**31:
            return -2**31
        return num

# print(Solution().myAtoi("42"))
# print(Solution().myAtoi("   -42"))
# print(Solution().myAtoi("4193 with words"))
# print(Solution().myAtoi("words and 987"))
# print(Solution().myAtoi("1337c0d3"))
# print(Solution().myAtoi("-91283472332"))


# 9. palindrome number

class Solution(object):
    def isPalindrome(self, x):
        x = str(x)
        if x == x[::-1]:
            return True
        else:
            return False

# print(Solution().isPalindrome(121))


# 10. regular expression matching


class Solution(object):
    def isMatch(self, s, p):
        memo = {}

        def dp(i, j):
            if (i, j) in memo:
                return memo[(i, j)]

            if j == len(p):
                return i == len(s)

            first = i < len(s) and (p[j] == s[i] or p[j] == '.')

            if j + 1 < len(p) and p[j + 1] == '*':
                ans = dp(i, j + 2) or (first and dp(i + 1, j))
            else:
                ans = first and dp(i + 1, j + 1)

            memo[(i, j)] = ans
            return ans

        return dp(0, 0)

            
print(Solution().isMatch("aa", "a"))  
print(Solution().isMatch("aa", "a*"))
print(Solution().isMatch("ab", ".*"))
print(Solution().isMatch("aab", "c*a*b"))
print(Solution().isMatch("mississippi", "mis*is*p*."))




# 11. container with most water