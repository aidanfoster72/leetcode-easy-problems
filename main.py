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

            
# print(Solution().isMatch("aa", "a"))  
# print(Solution().isMatch("aa", "a*"))
# print(Solution().isMatch("ab", ".*"))
# print(Solution().isMatch("aab", "c*a*b"))
# print(Solution().isMatch("mississippi", "mis*is*p*."))




# 11. container with most water



# 18. 4 sum

class Solution(object):
    def fourSum(self, nums, target):
        nums.sort()
        result = []
        n = len(nums)

        for i in range(n - 3):
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            for j in range(i + 1, n - 2):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue

                left, right = j + 1, n - 1
                while left < right:
                    current_sum = nums[i] + nums[j] + nums[left] + nums[right]
                    if current_sum == target:
                        result.append([nums[i], nums[j], nums[left], nums[right]])
                        while left < right and nums[left] == nums[left + 1]:
                            left += 1
                        while left < right and nums[right] == nums[right - 1]:
                            right -= 1
                        left += 1
                        right -= 1
                    elif current_sum < target:
                        left += 1
                    else:
                        right -= 1

        return result

# print(Solution().fourSum([1,0,-1,0,-2,2], 0))
# print(Solution().fourSum([2,2,2,2,2], 8))



# 19. remove nth node from end of list

class Solution(object):
    def removeNthFromEnd(self, head, n):
        dummy = ListNode(0)
        dummy.next = head
        first = dummy
        second = dummy
        for i in range(n + 1):
            first = first.next
        while first:
            first = first.next
            second = second.next
        second.next = second.next.next
        return dummy.next



def build_linked_list(lst):
    dummy = ListNode(0)
    current = dummy
    for val in lst:
        current.next = ListNode(val)
        current = current.next
    return dummy.next

def print_linked_list(node):
    result = []
    while node:
        result.append(node.val)
        node = node.next
    print(result)

# l1 = build_linked_list([1,2,3,4,5])
# print_linked_list(l1)
# print("--------------------------------")
# print_linked_list(Solution().removeNthFromEnd(l1, 2))










class Solution(object):
    def FizzBuzz(self, n):
        result = []
        for i in range(1, n+1):
            if i % 3 == 0 and i % 5 == 0:
                result.append("FizzBuzz")
            elif i % 3 == 0:
                result.append("Fizz")
            elif i % 5 == 0:
                result.append("Buzz")
            else:
                result.append(str(i))
        return result

# print(Solution().FizzBuzz(100))


class Solution(object):

    def removeIslands(self, matrix):
        rows, cols = len(matrix), len(matrix[0])
        visited = [[0] * cols for _ in range(rows)]

        for i in range(rows):
                if matrix[i][0] == 1 and visited[i][0] == 0:
                    self.dfs(matrix, i, 0, visited)
        for i in range(rows):
                if matrix[i][cols-1] == 1 and visited[i][cols-1] == 0:
                    self.dfs(matrix, i, cols-1, visited)
        for i in range(cols):
                if matrix[0][i] == 1 and visited[0][i] == 0:
                    self.dfs(matrix, 0, i, visited)
        for i in range(cols):
                if matrix[rows-1][i] == 1 and visited[rows-1][i] == 0:
                    self.dfs(matrix, rows-1, i, visited)

        for i in range(rows):
            print(visited[i])
        return visited


    def dfs(self, matrix, x, y, visited):
        rows, cols = len(matrix), len(matrix[0])

        if x < 0 or x >= rows or y < 0 or y >=cols:
            return

        if matrix[x][y] == 0 or visited[x][y] == 1:
            return
        
        visited[x][y] = 1

        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        for dx, dy in directions:
            self.dfs(matrix, x + dx, y + dy, visited)
        

# print(Solution().removeIslands([[1,1,1,1,1,1],[1,0,0,0,0,1],[1,0,1,1,0,1],[1,0,1,1,0,1],[1,0,1,0,0,1],[1,1,1,1,1,1]]))





class Solution(object):
    def generate(self, result, current, left, right, n):
        if len(current) == 2 * n:
            result.append(current)
            return
        if left < n:
            self.generate(result, current + "(", left + 1, right, n)        
        if right < left:
            self.generate(result, current + ")", left, right + 1, n)

    def generateParenthesis(self, n):
        result = []
        self.generate(result, "", 0, 0, n)
        return result

# print(Solution().generateParenthesis(3))


# 23. merge k sorted lists

class Solution(object):
    def mergeKLists(self, lists):
        if not lists:
            return None
        while len(lists) > 1:
            merged = []
            for i in range(0, len(lists), 2):
                l1 = lists[i]
                l2 = lists[i+1] if i+1 < len(lists) else None
                merged.append(self._mergeTwo(l1, l2))
            lists = merged
        return lists[0]

    def _mergeTwo(self, a, b):
        dummy = tail = ListNode(0)
        while a and b:
            if a.val <= b.val:
                tail.next, a = a, a.next
            else:
                tail.next, b = b, b.next
            tail = tail.next
        tail.next = a or b
        return dummy.next
    




# 498 Diagonal Traverse

class Solution(object):
    def findDiagonalOrder(self, mat):
        if not mat or not mat[0]:
            return []

        m, n = len(mat), len(mat[0])
        result = []
        row, col = 0, 0
        direction = 1  # 1 = up-right, -1 = down-left

        for _ in range(m * n):
            result.append(mat[row][col])

            # Moving up-right
            if direction == 1:
                if col == n - 1:
                    row += 1
                    direction = -1
                elif row == 0:
                    col += 1
                    direction = -1
                else:
                    row -= 1
                    col += 1

            # Moving down-left
            else:
                if row == m - 1:
                    col += 1
                    direction = 1
                elif col == 0:
                    row += 1
                    direction = 1
                else:
                    row += 1
                    col -= 1

        return result


# Test cases
# print(Solution().findDiagonalOrder([[1,2,3],[4,5,6],[7,8,9]]))  # [1,2,4,7,5,3,6,8,9]
# print(Solution().findDiagonalOrder([[1,2],[3,4]]))              # [1,2,3,4]


class Solution(object):
    def longestValidParentheses(self, s):
        best = 0
        stack = [-1]  # base for length calc
        for i, ch in enumerate(s):
            if ch == '(':
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)       # reset base after unmatched ')'
                else:
                    best = max(best, i - stack[-1])
        return best

# print(Solution().longestValidParentheses("(((("))


class Solution(object):
    def search(self, nums, target):
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) // 2
            if nums[left] <= nums[mid]:
                if nums[left] <= target <= nums[mid]:
                    if nums[left] == target: return left
                    if nums[right] == target: return right
                    if nums[mid] == target: return mid
                    right = mid - 1
                else:
                    left = mid + 1
            elif nums[mid] < nums[right]:
                if nums[mid] <= target <= nums[right]:
                    if nums[left] == target: return left
                    if nums[right] == target: return right
                    if nums[mid] == target: return mid
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                break
        if (left == right) and (nums[left] == target):
            return left
        return -1

# print(Solution().search([4,5,6,7,0,1,2], 8))
        


class Solution(object):
    def searchRange(self, nums, target):
        max = 0
        min = 0
        minflag = True
        validflag = False
        counter = 0
        output = []
        for i in nums:
            if i > target:
                break
            if target == i:
                validflag = True
                if minflag == True:
                    min = counter
                    minflag = False
                max = counter
            counter += 1
        if validflag == True:
            output.append(min)
            output.append(max)
            return output
        else:
            return [-1,-1]
        


class Solution(object):
    def isValidSudoku(self, board):
        rows = [set() for _ in range(9)]
        cols = [set() for _ in range(9)]
        boxes = [set() for _ in range(9)]  # box index = (r//3)*3 + (c//3)

        for r in range(9):
            for c in range(9):
                val = board[r][c]
                if val == '.':
                    continue
                b = (r // 3) * 3 + (c // 3)
                if val in rows[r] or val in cols[c] or val in boxes[b]:
                    return False
                rows[r].add(val)
                cols[c].add(val)
                boxes[b].add(val)
        return True


class Solution(object):
    def solveSudoku(self, board):
        # Bit masks for digits 1..9 used in each row/col/box
        rows = [0] * 9
        cols = [0] * 9
        boxes = [0] * 9
        empties = []

        def box_id(r, c): return (r // 3) * 3 + (c // 3)
        def bit(d): return 1 << (ord(d) - ord('1'))  # '1'->1<<0, ..., '9'->1<<8

        # Initialize masks and collect empty cells
        for r in range(9):
            for c in range(9):
                ch = board[r][c]
                if ch == '.':
                    empties.append((r, c))
                else:
                    b = box_id(r, c)
                    mask = bit(ch)
                    rows[r] |= mask
                    cols[c] |= mask
                    boxes[b] |= mask

        # Precompute all digits as bitset (bits 0..8)
        FULL = (1 << 9) - 1

        # Choose-next-variable heuristic: pick the cell with fewest candidates
        def pick_next():
            best_i = -1
            best_opts = FULL + 1
            for i, (r, c) in enumerate(empties):
                b = box_id(r, c)
                used = rows[r] | cols[c] | boxes[b]
                opts = FULL & ~used
                cnt = bin(opts).count("1")  # count set bits (candidates)
                if cnt < best_opts:
                    best_opts = cnt
                    best_i = i
                    if cnt == 1:
                        break
            return best_i


        def backtrack():
            if not empties:
                return True
            i = pick_next()
            r, c = empties[i]
            # swap picked with last to pop efficiently
            empties[i], empties[-1] = empties[-1], empties[i]
            r, c = empties.pop()

            b = box_id(r, c)
            used = rows[r] | cols[c] | boxes[b]
            opts = FULL & ~used  # bits for possible digits

            while opts:
                lsb = opts & -opts
                d_idx = (lsb.bit_length() - 1)  # 0..8
                ch = chr(ord('1') + d_idx)

                # place
                board[r][c] = ch
                rows[r] |= lsb
                cols[c] |= lsb
                boxes[b] |= lsb

                if backtrack():
                    return True

                # undo
                board[r][c] = '.'
                rows[r] ^= lsb
                cols[c] ^= lsb
                boxes[b] ^= lsb

                opts ^= lsb  # try next digit

            # restore the empty slot and fail this branch
            empties.append((r, c))
            return False

        backtrack()

            

        

class Solution(object):
    def countAndSay(self, n):
        word = "1"
        for _ in range(n - 1):  # n=1 means "1" directly
            word = self.rle(word)
        return word

    def rle(self, s):
        result = ""
        count = 1
        
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                count += 1
            else:
                result += str(count) + s[i - 1]
                count = 1
        
        result += str(count) + s[-1]  # append last group
        return result

# print(Solution().countAndSay(1))  # Output: "1"
# print(Solution().countAndSay(4))  # Output: "1211"




class Solution(object):
    def nonrep(self, s):
        map = {}
        for i in s:
            if i not in map:
                map[i] = 1
            else: 
                map[i] += 1
        for i in map:
            if map[i] == 1:
                return i
        return "_"
    
# print(Solution().nonrep("adbacaba"))


class Solution(object):
    def combinationSum(self, candidates, target):
        candidates.sort()  # enables pruning
        res = []
        path = []

        def dfs(start, remain):
            if remain == 0:
                res.append(list(path))
                return
            for i in range(start, len(candidates)):
                x = candidates[i]
                if x > remain:
                    break  # prune
                path.append(x)
                # i (not i+1) because we can reuse the same number
                dfs(i, remain - x)
                path.pop()

            # done exploring this level

        dfs(0, target)
        return res

# print(Solution().combinationSum([2,3,6,7], 7))


class Solution(object):
    def areaOfMaxDiagonal(self, dimensions):
        w, h = max(dimensions, key=lambda x: (x[0]*x[0] + x[1]*x[1], x[0]*x[1]))
        return w*h


        
# print(Solution().areaOfMaxDiagonal([[9,3],[8,6],[5,9]]))



class Solution(object):
    def combinationSum2(self, candidates, target):
        candidates.sort()        # sort so we can prune and skip duplicates
        res = []
        path = []

        def dfs(start, remain):
            if remain == 0:
                res.append(list(path))
                return

            for i in range(start, len(candidates)):
                # Skip duplicates at the same tree level
                if i > start and candidates[i] == candidates[i - 1]:
                    continue

                x = candidates[i]
                if x > remain:
                    break  # further numbers are larger (sorted), so stop

                path.append(x)
                dfs(i + 1, remain - x)  # i+1 because each number can be used once
                path.pop()

        dfs(0, target)
        return res



class Solution(object):
    def firstMissingPositive(self, nums):
        nums = set(nums)
        if 1 not in nums:
            return 1
        min = 1
        while min in nums:
            min +=1
        return min
        
# print(Solution().firstMissingPositive([1,2,0]))
# print(Solution().firstMissingPositive([7,8,9,11,12]))
# print(Solution().firstMissingPositive([3,4,-1,1]))



class Solution(object):
    def trap(self, height):
        h = len(height)
        left, right = 0, h-1
        leftMax, rightMax = 0, 0
        water = 0

        while left < right:
            if height[left] < height[right]:
                if height[left] >= leftMax:
                    leftMax = height[left]
                else:
                    water += leftMax - height[left]
                left += 1
            else:
                if height[right] >= rightMax:
                    rightMax = height[right]
                else: 
                    water += rightMax - height[right]
                right -= 1

        return water

            
# print(Solution().trap([0,1,0,2,1,0,1,3,2,1,2,1]))
# print(Solution().trap([4,2,0,3,2,5]))  




class Solution(object):
    def jump(self, nums):
        n = len(nums)
        if n <= 1:
            return 0

        jumps = 0
        cur_end = 0      # end of current jump's reach
        farthest = 0     # farthest reach seen so far

        # We don't need to process the last index
        for i in range(n - 1):
            if i + nums[i] > farthest:
                farthest = i + nums[i]
            if i == cur_end:        # time to "commit" a jump
                jumps += 1
                cur_end = farthest
                if cur_end >= n - 1:
                    break
        return jumps
    
# print(Solution().jump([2,6,4,1,1,1,1,1,1]))


class Solution(object):
    def permute(self, nums):
        res, path = [], []
        used = [False] * len(nums)

        def dfs():
            if len(path) == len(nums):
                res.append(path[:])
                return
            for i, x in enumerate(nums):
                if not used[i]:
                    used[i] = True
                    path.append(x)
                    dfs()
                    path.pop()
                    used[i] = False

        dfs()
        return res
    
# print(Solution().permute([1,2,3]))


class Solution(object):
    def permuteUnique(self, nums):
        res, path = [], []
        used = [False] * len(nums)

        def dfs():
            if len(path) == len(nums):
                res.append(path[:])
                return
            for i in range(len(nums)):
                if used[i]:
                    continue
                if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                    continue
                used[i] = True
                path.append(nums[i])
                dfs()
                path.pop()
                used[i] = False

        dfs()
        return res
    
print(Solution().permuteUnique([1,1,3]))