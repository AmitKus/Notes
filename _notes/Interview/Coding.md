
## Longest Substring Without Repeating Characters

**Example 1:**
**Input:** s = "abcabcbb"
**Output:** 3
**Explanation:** The answer is "abc", with the length of 3.

**Example 2:**
**Input:** s = "bbbbb"
**Output:** 1
**Explanation:** The answer is "b", with the length of 1.
```python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        left_ptr = 0
        char_index_map = {}
        maxlen = 0

        for right_ptr in range(len(s)):
            # If the character is already in the substring, move left_ptr
            if s[right_ptr] in char_index_map and char_index_map[s[right_ptr]] >= left_ptr:
                left_ptr = char_index_map[s[right_ptr]] + 1
            
            # Update the character's index in the map
            char_index_map[s[right_ptr]] = right_ptr
            
            # Calculate the maximum length of the current window
            maxlen = max(maxlen, right_ptr - left_ptr + 1)
        
        return maxlen
```

## Longest Substring with At Most Two Distinct Characters

Given a string `s`, return _the length of the longest_ _substring_ _that contains at most **two distinct characters**_.

**Example 1:**
**Input:** s = "eceba"
**Output:** 3
**Explanation:** The substring is "ece" which its length is 3.

**Example 2:**
**Input:** s = "ccaabbb"
**Output:** 5
**Explanation:** The substring is "aabbb" which its length is 5.


```python
class Solution:
    def lengthOfLongestSubstringTwoDistinct(self, s: str) -> int:
        # Sliding window approach
        n = len(s)
        if n <= 2:
            return n

        # Dictionary to store character frequencies
        char_count = {}
        left_ptr = 0
        maxlen = 0

        for right_ptr in range(n):
            # Add the current character to the window
            char_count[s[right_ptr]] = char_count.get(s[right_ptr], 0) + 1

            # Shrink the window if there are more than 2 distinct characters
            while len(char_count) > 2:
                char_count[s[left_ptr]] -= 1
                if char_count[s[left_ptr]] == 0:
                    del char_count[s[left_ptr]]
                left_ptr += 1

            # Update the maximum length of the substring
            maxlen = max(maxlen, right_ptr - left_ptr + 1)

        return maxlen
```

## 3Sum

**Example 1:**
**Input:** nums = [-1,0,1,2,-1,-4]
**Output:** [[-1,-1,2],[-1,0,1]]
**Explanation:** 
nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0.
The distinct triplets are [-1,0,1] and [-1,-1,2].
Notice that the order of the output and the order of the triplets does not matter.

**Example 2:**
**Input:** nums = [0,1,1]
**Output:** []
**Explanation:** The only possible triplet does not sum up to 0.

```python
class Solution(object):
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
    
        # Strategy:
        # Sort
        # Loop over each item
        # For each item do a two pointers approach: sorting was required for this

        output = set()
        nums = sorted(nums)
        for i in range(len(nums)):
                       
            target = -nums[i] # sum of other two numbers
            
            left_ptr = i + 1
            right_ptr = len(nums)-1
            while left_ptr < right_ptr:
                
                sum_lr = nums[left_ptr] + nums[right_ptr]
                if sum_lr < target:
                    left_ptr += 1
                elif sum_lr > target:
                    right_ptr -= 1
                else:
                    output.add((nums[i], nums[left_ptr], nums[right_ptr]))
                    left_ptr += 1
                    right_ptr -= 1
        return [list(o) for o in output]
```


## Multiply Strings

Solution

Given two non-negative integers `num1` and `num2` represented as strings, return the product of `num1` and `num2`, also represented as a string.

**Note:** You must not use any built-in BigInteger library or convert the inputs to integer directly.

**Example 1:**

**Input:** num1 = "2", num2 = "3"
**Output:** "6"

**Example 2:**

**Input:** num1 = "123", num2 = "456"
**Output:** "56088"


```python
class Solution(object):
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """

        if num1 == '0' or num2 == '0':
            return '0'
		## Line not required
        if len(num1) < len(num2):
            num1, num2 = num2, num1
            
        # Reverse the strings to ease multiplication
        num1 = num1[::-1]
        num2 = num2[::-1]
        
        n1 = len(num1)
        n2 = len(num2)
        
        # Initialize the max output size
        output = [0] * (n1 + n2)
        
        for j in range(n2):
            carry = 0
            for i in range(n1):
                tmp_output = int(num1[i]) * int(num2[j]) + carry + output[i + j]
                output[i + j] = tmp_output % 10  # Remainder stays at position
                carry = tmp_output // 10  # Carry for the next position
                
            # Add remaining carry to the next position
            if carry > 0:
                output[j + n1] += carry

        # Remove leading zeros and reverse to form the final result
        while len(output) > 1 and output[-1] == 0:
            output.pop()

        return ''.join(map(str, output[::-1]))

```

## Rotate Image

Solution

You are given an `n x n` 2D `matrix` representing an image, rotate the image by **90** degrees (clockwise).

You have to rotate the image [**in-place**](https://en.wikipedia.org/wiki/In-place_algorithm), which means you have to modify the input 2D matrix directly. **DO NOT** allocate another 2D matrix and do the rotation.

### Solutions: Transpose and reverse rows
```python
class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        
        n = len(matrix)

        # Transpose and reverse might be better
        for i in range(n):
            for j in range(i,n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
                
        # Reverse the rows
        for i in range(n):
            matrix[i][:] = matrix[i][::-1]
```

## Minimum Window Substring

Solution

Given two strings `s` and `t` of lengths `m` and `n` respectively, return _the **minimum window**_ **_substring_** _of_ `s` _such that every character in_ `t` _(**including duplicates**) is included in the window_. If there is no such substring, return _the empty string_ `""`.

The testcases will be generated such that the answer is **unique**.

**Example 1:**

**Input:** s = "ADOBECODEBANC", t = "ABC"
**Output:** "BANC"
**Explanation:** The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.

**Example 2:**

**Input:** s = "a", t = "a"
**Output:** "a"
**Explanation:** The entire string s is the minimum window.

### Two pointer approach
- Start with (0,0) and extend the right pointer to capture all the chars in t
- Then start moving left pointer and keep track of min window size
- Hops between incrementing right and left pointer
- When all chars in t are captured then left pointer moves
- Once chars are missing the right pointer moves until all the chars captured again
```python
from collections import Counter

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if not s or not t:
            return ''
        
        t_dict = Counter(t)
        s_dict = {}
        
        l_ptr = 0
        minwindow = float('inf')
        minwin_index = (0, 0)
        required = len(t_dict)
        formed = 0
        
        for r_ptr in range(len(s)):
            r_char = s[r_ptr]
            s_dict[r_char] = s_dict.get(r_char, 0) + 1
            
            if r_char in t_dict and s_dict[r_char] == t_dict[r_char]:
                formed += 1
            
            while l_ptr <= r_ptr and formed == required:
                l_char = s[l_ptr]
                
                if (r_ptr - l_ptr + 1) < minwindow:
                    minwindow = r_ptr - l_ptr + 1
                    minwin_index = (l_ptr, r_ptr + 1)
                
                s_dict[l_char] -= 1
                if l_char in t_dict and s_dict[l_char] < t_dict[l_char]:
                    formed -= 1
                
                l_ptr += 1
        
        if minwindow == float('inf'):
            return ''
        else:
            return s[minwin_index[0]:minwin_index[1]]

```

##   Read N Characters Given Read4 II - Call multiple times

**Example 1:**
**Input:** file = "abc", queries = [1,2,1]
**Output:** [1,2,0]
**Explanation:** The test case represents the following scenario:
File file("abc");
Solution sol;
sol.read(buf, 1); // After calling your read method, buf should contain "a". We read a total of 1 character from the file, so return 1.
sol.read(buf, 2); // Now buf should contain "bc". We read a total of 2 characters from the file, so return 2.
sol.read(buf, 1); // We have reached the end of file, no more characters can be read. So return 0.
Assume buf is allocated and guaranteed to have enough space for storing all characters from the file.

```python
# The read4 API is already defined for you.
# def read4(buf4: List[str]) -> int:

class Solution:
    
    def __init__(self):
        self.extra_reads = []
        
    def read(self, buf: List[str], n: int) -> int:
        buf4 = ['']*4
        while (n > len(self.extra_reads)):
            new_reads = read4(buf4)
            self.extra_reads.extend(buf4[:new_reads])
            
            #Indicates EOF
            if new_reads < 4:
                break
        
        return_val = min(len(self.extra_reads), n)
        for ind in range(return_val):
            buf[ind] = self.extra_reads.pop(0)
        return return_val
```


## Given a `time` represented in the format `"HH:MM"`, form the next closest time by reusing the current digits. There is no limit on how many times a digit can be reused.

You may assume the given input string is always valid. For example, `"01:34"`, `"12:09"` are all valid. `"1:34"`, `"12:9"` are all invalid.

**Example 1:**
**Input:** time = "19:34"
**Output:** "19:39"
**Explanation:** The next closest time choosing from digits **1**, **9**, **3**, **4**, is **19:39**, which occurs 5 minutes later.
It is not **19:33**, because this occurs 23 hours and 59 minutes later.

**Example 2:**

**Input:** time = "23:59"
**Output:** "22:22"
**Explanation:** The next closest time choosing from digits **2**, **3**, **5**, **9**, is **22:22**.
It may be assumed that the returned time is next day's time since it is smaller than the input time numerically.


**Brute-force but works on leetcode**
```python
class Solution:
    def nextClosestTime(self, time: str) -> str:
        digits = set()
        for t in time:
            if t == ':':
                continue
            digits.add(t)

        hours = int(time[:2])
        minutes = int(time[3:])

        hrs_counter = hours
        min_counter = minutes + 1
        while True:
            num1 = str(hrs_counter // 10)
            num2 = str(hrs_counter % 10)
            if (num1 in digits) and (num2 in digits):
                for mins in range(min_counter, 60):
                    num3 = str(mins // 10)
                    num4 = str(mins % 10)
                    if (num3 in digits) and (num4 in digits):
                        return num1 + num2 + ':' + num3 + num4

            hrs_counter += 1
            hrs_counter = hrs_counter % 24
            min_counter = 0

```

- Convert to minutes and increment
- Use isdigit for set creations
- Use issubset for checking in one go

```python
class Solution:
    def nextClosestTime(self, time: str) -> str:
        # Extract the digits and convert them to integers
        digits = {int(c) for c in time if c.isdigit()}
        
        # Convert the current time to minutes
        current_minutes = int(time[:2]) * 60 + int(time[3:])
        
        while True:
            # Increment the time by 1 minute
            current_minutes = (current_minutes + 1) % (24 * 60)
            # Convert the time back to hours and minutes
            hours, minutes = divmod(current_minutes, 60)
            # Split the time into individual digits
            next_time_digits = {hours // 10, hours % 10, minutes // 10, minutes % 10}
            # Check if all digits are valid
            if next_time_digits.issubset(digits):
                return f"{hours:02}:{minutes:02}"

```

## Add Two Numbers
You are given two **non-empty** linked lists representing two non-negative integers. The digits are stored in **reverse order**, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

**Example 1:**

![](attachments/6d807a9b7486c05db0d3ae91d4dec327_MD5.jpg)

**Input:** l1 = [2,4,3], l2 = [5,6,4]
**Output:** [7,0,8]
**Explanation:** 342 + 465 = 807.

**Example 2:**

**Input:** l1 = [0], l2 = [0]
**Output:** [0]

**Example 3:**

**Input:** l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
**Output:** [8,9,9,9,0,0,0,1]


**Original code**
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        
        l1_ptr = l1
        l2_ptr = l2
        l1_l2_head = l1_l2_ptr = ListNode()
        carry = 0
        while (l1_ptr is not None) or (l2_ptr is not None):

            suml1l2 = carry
            
            if l1_ptr is not None:
                suml1l2 += l1_ptr.val
                l1_ptr = l1_ptr.next
                
            if l2_ptr is not None:
                suml1l2 += l2_ptr.val
                l2_ptr = l2_ptr.next
            
            l1_l2_ptr.val = suml1l2%10
            carry = suml1l2//10
            
            if (l1_ptr is not None) or (l2_ptr is not None):
                l1_l2_ptr.next = ListNode()
                l1_l2_ptr = l1_l2_ptr.next
            elif carry > 0:
                l1_l2_ptr.next = ListNode(carry)
                l1_l2_ptr = l1_l2_ptr.next

            
        return l1_l2_head
            
            
```

**Cleaned code**
```python
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        # Dummy node to simplify edge cases
        dummy_head = ListNode()
        current = dummy_head
        carry = 0

        # Traverse both lists
        while l1 or l2 or carry:
            # Compute sum for the current digits + carry
            val1 = l1.val if l1 else 0
            val2 = l2.val if l2 else 0
            total = val1 + val2 + carry

            # Update carry and current node
            carry = total // 10
            current.next = ListNode(total % 10)

            # Move pointers forward
            current = current.next
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next

        return dummy_head.next

```


