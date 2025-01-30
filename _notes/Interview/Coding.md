
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


## LRU Cache

Solution

Design a data structure that follows the constraints of a **[Least Recently Used (LRU) cache](https://en.wikipedia.org/wiki/Cache_replacement_policies#LRU)**

Implement the `LRUCache` class:

- `LRUCache(int capacity)` Initialize the LRU cache with **positive** size `capacity`.
- `int get(int key)` Return the value of the `key` if the key exists, otherwise return `-1`.
- `void put(int key, int value)` Update the value of the `key` if the `key` exists. Otherwise, add the `key-value` pair to the cache. If the number of keys exceeds the `capacity` from this operation, **evict** the least recently used key.

The functions `get` and `put` must each run in `O(1)` average time complexity.

**Example 1:**

**Input**
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
**Output**
[null, null, null, 1, null, -1, null, -1, 3, 4]

**Explanation**
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // cache is {1=1}
lRUCache.put(2, 2); // cache is {1=1, 2=2}
lRUCache.get(1);    // return 1
lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
lRUCache.get(2);    // returns -1 (not found)
lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
lRUCache.get(1);    // return -1 (not found)
lRUCache.get(3);    // return 3
lRUCache.get(4);    // return 4


**Using OrderedDict**
```python
from collections import OrderedDict

class LRUCache:

    def __init__(self, capacity: int):
        """
        Initialize the LRU Cache with a fixed capacity.
        """
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        """
        Return the value of the key if it exists in the cache.
        Move the accessed key to the end to mark it as recently used.
        """
        if key in self.cache:
            self.cache.move_to_end(key)  # Mark as recently used
            return self.cache[key]
        return -1  # Key not found

    def put(self, key: int, value: int) -> None:
        """
        Add or update a key-value pair in the cache.
        If the cache exceeds its capacity, remove the least recently used item.
        """
        if key in self.cache:
            # Update the value and mark as recently used
            self.cache[key] = value
        else:
            # Evict the least recently used item if capacity is reached
            if len(self.cache) == self.capacity:
                self.cache.popitem(last=False)  # Remove the oldest item
            self.cache[key] = value  # Add the new item
        
        self.cache.move_to_end(key)  # Mark as recently used

        

```

Alternative method: Dictionary + Doubly-linked list
{key} = Node(key,val,prev,next)
LRU->.....Nodes------>MRU
Keep moving to MRU if used and pop LRU to maintain capability


## Median of Two Sorted Arrays

Solution

Given two sorted arrays `nums1` and `nums2` of size `m` and `n` respectively, return **the median** of the two sorted arrays.

The overall run time complexity should be `O(log (m+n))`.

**Example 1:**

**Input:** nums1 = [1,3], nums2 = [2]
**Output:** 2.00000
**Explanation:** merged array = [1,2,3] and median is 2.

**Example 2:**

**Input:** nums1 = [1,2], nums2 = [3,4]
**Output:** 2.50000
**Explanation:** merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.

Brute-force: Two ptr approach walking over the two given arrays
- Time: O(M+N)
- Space: O(1)

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        
        n1 = len(nums1)
        n2 = len(nums2)
        
        if (n1+n2)%2 == 0:
            med_ind1 = (n1+n2)//2 - 1
            med_ind2 = (n1+n2)//2 
        else:
            med_ind1 = med_ind2 = (n1+n2+1)//2 - 1
            
        if n1 == 0:
            return 0.5*(nums2[med_ind1] + nums2[med_ind2])
        if n2 == 0:
            return 0.5*(nums1[med_ind1] + nums1[med_ind2])            
            
            
        result_median = 0    
        
        n1_ptr = 0
        n2_ptr = 0
        n3_ptr = 0
        while (n3_ptr <= med_ind2):
            
            if (n1_ptr < n1) and (n2_ptr < n2):
                if (nums1[n1_ptr] <= nums2[n2_ptr]):
                    current_num = nums1[n1_ptr]
                    n1_ptr += 1
                else:
                    current_num = nums2[n2_ptr]
                    n2_ptr += 1
            elif (n1_ptr < n1):
                current_num = nums1[n1_ptr]
                n1_ptr += 1
            elif (n2_ptr < n2):
                current_num = nums2[n2_ptr]
                n2_ptr += 1
            
            # print(n1_ptr, n2_ptr, current_num)
            if (n1+n2)%2 == 0:
                if (n3_ptr == med_ind1) or (n3_ptr == med_ind2):
                    result_median += 0.5*(current_num)
            else:
                if (n3_ptr == med_ind1):
                    result_median = current_num
                    
            n3_ptr += 1
            
        return result_median
```

**Optimised approach: Binary search O(log(min(m,n)))
- Choose smaller array
- Run binary search on it: low = 0, high = m, i = (low+high)//2,
- j = (m+n+1)//2 - i  (in other array)
![](attachments/52e840956467f7e54b9e10384a9af062_MD5.jpeg)

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        m = len(nums1)
        n = len(nums2)
        
        # Ensure nums1 is the smaller array
        if m > n:
            nums1, nums2 = nums2, nums1
            m, n = n, m
            
        low = 0
        high = m
        num_half = (m + n + 1) // 2  # The size of the left partition
        
        while low <= high:
            i = (low + high) // 2  # Partition index for nums1
            j = num_half - i      # Partition index for nums2
            
            # Calculate boundary values for the partitions
            maxleft1 = -float('inf') if i == 0 else nums1[i - 1]
            minright1 = float('inf') if i == m else nums1[i]
            maxleft2 = -float('inf') if j == 0 else nums2[j - 1]
            minright2 = float('inf') if j == n else nums2[j]
            
            # Check if the partition is valid
            if maxleft1 <= minright2 and maxleft2 <= minright1:
                # Median calculation
                if (m + n) % 2 == 0:  # Even length
                    return 0.5 * (max(maxleft1, maxleft2) + min(minright1, minright2))
                else:  # Odd length
                    return max(maxleft1, maxleft2)
            
            # Adjust binary search bounds
            if maxleft1 > minright2:
                high = i - 1  # Move left in nums1
            else:
                low = i + 1   # Move right in nums1

```


For dynamic computation of median: Use min and max heap approach
## Binary Tree Maximum Path Sum

Solution

A **path** in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence **at most once**. Note that the path does not need to pass through the root.

The **path sum** of a path is the sum of the node's values in the path.

Given the `root` of a binary tree, return _the maximum **path sum** of any **non-empty** path_.

**Example 1:**

![](attachments/6d575c4ce9b534b980c2ee376508e084_MD5.jpg)

**Input:** root = [1,2,3]
**Output:** 6
**Explanation:** The optimal path is 2 -> 1 -> 3 with a path sum of 2 + 1 + 3 = 6.

**Example 2:**

![](attachments/d29a334361627eba646e84254604f53a_MD5.jpg)

**Input:** root = [-10,9,20,null,null,15,7]
**Output:** 42
**Explanation:** The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7 = 42.


- Path from a node is: node.val + left_sum + right_sum 
	- Path goes from left branch to right_branch through the node
- Return from function is node.val + max(left_sum, right_sum) 
	- Going either branch but starts above the node
- Note the difference between pathmax computation and the return value
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        # Initialize the maximum path sum as a class attribute
        self.pathmax = float('-inf')
        
        def get_node_lr_sum(node: Optional[TreeNode]) -> int:
            if node is None:
                return 0
            
            # Recurse into left and right subtrees
            left_sum = max(0, get_node_lr_sum(node.left))  # Ignore negative paths
            right_sum = max(0, get_node_lr_sum(node.right))
            
            # Update the global maximum path sum
            self.pathmax = max(self.pathmax, left_sum + right_sum + node.val)
            
            # Return the maximum sum of a path passing through this node
            return node.val + max(left_sum, right_sum)
        
        # Start the recursion
        get_node_lr_sum(root)
        return self.pathmax

```



Poor implementation:
	- Loop over word in wordList and check if next word
		- if yes recurse with next word as the beginWord
	- Some caching related stuff: frozenset etc

```python
from typing import List, Set

class Solution:
    def __init__(self):
        self.cache = {}
        self.cache_larger = {}
    
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wordSet = set(wordList)  # Convert word list to a set for O(1) lookups
        if endWord not in wordSet:
            return 0

        def is_valid_next_word(current_word: str, target_word: str) -> bool:
            """Check if two words differ by exactly one character."""
            if (current_word, target_word) in self.cache:
                return self.cache[(current_word, target_word)]
            
            char_diff = sum(a != b for a, b in zip(current_word, target_word))
            result = char_diff == 1
            self.cache[(current_word, target_word)] = result
            return result

        def helper(beginWord: str, endWord: str, wordSet: Set[str]) -> int:
            """Recursive helper to find the shortest path."""
            if beginWord == endWord:
                return 1
            if (beginWord, endWord, frozenset(wordSet)) in self.cache_larger:
                return self.cache_larger[(beginWord, endWord, frozenset(wordSet))]
            
            min_length = float('inf')
            for word in list(wordSet):  # Iterate over the current word set
                if is_valid_next_word(beginWord, word):
                    wordSet.remove(word)  # Remove the word from the set
                    result = helper(word, endWord, wordSet)
                    if result != float('inf'):
                        min_length = min(min_length, 1 + result)
                    wordSet.add(word)  # Add the word back to the set
            
            self.cache_larger[(beginWord, endWord, frozenset(wordSet))] = min_length
            return min_length

        # Compute the ladder length
        ladder_length = helper(beginWord, endWord, wordSet)
        return 0 if ladder_length == float('inf') else ladder_length

```

## Word Ladder

Solution

A **transformation sequence** from word `beginWord` to word `endWord` using a dictionary `wordList` is a sequence of words `beginWord -> s1 -> s2 -> ... -> sk` such that:

- Every adjacent pair of words differs by a single letter.
- Every `si` for `1 <= i <= k` is in `wordList`. Note that `beginWord` does not need to be in `wordList`.
- `sk == endWord`

Given two words, `beginWord` and `endWord`, and a dictionary `wordList`, return _the **number of words** in the **shortest transformation sequence** from_ `beginWord` _to_ `endWord`_, or_ `0` _if no such sequence exists._

**Example 1:**

**Input:** beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"]
**Output:** 5
**Explanation:** One shortest transformation sequence is "hit" -> "hot" -> "dot" -> "dog" -> cog", which is 5 words long.

- **Time Complexity: $O(N*M^2)$**
- **Space Complexity: $O(N*M)$**
- N: number of words, M: length of words

```python
from collections import deque
from typing import List

class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wordSet = set(wordList)  # Convert wordList to a set for quick lookups
        
        # Early exits for invalid cases
        if endWord not in wordSet:
            return 0
        if len(beginWord) != len(endWord):
            return 0
        if len(beginWord) == 1:
            return 2

        # Remove beginWord from the set (if it exists)
        wordSet.discard(beginWord)

        # Create a pattern dictionary for all words in wordSet
        wordlen = len(beginWord)
        pattern_dict = {}
        for word in wordSet:
            for i in range(wordlen):
                pattern = word[:i] + '*' + word[i+1:]
                pattern_dict.setdefault(pattern, []).append(word)

        # Perform BFS for the shortest path
        ladder = deque([(beginWord, 1)])  # (current word, current level)
        visited = set()  # Track visited words to avoid cycles

        while ladder:
            word, level = ladder.popleft()  # Process the front of the deque
            visited.add(word)

            for i in range(wordlen):
                pattern = word[:i] + '*' + word[i+1:]
                
                # Skip patterns not in the dictionary
                if pattern not in pattern_dict:
                    continue

                for neighbor in pattern_dict[pattern]:
                    if neighbor == endWord:
                        return level + 1  # Found the shortest path
                    if neighbor not in visited:
                        ladder.append((neighbor, level + 1))
                        visited.add(neighbor)  # Mark as visited

        return 0  # If no path is found

```

## Word Squares

Solution

Given an array of **unique** strings `words`, return _all the_ **[word squares](https://en.wikipedia.org/wiki/Word_square)** _you can build from_ `words`. The same word from `words` can be used **multiple times**. You can return the answer in **any order**.

A sequence of strings forms a valid **word square** if the `kth` row and column read the same string, where `0 <= k < max(numRows, numColumns)`.

- For example, the word sequence `["ball","area","lead","lady"]` forms a word square because each word reads the same both horizontally and vertically.

**Example 1:**

**Input:** words = ["area","lead","wall","lady","ball"]
**Output:** [["ball","area","lead","lady"],["wall","area","lead","lady"]]
**Explanation:**
The output consists of two word squares. The order of output does not matter (just the order of words in each word square matters).

- Solution: 
	- Create a dict with prefix -> allowed_words
	- Recurse over combinations
- 
	-![](attachments/a43da186a773d5f61cf64dfe7d9675cd_MD5.jpeg)


```python
from typing import List, Dict
from collections import defaultdict

class Solution:
    def wordSquares(self, words: List[str]) -> List[List[str]]:
        if not words or len(set(len(w) for w in words)) > 1:
            return []
        
        wordlen = len(words[0])
        prefix_dict = defaultdict(list)
        
        for w in words:
            for ind in range(len(w) + 1):
                prefix_dict[w[:ind]].append(w)

        def _backtrack(step: int, current_words: List[str], results: List[List[str]]) -> None:
            if step == wordlen:
                results.append(current_words[:])  # Add a copy of the current words
                return
            
            prefix = ''.join(w[step] for w in current_words)
            for allowed_word in prefix_dict.get(prefix, []):
                _backtrack(step + 1, current_words + [allowed_word], results)
              
        results = []
        for w in words:
            _backtrack(1, [w], results)
        
        return results

```

## Strobogrammatic Number II

Solution

Given an integer `n`, return all the **strobogrammatic numbers** that are of length `n`. You may return the answer in **any order**.

A **strobogrammatic number** is a number that looks the same when rotated `180` degrees (looked at upside down).

**Example 1:**

**Input:** n = 2
**Output:** ["11","69","88","96"]

**Example 2:**

**Input:** n = 1
**Output:** ["0","1","8"]


Solution:
	- start_num + _backtrack + end_num
	- Important line: **if n != total_len:**
```python
from typing import List

class Solution:
    def findStrobogrammatic(self, n: int) -> List[str]:
        def _backtrack(n: int, total_len: int) -> List[str]:
            # Base cases
            if n == 0:
                return [""]
            if n == 1:
                return ["0", "1", "8"]

            # Get strobogrammatic numbers of length n-2
            middle_numbers = _backtrack(n - 2, total_len)
            results = []

            # Add strobogrammatic pairs around the middle numbers
            for mid in middle_numbers:
                # Avoid adding leading zeros for numbers with more than one digit
                if n != total_len:
                    results.append("0" + mid + "0")
                results.append("1" + mid + "1")
                results.append("6" + mid + "9")
                results.append("8" + mid + "8")
                results.append("9" + mid + "6")
            
            return results
        
        return _backtrack(n, n)
```

### **Key Differences**

|Feature|Recursive Approach|Backtracking Approach|
|---|---|---|
|**How recursion is used**|Combines smaller results into bigger results.|Dynamically explores paths, pruning invalid ones.|
|**Construction style**|Generates all solutions in one pass.|Builds solutions step by step, character by character.|
|**Intermediate filtering**|No dynamic filtering, just avoids leading zeroes in results.|Dynamically stops invalid paths (e.g., leading zero).|
|**Focus**|Solve smaller problems and combine solutions.|Explore all possibilities with constraints.|
|**Flexibility**|Less flexible for adding new constraints.|Very flexible and interactive.|


## Min Stack

Solution

Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the `MinStack` class:

- `MinStack()` initializes the stack object.
- `void push(int val)` pushes the element `val` onto the stack.
- `void pop()` removes the element on the top of the stack.
- `int top()` gets the top element of the stack.
- `int getMin()` retrieves the minimum element in the stack.

You must implement a solution with `O(1)` time complexity for each function.

```python
class MinStack:
    def __init__(self):
        self.vals = []

    def push(self, val: int) -> None:
        min_val = val if not self.vals else min(val, self.vals[-1][1])
        self.vals.append((val, min_val))

    def pop(self) -> None:
        if self.vals:
            self.vals.pop()

    def top(self) -> int:
        return self.vals[-1][0] if self.vals else None  # Handle empty case

    def getMin(self) -> int:
        return self.vals[-1][1] if self.vals else None  # Handle empty case
```

