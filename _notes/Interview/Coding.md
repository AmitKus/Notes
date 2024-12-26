
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