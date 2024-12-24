
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