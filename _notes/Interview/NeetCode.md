
## [143. Reorder List](https://leetcode.com/problems/reorder-list/)

You are given the head of a singly linked-list. The list can be represented as:

L0 → L1 → … → Ln - 1 → Ln

_Reorder the list to be on the following form:_

L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …

You may not modify the values in the list's nodes. Only nodes themselves may be changed.

**Example 1:**

![](attachments/7a961eb04a5bb3acaa22335ebdea4955_MD5.jpg)

**Input:** head = [1,2,3,4]
**Output:** [1,4,2,3]

## Solution
![](attachments/2d99d2bccb0752adda44d6fd72e7e856_MD5.jpeg)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        if not head or not head.next:
            return

        # Step 1: Find the middle of the list
        slow_ptr = fast_ptr = head
        while fast_ptr and fast_ptr.next:
            slow_ptr = slow_ptr.next
            fast_ptr = fast_ptr.next.next

        # Step 2: Reverse the second half
        prev, curr = None, slow_ptr.next
        slow_ptr.next = None  # Split the list
        while curr:
            tmp = curr.next
            curr.next = prev
            prev = curr
            curr = tmp

        # Step 3: Merge two halves
        first, second = head, prev
        while second:
            tmp1, tmp2 = first.next, second.next
            first.next = second
            second.next = tmp1
            first, second = tmp1, tmp2
```

## Sum of Two Integers

Medium

Given two integers `a` and `b`, return the sum of the two integers without using the `+` and `-` operators.

**Example 1:**

```java
Input: a = 1, b = 1

Output: 2
```


### Step 1: Sum without carry = XOR (`a ^ b`)

Example:

makefile

CopyEdit

`a = 5 (0101) b = 3 (0011) a ^ b = 6 (0110)  # sum bits without carry`

---

### Step 2: Calculate carry = AND (`a & b`), then shift left (`<< 1`)

makefile

CopyEdit

`a = 5 (0101) b = 3 (0011) a & b = 1 (0001) carry = 1 << 1 = 2 (0010)  # carry bits to add in the next step`

---

### Step 3: Repeat the addition with sum and carry

Now add:

- `sum_without_carry = a ^ b`
    
- `carry` (shifted AND)
    

Apply again:

- `a = sum_without_carry`
    
- `b = carry`
    

Repeat the process until carry (`b`) becomes zero.



```python
def getSum(a, b):
    while b != 0:
        carry = (a & b) << 1   # calculate carry
        a = a ^ b              # sum without carry
        b = carry              # prepare next carry

    return a

```


## Valid Binary Search Tree

Solved 

Medium

Given the `root` of a binary tree, return `true` if it is a **valid binary search tree**, otherwise return `false`.

A **valid binary search tree** satisfies the following constraints:

- The left subtree of every node contains only nodes with keys **less than** the node's key.
- The right subtree of every node contains only nodes with keys **greater than** the node's key.
- Both the left and right subtrees are also binary search trees.

**Example 1:**

![](attachments/7d028b26c23a5345f5901feb772d7f15_MD5.png)

```java
Input: root = [2,1,3]

Output: true
```

### Solution

![](attachments/84d762e7fbc0776a3b87ef29edf33065_MD5.jpeg)

```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def helper(node, minval, maxval):
            if not node:
                return True
            if not (minval < node.val < maxval):
                return False
            return (
                helper(node.left, minval, node.val) and
                helper(node.right, node.val, maxval)
            )
        return helper(root, -float('inf'), float('inf'))

```


## Encode and Decode Strings

Medium

Design an algorithm to encode a list of strings to a single string. The encoded string is then decoded back to the original list of strings.

Please implement `encode` and `decode`

**Example 1:**

```java
Input: ["neet","code","love","you"]

Output:["neet","code","love","you"]
```

### Solution
- Append len(strs) in the string with ':' for identification
	- 4:code
	- : helps with > 1 digit lengths
- During decode:
	- loop over all string i
	- use j to extract strings




```python
class Codec:

    def encode(self, strs: List[str]) -> str:
        # Use length + ':' + string as the encoding format
        return ''.join(f'{len(s)}:{s}' for s in strs)

    def decode(self, s: str) -> List[str]:
        res = []
        i = 0
        while i < len(s):
            # Find the colon separator to extract the length
            j = i
            while s[j] != ':':
                j += 1
            length = int(s[i:j])
            j += 1  # move past ':'
            res.append(s[j:j+length])
            i = j + length
        return res

```


## Kth Smallest Integer in BST

Solved 

Medium

Given the `root` of a binary search tree, and an integer `k`, return the `kth` smallest value (**1-indexed**) in the tree.

A **binary search tree** satisfies the following constraints:

- The left subtree of every node contains only nodes with keys **less than** the node's key.
- The right subtree of every node contains only nodes with keys **greater than** the node's key.
- Both the left and right subtrees are also binary search trees.

**Example 1:**

![](attachments/7d028b26c23a5345f5901feb772d7f15_MD5.png)


### Solution:
- BST is sorted so start from the left and keep the count


```python
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        self.count = 0
        self.result = None
        
        def inorder(node):
            if not node or self.result is not None:
                return
            inorder(node.left)
            self.count += 1
            if self.count == k:
                self.result = node.val
                return
            inorder(node.right)
        
        inorder(root)
        return self.result
```

## Subtree of Another Tree

Solved 

Easy

Given the roots of two binary trees `root` and `subRoot`, return `true` if there is a subtree of `root` with the same structure and node values of `subRoot` and `false` otherwise.

A subtree of a binary tree `tree` is a tree that consists of a node in `tree` and all of this node's descendants. The tree `tree` could also be considered as a subtree of itself.

**Example 1:**

![](attachments/0b086daa86bb5df6bc488d75f51ff9bc_MD5.png)

```java
Input: root = [1,2,3,4,5], subRoot = [2,4,5]

Output: true
```

Copy

**Example 2:**

![](attachments/b2778ed2a6a1a1881ccae60d6e051b32_MD5.png)

```java
Input: root = [1,2,3,4,5,null,null,6], subRoot = [2,4,5]

Output: false
```

### Solution:
- Pay attention to the second example False. It has to match exactly

```python
class Solution:   
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        if not root:
            return False

        if self.isequal(root, subRoot):
            return True

        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
    
    def isequal(self, root1, root2):
        if not root1 and not root2:
            return True
        if not root1 or not root2:
            return False

        return (root1.val == root2.val and 
                self.isequal(root1.left, root2.left) and 
                self.isequal(root1.right, root2.right))
```


# # House Robber II

Solved 

Medium

You are given an integer array `nums` where `nums[i]` represents the amount of money the `i`th house has. The houses are arranged in a circle, i.e. the first house and the last house are neighbors.

You are planning to rob money from the houses, but you cannot rob **two adjacent houses** because the security system will automatically alert the police if two adjacent houses were _both_ broken into.

Return the _maximum_ amount of money you can rob **without** alerting the police.

**Example 1:**

```java
Input: nums = [3,4,3]

Output: 4
```

Copy

Explanation: You cannot rob `nums[0] + nums[2] = 6` because `nums[0]` and `nums[2]` are adjacent houses. The maximum you can rob is `nums[1] = 4`.

**Example 2:**

```java
Input: nums = [2,9,8,3,6]

Output: 15
```

### Solution

![](attachments/3929b85d7f7d5dc886eb06aab985f47f_MD5.jpeg)

# ## Word Break

Solved 

Medium

Given a string `s` and a dictionary of strings `wordDict`, return `true` if `s` can be segmented into a space-separated sequence of dictionary words.

You are allowed to reuse words in the dictionary an unlimited number of times. You may assume all dictionary words are unique.

**Example 1:**

```java
Input: s = "neetcode", wordDict = ["neet","code"]

Output: true
```

Copy

Explanation: Return true because "neetcode" can be split into "neet" and "code".

**Example 2:**

```java
Input: s = "applepenapple", wordDict = ["apple","pen","ape"]

Output: true
```

### Solution
- list to set for efficient lookup
- Complexity:
	- Time: O(m*n*k)
		- sub_str.startswith(word) -> k
		- m -> len(wordDict)
		- n -> len(s)
	- Space: O(n) + O($sum$ len(wordDict))


```python
from functools import lru_cache
from typing import List

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        word_set = set(wordDict)

        @lru_cache(maxsize=None)
        def can_break(sub_str: str) -> bool:
            if not sub_str:
                return True

            for word in word_set:
                if sub_str.startswith(word) and can_break(sub_str[len(word):]):
                    return True
            return False

        return can_break(s)

```


#


### Solution
- Without the trie implementation
- Time complexity:
	- W = #words
	- M x N = board size
	- L: maximum lenght of word in words
	- DFS of L: O(4^L)
	- Total complexity: O(W x 4^L X M X N)
- Space complexity:
	- O(MXNXL)

```python
from collections import defaultdict
from typing import List

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        m, n = len(board), len(board[0])
        words = set(words)

        if m == 0 or n == 0 or not words:
            return []

        def wordExists(word: str, loc: tuple, visited: set) -> bool:
            if not word:
                return True

            i, j = loc
            if board[i][j] != word[0]:
                return False

            if len(word) == 1:
                return True

            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + di, j + dj
                new_loc = (ni, nj)

                if 0 <= ni < m and 0 <= nj < n and new_loc not in visited:
                    visited.add(new_loc)
                    if wordExists(word[1:], new_loc, visited):
                        return True
                    visited.remove(new_loc)

            return False

        board_dict = defaultdict(list)
        for i in range(m):
            for j in range(n):
                board_dict[board[i][j]].append((i, j))

        words_found = []
        for w in words:
            for loc in board_dict[w[0]]:
                visited = set([loc])
                if wordExists(w, loc, visited):
                    words_found.append(w)
                    break

        return words_found

```

### Trie implementation

- Complexity:
	- Time: O(WXL +MXNX4^L)
	- Space: O(WXL + T + M X N)

```python
from functools import lru_cache
from typing import List

class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        
        m = len(board)
        n = len(board[0])
        words = set(words)

        if m == 0 or n == 0 or len(words) == 0:
            return []

        # Create a trie data structure
        head = {}
        for w in words:
            node = head
            for c in w:
                if c not in node:
                    node[c] = {}
                node = node[c]
            node['end'] = True

        res = set()

        def dfs(i, j, node, path, visited):
            if 'end' in node:
                res.add(path)
            
            visited.add((i, j))

            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj

                if 0 <= ni < m and 0 <= nj < n and (ni, nj) not in visited:
                    c = board[ni][nj]
                    if c in node:
                        dfs(ni, nj, node[c], path + c, visited)
            
            visited.remove((i, j))

        for i in range(m):
            for j in range(n):
                c = board[i][j]
                if c in head:
                    dfs(i, j, head[c], c, set([(i, j)]))
        
        return list(res)

```


## Find Median From Data Stream

Solved 

Hard

The **[median](https://en.wikipedia.org/wiki/Median)** is the middle value in a sorted list of integers. For lists of _even_ length, there is no middle value, so the median is the [mean](https://en.wikipedia.org/wiki/Mean) of the two middle values.

For example:

- For `arr = [1,2,3]`, the median is `2`.
- For `arr = [1,2]`, the median is `(1 + 2) / 2 = 1.5`

### Solved
- Use two min heap

```python
import heapq

class MedianFinder:

    def __init__(self):
        # Lower half (max-heap via negation) and upper half (min-heap)
        self.nums_lower_half = []
        self.nums_upper_half = []
        self.total_length = 0

    def addNum(self, num: int) -> None:
        # Decide which heap to push into
        if self.total_length == 0 or num >= self.nums_upper_half[0]:
            heapq.heappush(self.nums_upper_half, num)
        else:
            heapq.heappush(self.nums_lower_half, -num)

        self.total_length += 1

        # Balance the heaps to maintain invariant:
        # len(lower) == len(upper) or len(lower) == len(upper) + 1
        while len(self.nums_lower_half) > self.total_length // 2:
            heapq.heappush(self.nums_upper_half, -heapq.heappop(self.nums_lower_half))
        while len(self.nums_lower_half) < self.total_length // 2:
            heapq.heappush(self.nums_lower_half, -heapq.heappop(self.nums_upper_half))

    def findMedian(self) -> float:
        if self.total_length % 2 == 0:
            return 0.5 * (-self.nums_lower_half[0] + self.nums_upper_half[0])
        else:
            return self.nums_upper_half[0]

```

### Cleaner version
- Always push on the lower and keep lower larger

```python
import heapq

class MedianFinder:

    def __init__(self):
        # Max-heap for the lower half (stored as negated values)
        self.lower = []
        # Min-heap for the upper half
        self.upper = []

    def addNum(self, num: int) -> None:
        # Always push to lower first (as max-heap)
        heapq.heappush(self.lower, -num)

        # Balance: move the max of lower to upper
        heapq.heappush(self.upper, -heapq.heappop(self.lower))

        # Maintain size invariant: lower can have one more element than upper
        if len(self.upper) > len(self.lower):
            heapq.heappush(self.lower, -heapq.heappop(self.upper))

    def findMedian(self) -> float:
        if len(self.lower) > len(self.upper):
            return -self.lower[0]
        else:
            return (-self.lower[0] + self.upper[0]) / 2

```

## [Design Add and Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure/)

Design a data structure that supports adding new words and finding if a string matches any previously added string.

Implement the `WordDictionary` class:

- `WordDictionary()` Initializes the object.
- `void addWord(word)` Adds `word` to the data structure, it can be matched later.
- `bool search(word)` Returns `true` if there is any string in the data structure that matches `word` or `false` otherwise. `word` may contain dots `'.'` where dots can be matched with any letter.

**Example:**

**Input**
["WordDictionary","addWord","addWord","addWord","search","search","search","search"]
[[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]
**Output**
[null,null,null,null,false,true,true,true] 

### Solution
- Trie data structure
### ✅ Summary Table

|Operation|Time Complexity|Space Complexity|
|---|---|---|
|`addWord(L)`|O(L)|O(L) per new word|
|`search(L)`|O(26^d) worst, O(L) best|O(N) for memoization|

```python
class WordDictionary:

    def __init__(self):
        self.trie_struct = {}   
        self.memory = {}  

    def addWord(self, word: str) -> None:
        node = self.trie_struct
        for c in word:
            if c not in node:
                node[c] = {}
            node = node[c]    
        node['end'] = True
        self.memory = {}

    def search(self, word: str) -> bool:

        def word_exists(node, ind: int) -> bool:
            if not node or node is True: #when node['end'] comes in
                return False

            if ind == len(word):
                return 'end' in node

            if word[ind] in node:
                return word_exists(node[word[ind]], ind + 1)
            
            if word[ind] == '.':
                return any(word_exists(node[c], ind + 1) for c in node)

            return False

        if word in self.memory:
            return self.memory[word]
        else:
            self.memory[word] = word_exists(self.trie_struct, 0)
            return self.memory[word]

```


## Invert Binary tree

### Solution: 
- return node in the end
- Time: O(N)
- Space: Stack O(h)
	- h: height of tree
		- worst case: O(n)
		- average: O(log n)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        
        def invert(node):
            if not node:
                return None
            node.left, node.right = invert(node.right), invert(node.left)
            return node

        return invert(root)
```