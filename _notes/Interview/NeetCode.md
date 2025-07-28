
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

# Rotate Image

Solved 

Medium

Given a square `n x n` matrix of integers `matrix`, rotate it by 90 degrees _clockwise_.

You must rotate the matrix _in-place_. Do not allocate another 2D matrix and do the rotation.

**Example 1:**

![](attachments/26097c4a98b914e6f16fe407d97d4656_MD5.png)


### Solution:
- Rotation clockwise spirally: start with top and keep going down. The number of elements will decrease as we go down

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix)

        for depth in range(n // 2):
            for i in range(n - 1 - 2 * depth):
                tmp, matrix[i+depth][n-1-depth] = matrix[i+depth][n-1-depth], matrix[depth][i+depth]
                tmp, matrix[n-1-depth][n-1-i-depth] = matrix[n-1-depth][n-1-i-depth], tmp
                tmp, matrix[n-1-i-depth][depth] = matrix[n-1-i-depth][depth], tmp
                matrix[depth][i+depth] = tmp
```



## # Number of Islands

Solved 

Medium

Given a 2D grid `grid` where `'1'` represents land and `'0'` represents water, count and return the number of islands.

An **island** is formed by connecting adjacent lands horizontally or vertically and is surrounded by water. You may assume water is surrounding the grid (i.e., all the edges are water).

### Solution:
- Option1: Keep track using visited and traverse
- Option2: Update grid itself




```python
from typing import List

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m = len(grid)
        n = len(grid[0])

        def traverse_island(i: int, j: int):
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == '1':
                    grid[ni][nj] = '0'
                    traverse_island(ni, nj)

        num_islands = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    grid[i][j] = '0'
                    traverse_island(i, j)
                    num_islands += 1

        return num_islands

```

```python
from typing import List, Set, Tuple

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m = len(grid)
        n = len(grid[0])

        def traverse_island(i: int, j: int, visited: Set[Tuple[int, int]]):
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if (
                    0 <= ni < m and
                    0 <= nj < n and
                    (ni, nj) not in visited and
                    grid[ni][nj] == '1'
                ):
                    visited.add((ni, nj))
                    traverse_island(ni, nj, visited)

        num_islands = 0
        visited = set()

        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1' and (i, j) not in visited:
                    visited.add((i, j))
                    traverse_island(i, j, visited)
                    num_islands += 1

        return num_islands

```


```python [stack approach]
from typing import List

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m = len(grid)
        n = len(grid[0])

        def traverse_island(i: int, j: int):
            stack = [(i, j)]
            while stack:
                ci, cj = stack.pop()
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == '1':
                        grid[ni][nj] = '0'
                        stack.append((ni, nj))

        num_islands = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    grid[i][j] = '0'
                    traverse_island(i, j)
                    num_islands += 1

        return num_islands

```


## # Graph Valid Tree

Solved 

Medium

Given `n` nodes labeled from `0` to `n - 1` and a list of **undirected** edges (each edge is a pair of nodes), write a function to check whether these edges make up a valid tree.

**Example 1:**

```java
Input:
n = 5
edges = [[0, 1], [0, 2], [0, 3], [1, 4]]

Output:
true
```

Copy

**Example 2:**

```java
Input:
n = 5
edges = [[0, 1], [1, 2], [2, 3], [1, 3], [1, 4]]

Output:
false
```

### Solution:
- n-1 edges for n nodes and all nodes reachable
- No need to check for cycles because of first condition
- Complexity:
	- Time: O(nodes)
	- Space: 
		- adj_matrix: O(edges) ~ O(nodes)
		- queue: O(nodes)

```python
from collections import defaultdict, deque
from typing import List

class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        if len(edges) != n - 1:
            return False

        adj_matrix = defaultdict(list)
        for u, v in edges:
            adj_matrix[u].append(v)
            adj_matrix[v].append(u)

        visited = {0}
        q = deque([0])
        while q:
            node = q.popleft()
            for neighbor in adj_matrix[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append(neighbor)

        return len(visited) == n

```


## # Linked List Cycle Detection

Easy

Given the beginning of a linked list `head`, return `true` if there is a cycle in the linked list. Otherwise, return `false`.

There is a cycle in a linked list if at least one node in the list can be visited again by following the `next` pointer.

Internally, `index` determines the index of the beginning of the cycle, if it exists. The tail node of the list will set it's `next` pointer to the `index-th` node. If `index = -1`, then the tail node points to `null` and no cycle exists.

**Note:** `index` is **not** given to you as a parameter.

**Example 1:**

![](attachments/7b7eaa104ba3158d748a72a608d6cc93_MD5.png)

```java
Input: head = [1,2,3,4], index = 1

Output: true
```


## # Meeting Rooms II

Solved 

Medium

Given an array of meeting time interval objects consisting of start and end times `[[start_1,end_1],[start_2,end_2],...] (start_i < end_i)`, find the minimum number of days required to schedule all meetings without any conflicts.

**Note:** (0,8),(8,10) is not considered a conflict at 8.

**Example 1:**

```java
Input: intervals = [(0,40),(5,10),(15,20)]

Output: 2
```

Copy

Explanation:  
day1: (0,40)  
day2: (5,10),(15,20)

**Example 2:**

```java
Input: intervals = [(4,9)]

Output: 1
```



### Solution:
- Iterate through meetings based on starting time and keep a heap of en times.
- Pop the meeting where the new meetings start time is larger than end times of meeting in the heap
- Complexity:
	- Time: Min heap O(Nlog(N)) -- sorting, min-heap push
	- Space: O(N)

```python
import heapq

class Solution:
    def minMeetingRooms(self, intervals: List[Interval]) -> int:
        
        if not intervals:
            return 0

        intervals = sorted(intervals, key=lambda x: x.start)
        queue = []
        min_room = 0

        for interval in intervals:
            st = interval.start
            en = interval.end

            while queue and queue[0] <= st:
                heapq.heappop(queue)

            heapq.heappush(queue, en)
            min_room = max(min_room, len(queue))

        return min_room
```


# Top K Frequent Elements

Solved 

Medium

Given an integer array `nums` and an integer `k`, return the `k` most frequent elements within the array.

The test cases are generated such that the answer is always **unique**.

You may return the output in **any order**.

**Example 1:**

```java
Input: nums = [1,2,2,3,3,3], k = 2

Output: [2,3]
```

### Solutions: Three approaches

Time: O(n log(n))
```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = {}
        for num in nums:
            count[num] = 1 + count.get(num, 0)

        arr = []
        for num, cnt in count.items():
            arr.append([cnt, num])
        arr.sort()

        res = []
        while len(res) < k:
            res.append(arr.pop()[1])
        return res
```

Time: O(n log(k))
```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = {}
        for num in nums:
            count[num] = 1 + count.get(num, 0)

        heap = []
        for num in count.keys():
            heapq.heappush(heap, (count[num], num))
            if len(heap) > k:
                heapq.heappop(heap)

        res = []
        for i in range(k):
            res.append(heapq.heappop(heap)[1])
        return res
```

Time: O(n)
- max_freq = len(nums)
- min_freq = 0
```python
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = {}
        freq = [[] for i in range(len(nums) + 1)]

        for num in nums:
            count[num] = 1 + count.get(num, 0)
        for num, cnt in count.items():
            freq[cnt].append(num)
        
        res = []
        for i in range(len(freq) - 1, 0, -1):
            for num in freq[i]:
                res.append(num)
                if len(res) == k:
                    return res
```

## # Longest Repeating Character Replacement

Medium

You are given a string `s` consisting of only uppercase english characters and an integer `k`. You can choose up to `k` characters of the string and replace them with any other uppercase English character.

After performing at most `k` replacements, return the length of the longest substring which contains only one distinct character.

**Example 1:**

```java
Input: s = "XYYX", k = 2

Output: 4
```

Copy

Explanation: Either replace the 'X's with 'Y's, or replace the 'Y's with 'X's.

**Example 2:**

```java
Input: s = "AAABABB", k = 1

Output: 5
```

Copy

### Solution
- Sliding window,
- Keep a dict of frequency of chars and if the whole window is valid:
	- K changes will turn it into window with same chars

```python
from collections import defaultdict

class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        freq_dict = defaultdict(int)
        max_count = 0
        start = 0
        maxwindow = 0

        for end in range(len(s)):
            freq_dict[s[end]] += 1
            max_count = max(max_count, freq_dict[s[end]])

            if end - start + 1 - max_count > k:
                freq_dict[s[start]] -= 1
                start += 1

            maxwindow = max(maxwindow, end - start + 1)

        return maxwindow
```

## Longest common substrig

### **Problem Statement**

Given two strings `text1` and `text2`, find the **length of the longest subsequence** that is **common to both strings**.

- A **subsequence** is a sequence that appears in the same order but **not necessarily contiguously**.
    
- Not to be confused with a **substring**, which must be contiguous.


### Solution
- Basic idea

```
dp[i][j] = length of LCS between text1[0...i-1] and text2[0...j-1] 
```

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        if not text1 or not text2:
            return 0

        n1, n2 = len(text1), len(text2)
        dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]

        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[n1][n2]
```


## Search in Rotated Sorted Array

Solved 

Medium

You are given an array of length `n` which was originally sorted in ascending order. It has now been **rotated** between `1` and `n` times. For example, the array `nums = [1,2,3,4,5,6]` might become:

- `[3,4,5,6,1,2]` if it was rotated `4` times.
- `[1,2,3,4,5,6]` if it was rotated `6` times.

Given the rotated sorted array `nums` and an integer `target`, return the index of `target` within `nums`, or `-1` if it is not present.

You may assume all elements in the sorted rotated array `nums` are **unique**,

A solution that runs in `O(n)` time is trivial, can you write an algorithm that runs in `O(log n) time`?

**Example 1:**

```java
Input: nums = [3,4,5,6,1,2], target = 1

Output: 4
```


### Solution:

- Step 1: Find the sorted half: left or right based on start,mid
- Step 2: Find if target in sorted half then update start/end and else will take care of other

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        start, end = 0, len(nums) - 1

        while start <= end:
            mid = (start + end) // 2

            if nums[mid] == target:
                return mid

            # Check if the left half is sorted
            if nums[start] <= nums[mid]:
                if nums[start] <= target < nums[mid]:
                    end = mid - 1
                else:
                    start = mid + 1
            # Otherwise, the right half must be sorted
            else:
                if nums[mid] < target <= nums[end]:
                    start = mid + 1
                else:
                    end = mid - 1

        return -1

```

## Find Minimum in Rotated Sorted Array

Solved 

Medium

You are given an array of length `n` which was originally sorted in ascending order. It has now been **rotated** between `1` and `n` times. For example, the array `nums = [1,2,3,4,5,6]` might become:

- `[3,4,5,6,1,2]` if it was rotated `4` times.
- `[1,2,3,4,5,6]` if it was rotated `6` times.

Notice that rotating the array `4` times moves the last four elements of the array to the beginning. Rotating the array `6` times produces the original array.

Assuming all elements in the rotated sorted array `nums` are **unique**, return the minimum element of this array.

A solution that runs in `O(n)` time is trivial, can you write an algorithm that runs in `O(log n) time`?

**Example 1:**

```java
Input: nums = [3,4,5,6,1,2]

Output: 1
```

### Solution:
- Think of simple logic:
	- if nums[mid] > nums[right] -> min is on right of mid so left = mid + 1
	- else: mid is min or to the left -> focus on right = mid (not mid-1)
- For duplicates: shrink the window: left +=1 or right -= 1


```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        if not nums:
            return -1

        left, right = 0, len(nums) - 1

        while left < right:
            mid = (left + right) // 2
            if nums[mid] > nums[right]:
                left = mid + 1  # move right, min must be after mid
            else:
                right = mid     # min is at mid or to the left

        return nums[left]
```


## Binary tree

## 1. **Preorder Traversal** (Root → Left → Right)

### 🔁 Recursive


```python
def preorder(root):
    result = []
    def dfs(node):
        if not node:
            return
        result.append(node.val)   # visit root
        dfs(node.left)            # left
        dfs(node.right)           # right
    dfs(root)
    return result

```

### 🔁 Iterative

```python
def preorder(root):
    if not root:
        return []
    stack = [root]
    result = []
    while stack:
        node = stack.pop()
        result.append(node.val)   # visit root
        if node.right:
            stack.append(node.right)  # right pushed first
        if node.left:
            stack.append(node.left)   # left will be popped first
    return result

```


## ✅ 2. **Inorder Traversal** (Left → Root → Right)

### 🔁 Recursive

```python
def inorder(root):
    result = []
    def dfs(node):
        if not node:
            return
        dfs(node.left)            # left
        result.append(node.val)   # visit root
        dfs(node.right)           # right
    dfs(root)
    return result

```

### 🔁 Iterative

```python
def inorder(root):
    result = []
    stack = []
    curr = root
    while stack or curr:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        result.append(curr.val)   # visit root
        curr = curr.right
    return result

```

## ✅ 3. **Postorder Traversal** (Left → Right → Root)

### 🔁 Recursive

```python
def postorder(root):
    result = []
    def dfs(node):
        if not node:
            return
        dfs(node.left)            # left
        dfs(node.right)           # right
        result.append(node.val)   # visit root
    dfs(root)
    return result
```

### 🔁 Iterative (Using 2 stacks or modified preorder)

```python
def postorder(root):
    if not root:
        return []
    stack = [root]
    result = []
    while stack:
        node = stack.pop()
        result.append(node.val)  # root
        if node.left:
            stack.append(node.left)   # left
        if node.right:
            stack.append(node.right)  # right
    return result[::-1]  # reverse to get left→right→root
```


## # Construct Binary Tree from Preorder and Inorder Traversal

Solved 

Medium

You are given two integer arrays `preorder` and `inorder`.

- `preorder` is the preorder traversal of a binary tree
- `inorder` is the inorder traversal of the same tree
- Both arrays are of the same size and consist of unique values.

Rebuild the binary tree from the preorder and inorder traversals and return its root.

**Example 1:**

![](attachments/9e6d27459873566599de10930ca92d27_MD5.png)

```java
Input: preorder = [1,2,3,4], inorder = [2,1,3,4]

Output: [1,2,3,null,null,null,4]
```

Copy

**Example 2:**

```java
Input: preorder = [1], inorder = [1]

Output: [1]
```


### Solution:
- Main idea:
	- preorder: root->left->right
	- inorder: left->root->right
	- Pick root from preorder, and use inorder to find the left and right nodes

### Example for logic

preorder = [1, 2, 4, 5, 3, 6, 7]
inorder  = [4, 2, 5, 1, 6, 3, 7]

### 🔁 Step 1: Root from preorder

`preorder[0] = 1 → root = 1`

Find `1` in `inorder`:

`inorder.index(1) = 3`

- Left inorder: `inorder[:3] = [4, 2, 5]
- Right inorder: `inorder[4:] = [6, 3, 7]`

```python
root.left  = self.buildTree([2, 4, 5], [4, 2, 5])
root.right = self.buildTree([3, 6, 7], [6, 3, 7])
```
```
        1
       / \
      2   3
     / \ / \
    4  5 6  7

```

Time: O(n2)
Space: O(n)

```python

from typing import List, Optional

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder or not inorder:
            return None

        root_val = preorder[0]
        root = TreeNode(root_val)
        mid = inorder.index(root_val)

        root.left = self.buildTree(preorder[1 : mid + 1], inorder[:mid])
        root.right = self.buildTree(preorder[mid + 1 :], inorder[mid + 1 :])

        return root
```

Time O(n): Not slicing the list everytime
Space O(n)

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if not preorder or not inorder:
            return None

        # Map from value to index in inorder for O(1) lookup
        idx_map = {val: idx for idx, val in enumerate(inorder)}
        self.pre_idx = 0  # global pointer in preorder array

        def build(left, right):
            if left > right:
                return None

            # Pick the current root from preorder
            root_val = preorder[self.pre_idx]
            self.pre_idx += 1
            root = TreeNode(root_val)

            # Partition inorder into left and right subtrees
            index = idx_map[root_val]

            root.left = build(left, index - 1)
            root.right = build(index + 1, right)

            return root

        return build(0, len(inorder) - 1)
```


# Combination Sum

Solved 

Medium

You are given an array of **distinct** integers `nums` and a target integer `target`. Your task is to return a list of all **unique combinations** of `nums` where the chosen numbers sum to `target`.

The **same** number may be chosen from `nums` an **unlimited number of times**. Two combinations are the same if the frequency of each of the chosen numbers is the same, otherwise they are different.

You may return the combinations in **any order** and the order of the numbers in each combination can be in **any order**.

**Example 1:**

```java
Input: 
nums = [2,5,6,9] 
target = 9

Output: [[2,2,5],[9]]
```

### Solution:
- Time: O(2^T/m) - choose or not choose, m is min(nums)
- SPACE: O(T/m) - number of valid combinatiosn




```python
class Solution:
    def combinationSum(self, nums: List[int], target: int) -> List[List[int]]:
        res = []

        def backtrack(start_idx: int, path: List[int], target_sum: int):
            if target_sum == 0:
                res.append(path[:])  # make a copy
                return
            for i in range(start_idx, len(nums)):
                if nums[i] <= target_sum:
                    path.append(nums[i])
                    backtrack(i, path, target_sum - nums[i])  # reduce target
                    path.pop()

        backtrack(0, [], target)
        return res

```

Bottom up approach
- Time: O(TNA)
- Space: O(C*A)


```python
class Solution:
    def combinationSum(self, nums: List[int], target: int) -> List[List[int]]:
        dp = [[] for _ in range(target + 1)]
        dp[0] = [[]]  # Base case: one way to make 0 — empty list

        for t in range(1, target + 1):
            for n in nums:
                if t - n >= 0:
                    for comb in dp[t - n]:
                        # Only add if numbers are non-decreasing (to avoid duplicates like [2,3] and [3,2])
                        if not comb or n >= comb[-1]:
                            dp[t].append(comb + [n])

        return dp[target]
```


## Maximum Subarray

Solved 

Medium

Given an array of integers `nums`, find the subarray with the largest sum and return the sum.

A **subarray** is a contiguous non-empty sequence of elements within an array.

**Example 1:**

```java
Input: nums = [2,-3,4,-2,2,1,-1,4]

Output: 8
```

### Solution:
- Kadane' algorithm

```python
class Solution:

def maxSubArray(self, nums: List[int]) -> int:

	current_sum = max_sum = nums[0]
	for num in nums[1:]:
		current_sum = max(num, current_sum+num)
		max_sum = max(max_sum, current_sum)
return max_sum
```


#### Naive DP

```python
from typing import List

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [0] * n
        dp[0] = nums[0]
        max_sum = dp[0]

        for i in range(1, n):
            dp[i] = max(nums[i], dp[i-1] + nums[i])
            max_sum = max(max_sum, dp[i])

        return max_sum

```


## 3sum, 4sum, ksum problem

# 3Sum

Solved 

Medium

Given an integer array `nums`, return all the triplets `[nums[i], nums[j], nums[k]]` where `nums[i] + nums[j] + nums[k] == 0`, and the indices `i`, `j` and `k` are all distinct.

The output should _not_ contain any duplicate triplets. You may return the output and the triplets in **any order**.

**Example 1:**

```java
Input: nums = [-1,0,1,2,-1,-4]

Output: [[-1,-1,2],[-1,0,1]]
```


### Solution:
- Sort the array and then two pointer

### 3SUM -> 0/target

```python
from typing import List

class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        res = []

        for i in range(n):
            if i > 0 and nums[i] == nums[i-1]:
                continue  # Skip duplicates

            left, right = i + 1, n - 1
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                if total == 0:
                    res.append([nums[i], nums[left], nums[right]])
                    left += 1
                    right -= 1
                    # Skip duplicates for left and right
                    while left < right and nums[left] == nums[left - 1]:
                        left += 1
                    while left < right and nums[right] == nums[right + 1]:
                        right -= 1
                elif total < 0:
                    left += 1
                else:
                    right -= 1

        return res

```

#### 4SUM -> Target
- Same 2 pointer logic but now under two loops i,j
- See the duplicates part which is important

```python
from typing import List

class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        res = []

        for i in range(n):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            for j in range(i + 1, n):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue

                left, right = j + 1, n - 1
                while left < right:
                    sum_total = nums[i] + nums[j] + nums[left] + nums[right]
                    if sum_total == target:
                        res.append([nums[i], nums[j], nums[left], nums[right]])
                        left += 1
                        right -= 1
                        while left < right and nums[left] == nums[left - 1]:
                            left += 1
                        while left < right and nums[right] == nums[right + 1]:
                            right -= 1
                    elif sum_total < target:
                        left += 1
                    else:
                        right -= 1

        return res
```


# # Lowest Common Ancestor in Binary Search Tree

Solved 

Medium

Given a binary search tree (BST) where all node values are _unique_, and two nodes from the tree `p` and `q`, return the lowest common ancestor (LCA) of the two nodes.

The lowest common ancestor between two nodes `p` and `q` is the lowest node in a tree `T` such that both `p` and `q` as descendants. The ancestor is allowed to be a descendant of itself.

**Example 1:**

![](attachments/1f95023cca8ec9bc5915b33b50b075e5_MD5.png)

```java
Input: root = [5,3,8,1,4,7,9,null,2], p = 3, q = 8

Output: 5
```

### Solution
- Find the split point for p and q
	- If both are smaller than node: node.left
	- If both are larger than node: node.right
	- At node: node.val will be between p and q


```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        # Ensure p.val <= q.val for simplicity (optional)
        if p.val > q.val:
            p, q = q, p

        node = root
        while node:
            if q.val < node.val:
                # Both p and q are smaller → go left
                node = node.left
            elif p.val > node.val:
                # Both p and q are greater → go right
                node = node.right
            else:
                # This is the split point → LCA found
                return node

```


## # Jump Game

Solved 

Medium

You are given an integer array `nums` where each element `nums[i]` indicates your maximum jump length at that position.

Return `true` if you can reach the last index starting from index `0`, or `false` otherwise.

**Example 1:**

```java
Input: nums = [1,2,0,1,0]

Output: true
```



Explanation: First jump from index 0 to 1, then from index 1 to 3, and lastly from index 3 to 4.


### Solution
- Time: O(n2)
- Space: O(n)

```python
# Bottom-up
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        dp = [False] * n
        dp[-1] = True  # You can always "reach" the end from the end

        for i in range(n - 2, -1, -1):  # Start from second last index
            for j in range(i + 1, min(i + nums[i] + 1, n)):  # Look ahead within jump range
                if dp[j]:
                    dp[i] = True
                    break

        return dp[0]
```

```python
# Top-Down
from functools import lru_cache
from typing import List

class Solution:
    def canJump(self, nums: List[int]) -> bool:
        
        n = len(nums)

        if n <= 1:
            return True

        @lru_cache(maxsize=None)
        def jump(ind: int):
            if ind == n - 1:
                return True

            for step in range(1, nums[ind] + 1):
                if ind + step < n and jump(ind + step):
                    return True

            return False
        
        return jump(0)

```


## Hard: # Minimum Window Substring

Solved 

Hard

Given two strings `s` and `t`, return the shortest **substring** of `s` such that every character in `t`, including duplicates, is present in the substring. If such a substring does not exist, return an empty string `""`.

You may assume that the correct output is always unique.

**Example 1:**

```java
Input: s = "OUZODYXAZV", t = "XYZ"

Output: "YXAZ"
```

Copy

Explanation: `"YXAZ"` is the shortest substring that includes `"X"`, `"Y"`, and `"Z"` from string `t`.

**Example 2:**

```java
Input: s = "xyz", t = "xyz"

Output: "xyz"
```

Copy

**Example 3:**

```java
Input: s = "x", t = "xy"

Output: ""
```


### Solution:
 - Basic idea: running window to capture the frequncy when matches the t ...try shrinking the window and gt the smallest
- Time : O(s + t)
- Space: O(s + t)

```python
from collections import Counter, defaultdict

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        if len(t) > len(s):
            return ''

        t_dict = Counter(t)
        window = defaultdict(int)
        have, need = 0, len(t_dict)
        start = 0
        window_ind = [0, 0]
        window_len = float('inf')

        for end in range(len(s)):
            window[s[end]] += 1

            if s[end] in t_dict and window[s[end]] == t_dict[s[end]]:
                have += 1

            # Try shrinking the window from the left
            while have == need:
                # Update result if smaller window is found
                if end - start + 1 < window_len:
                    window_len = end - start + 1
                    window_ind = [start, end]

                window[s[start]] -= 1
                if s[start] in t_dict and window[s[start]] < t_dict[s[start]]:
                    have -= 1
                start += 1

        if window_len == float('inf'):
            return ''
        return s[window_ind[0]:window_ind[1] + 1]

```



## Hard # Binary Tree Maximum Path Sum

Solved 

Hard

Given the `root` of a _non-empty_ binary tree, return the maximum **path sum** of any _non-empty_ path.

A **path** in a binary tree is a sequence of nodes where each pair of adjacent nodes has an edge connecting them. A node can _not_ appear in the sequence more than once. The path does _not_ necessarily need to include the root.

The **path sum** of a path is the sum of the node's values in the path.

**Example 1:**

![](attachments/52c07dca1a0443c439f137c5b6d78a7b_MD5.png)

```java
Input: root = [1,2,3]

Output: 6
```


### Solution:



### 🧠 Example:

Suppose you have this tree:

markdown

     -10
     /  \
    9   20
        / \
       15  7


Here’s what happens:

- `dfs(15)` returns `15`
    
- `dfs(7)` returns `7`
    
- At `20`, the best path _through_ it is `15 + 20 + 7 = 42`, so we update `self.maxpathsum`


- Keep track of maxpathsum: nodel.val + left + right
- Only return the max branch up so:  ```return node.val + max(left_pathsum, right_pathsum) ```

- Complexity: Time O(node)
- Space: O(H)

```python
from typing import Optional

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        
        self.maxpathsum = -float('inf')

        def dfs(node: Optional[TreeNode]) -> int:
            if not node:
                return 0

            left_pathsum = max(0, dfs(node.left))
            right_pathsum = max(0, dfs(node.right))

            self.maxpathsum = max(self.maxpathsum, node.val + left_pathsum + right_pathsum)

            return node.val + max(left_pathsum, right_pathsum)

        dfs(root)
        return self.maxpathsum

```



## # Word Search

Solved 

Medium

Given a 2-D grid of characters `board` and a string `word`, return `true` if the word is present in the grid, otherwise return `false`.

For the word to be present it must be possible to form it with a path in the board with horizontally or vertically neighboring cells. The same cell may not be used more than once in a word.

**Example 1:**

![](attachments/30e64c8e43e93e3b4c0a457f72945fb4_MD5.png)

```java
Input: 
board = [
  ["A","B","C","D"],
  ["S","A","A","T"],
  ["A","C","A","E"]
],
word = "CAT"
```


Complexity:
 - Time: O(MXNX4^L)
 - Space: O(L)

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        
        m = len(board)
        n = len(board[0])

        def dfs(i: int, j: int, ind: int, visited: Set[Tuple[int, int]]) -> bool:
            if ind == len(word):
                return True

            for di, dj in [(1,0), (-1,0), (0,1), (0,-1)]:
                ni, nj = i + di, j + dj
                if (0 <= ni < m) and (0 <= nj < n) and (ni, nj) not in visited:
                    if board[ni][nj] == word[ind]:
                        visited.add((ni, nj))
                        if dfs(ni, nj, ind + 1, visited):
                            return True
                        visited.remove((ni, nj))
            return False

        for i in range(m):
            for j in range(n):
                if board[i][j] == word[0]:
                    visited = {(i, j)}
                    if dfs(i, j, 1, visited):
                        return True
                    
        return False

```


## # Course Schedule

Medium

You are given an array `prerequisites` where `prerequisites[i] = [a, b]` indicates that you **must** take course `b` first if you want to take course `a`.

The pair `[0, 1]`, indicates that must take course `1` before taking course `0`.

There are a total of `numCourses` courses you are required to take, labeled from `0` to `numCourses - 1`.

Return `true` if it is possible to finish all courses, otherwise return `false`.

**Example 1:**

```java
Input: numCourses = 2, prerequisites = [[0,1]]

Output: true
```

Copy

Explanation: First take course 1 (no prerequisites) and then take course 0.

**Example 2:**

```java
Input: numCourses = 2, prerequisites = [[0,1],[1,0]]

Output: false
```


### Solution
- Think of cycle detection as the general problem here.
- Keep the visit state of the node

```python
def canFinish(numCourses, prerequisites):
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[course].append(prereq)

    visit_state = [0] * numCourses  # 0 = unvisited, 1 = visiting, 2 = visited

    def dfs(course):
        if visit_state[course] == 1:  # Cycle detected
            return False
        if visit_state[course] == 2:  # Already checked
            return True

        visit_state[course] = 1  # Mark as visiting
        for prereq in graph[course]:
            if not dfs(prereq):
                return False
        visit_state[course] = 2  # Mark as visited
        return True

    for course in range(numCourses):
        if not dfs(course):
            return False
    return True

```



## # Decode Ways

Solved 

Medium

A string consisting of uppercase english characters can be encoded to a number using the following mapping:

```java
'A' -> "1"
'B' -> "2"
...
'Z' -> "26"
```

Copy

To **decode** a message, digits must be grouped and then mapped back into letters using the reverse of the mapping above. There may be multiple ways to decode a message. For example, `"1012"` can be mapped into:

- `"JAB"` with the grouping `(10 1 2)`
- `"JL"` with the grouping `(10 12)`

The grouping `(1 01 2)` is invalid because `01` cannot be mapped into a letter since it contains a leading zero.

Given a string `s` containing only digits, return the number of ways to **decode** it. You can assume that the answer fits in a **32-bit** integer.

**Example 1:**

```java
Input: s = "12"

Output: 2

Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).
```

Copy

### Solution
- Complexity: O(N)
- Space: O(N)


```python
class Solution:
    def numDecodings(self, s: str) -> int:
        if not s or s[0] == '0':
            return 0

        n = len(s)
        dp = [0] * (n + 1)
        dp[0] = 1
        dp[1] = 1

        for i in range(2, n + 1):
            if 1 <= int(s[i - 1]) <= 9:
                dp[i] += dp[i - 1]
            if 10 <= int(s[i - 2:i]) <= 26:
                dp[i] += dp[i - 2]

        return dp[-1]
```



## Spiral traversing of matrix


```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        
        m = len(matrix)
        n = len(matrix[0])

        res = []
        row_st = 0
        row_en = m - 1
        col_st = 0
        col_en = n - 1

        while (row_st <= row_en) and (col_st <= col_en):
            for col in range(col_st, col_en + 1):
                res.append(matrix[row_st][col])
            row_st += 1

            for row in range(row_st, row_en + 1):
                res.append(matrix[row][col_en])
            col_en -= 1

            if row_st <= row_en:
                for col in range(col_en, col_st - 1, -1):
                    res.append(matrix[row_en][col])
                row_en -= 1

            if col_st <= col_en:
                for row in range(row_en, row_st - 1, -1):
                    res.append(matrix[row][col_st])
                col_st += 1

        return res

```


## # Counting Bits

Easy

Given an integer `n`, count the number of `1`'s in the binary representation of every number in the range `[0, n]`.

Return an array `output` where `output[i]` is the number of `1`'s in the binary representation of `i`.

**Example 1:**

```java
Input: n = 4

Output: [0,1,1,2,1]
```

Copy

Explanation:  
0 --> 0  
1 --> 1  
2 --> 10  
3 --> 11  
4 --> 100

### Solution: DP approach

- Time complextiy: O(n2)

```python
from typing import List

class Solution:
    def countBits(self, n: int) -> List[int]:
        # Initialize a list to store the number of 1's for each number from 0 to n
        dp = [0] * (n + 1)

        # 'offset' tracks the largest power of 2 less than or equal to i
        offset = 1

        for i in range(1, n + 1):
            # When i reaches the next power of 2, update the offset
            if offset * 2 == i:
                offset = i

            # The number of 1's in i is 1 + number of 1's in (i - offset)
            dp[i] = 1 + dp[i - offset]

        return dp

```

- TIme complexity: O(nlog(n))

```python
from bisect import bisect_left
from typing import List

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        sub = []
        for num in nums:
            i = bisect_left(sub, num)

            # If num is greater than any element in sub
            if i == len(sub):
                sub.append(num)
            # Otherwise, replace the first element in sub greater than or equal to num
            else:
                sub[i] = num

            print(sub)  # Debug: Current state of 'sub'
        
        return len(sub)

```


## # Palindromic Substrings

Solved 

Medium

Given a string `s`, return the number of substrings within `s` that are palindromes.

A **palindrome** is a string that reads the same forward and backward.

**Example 1:**

```java
Input: s = "abc"

Output: 3
```

Copy

Explanation: "a", "b", "c".

**Example 2:**

```java
Input: s = "aaa"

Output: 6
```


```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        total_palindromes = 0

        # Expand around each character (odd-length palindromes)
        for i in range(n):
            left = right = i
            while left >= 0 and right < n and s[left] == s[right]:
                total_palindromes += 1
                left -= 1
                right += 1

        # Expand between characters (even-length palindromes)
        for i in range(n - 1):
            left, right = i, i + 1
            while left >= 0 and right < n and s[left] == s[right]:
                total_palindromes += 1
                left -= 1
                right += 1

        return total_palindromes
```