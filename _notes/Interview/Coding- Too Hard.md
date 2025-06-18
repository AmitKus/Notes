
## Split Array Largest Sum [ This is tricky!]

Solution

Given an integer array `nums` and an integer `k`, split `nums` into `k` non-empty subarrays such that the largest sum of any subarray is **minimized**.

Return _the minimized largest sum of the split_.

A **subarray** is a contiguous part of the array.

**Example 1:**

**Input:** nums = [7,2,5,10,8], k = 2
**Output:** 18
**Explanation:** There are four ways to split nums into two subarrays.
The best way is to split it into [7,2,5] and [10,8], where the largest sum among the two subarrays is only 18.



#### Solution
- Binary search between max(nums) and sum(nums)
- Check if can be done with k splits
```python
class Solution:
    def splitArray(self, nums: List[int], k: int) -> int:
        
        def can_split(max_sum_allowed):
            
            current_sum = 0
            count = 1 
            for num in nums:
                
                if current_sum + num > max_sum_allowed:
                    count += 1
                    current_sum = num
                else:
                    current_sum += num
        
            return count <= k
        
        left, right = max(nums), sum(nums)
        while left < right:
            
            mid = (left + right)//2
            
            if can_split(mid):
                right = mid
            else:
                left = mid + 1
                
        return left
```

- DP formula
- Let `dp[i][k]` = minimum largest sum when splitting the first `i` elements into `k` subarrays.
$dp[i][k] = min(
    max(dp[j][k-1], sum(nums[j:i]))
    for j in range(k-1, i)
)$


## Count of Smaller Numbers After Self

Solution

Given an integer array `nums`, return _an integer array_ `counts` _where_ `counts[i]` _is the number of smaller elements to the right of_ `nums[i]`.

**Example 1:**

**Input:** nums = [5,2,6,1]
**Output:** [2,1,1,0]
**Explanation:**
To the right of 5 there are **2** smaller elements (2 and 1).
To the right of 2 there is only **1** smaller element (1).
To the right of 6 there is **1** smaller element (1).
To the right of 1 there is **0** smaller element.

**Example 2:**

**Input:** nums = [-1]
**Output:** [0]

**Example 3:**

**Input:** nums = [-1,-1]
**Output:** [0,0]

**Constraints:**

- `1 <= nums.length <= 105`
- `-104 <= nums[i] <= 104`


- BST for a simple solution but times-out


## Android Unlock Patterns

- Solution:
	- Backtracking
	- Use ```visited = [False]*10```
		- Use 1-9 for tracking and leave 0 unused
	- Symmetry:
		- Starting from 1 times 4 for all corners
		- Starting from 2 times 4 for all middle numbers
		- Starting with 5 times 1 as its unique starting position

```python
class Solution:
    def numberOfPatterns(self, m: int, n: int) -> int:
        
        # Initialize skips
        skip = [[0]*10 for _ in range(10)]
        
        skip[1][3] = skip[3][1] = 2
        skip[4][6] = skip[6][4] = 5
        skip[7][9] = skip[9][7] = 8
        skip[2][8] = skip[8][2] = 5
        skip[1][7] = skip[7][1] = 4
        skip[3][9] = skip[9][3] = 6
        skip[1][9] = skip[9][1] = 5
        skip[3][7] = skip[7][3] = 5
        
        visited = [False]*10
        
        def backtrack(curr: int, length: int) -> int:
            
            if length > n:
                return 0
            count = 0
            if length > m:
                count += 1
                
            visited[curr] = True
            for next in range(1, 10):
                if not visited[next] and (skip[curr][next] == 0 or visited[skip[curr][next]]):
                    count += backtrack(next, length + 1)
            visited[curr] = False
            return count
        
        return (backtrack(1, 1) * 4 + backtrack(2, 1) * 4 + backtrack(5, 1))
```


## Cracking the Safe

There is a safe protected by a password. The password is a sequence of `n` digits where each digit can be in the range `[0, k - 1]`.

The safe has a peculiar way of checking the password. When you enter in a sequence, it checks the **most recent** `n` **digits** that were entered each time you type a digit.

- For example, the correct password is `"345"` and you enter in `"012345"`:
    - After typing `0`, the most recent `3` digits is `"0"`, which is incorrect.
    - After typing `1`, the most recent `3` digits is `"01"`, which is incorrect.
    - After typing `2`, the most recent `3` digits is `"012"`, which is incorrect.
    - After typing `3`, the most recent `3` digits is `"123"`, which is incorrect.
    - After typing `4`, the most recent `3` digits is `"234"`, which is incorrect.
    - After typing `5`, the most recent `3` digits is `"345"`, which is correct and the safe unlocks.

Return _any string of **minimum length** that will unlock the safe **at some point** of entering it_.

**Background**:
- De Bruijn Sequence: size n (digits) and base k (0...k-1)
	- n = 2, k =2
		- 00, 01, 11, 10
**Algo**:
	- Use graph
	- Nodes are the n possible substring (00, 01, 10)
	- Edges only when last n-1 digits append with (0,k-1) transform Node A to Node B
	- Building all combinations:
		- '0', '1'
		- graph['0'] = ['0','1']
		- graph['1'] = ['0','1']

```python
from collections import defaultdict

class Solution:
    def crackSafe(self, n: int, k: int) -> str:

        def all_combinations(n, k):
            digits = [str(i) for i in range(k)]
            results = ['']
            for _ in range(n):
                next_results = []
                for prefix in results:
                    for d in digits:
                        next_results.append(prefix + d)
                results = next_results
            return results

        # nodes are length n - 1
        nodes = all_combinations(n - 1, k)

        graph = defaultdict(list)
        for node in nodes:
            for d in range(k):
                graph[node].append(str(d))

        res = []
        def dfs(node):
            while graph[node]:
                digit = graph[node].pop()
                next_node = node[1:] + digit
                dfs(next_node)
                res.append(digit)  # append AFTER recursion (post-order)

        start = '0' * (n - 1)
        dfs(start)

        return start + ''.join(reversed(res))

```