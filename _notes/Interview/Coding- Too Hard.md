
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

## [Robot Room Cleaner](https://leetcode.com/problems/robot-room-cleaner/)

Solved

Hard

Topics

![conpanies icon](attachments/4b834bdc05b266826a5de87b0fefe59c_MD5.svg)Companies

You are controlling a robot that is located somewhere in a room. The room is modeled as an `m x n` binary grid where `0` represents a wall and `1` represents an empty slot.

The robot starts at an unknown location in the room that is guaranteed to be empty, and you do not have access to the grid, but you can move the robot using the given API `Robot`.

You are tasked to use the robot to clean the entire room (i.e., clean every empty cell in the room). The robot with the four given APIs can move forward, turn left, or turn right. Each turn is `90` degrees.

When the robot tries to move into a wall cell, its bumper sensor detects the obstacle, and it stays on the current cell.

Design an algorithm to clean the entire room using the following APIs:

```python
class Solution:
    def cleanRoom(self, robot):
        visited = set()
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left

        def go_back():
            robot.turnLeft()
            robot.turnLeft()
            robot.move()
            robot.turnLeft()
            robot.turnLeft()

        def dfs(x, y, d):
            robot.clean()
            visited.add((x, y))

            for i in range(4):
                new_d = (d + i) % 4
                dx, dy = directions[new_d]
                nx, ny = x + dx, y + dy

                if (nx, ny) not in visited:
                    if robot.move():
                        dfs(nx, ny, new_d)
                        go_back()
                robot.turnRight()

        dfs(0, 0, 0)

```

##   Minimum Cost to Hire K Workers



There are `n` workers. You are given two integer arrays `quality` and `wage` where `quality[i]` is the quality of the `ith` worker and `wage[i]` is the minimum wage expectation for the `ith` worker.

We want to hire exactly `k` workers to form a **paid group**. To hire a group of `k` workers, we must pay them according to the following rules:

1. Every worker in the paid group must be paid at least their minimum wage expectation.
2. In the group, each worker's pay must be directly proportional to their quality. This means if a worker’s quality is double that of another worker in the group, then they must be paid twice as much as the other worker.

Given the integer `k`, return _the least amount of money needed to form a paid group satisfying the above conditions_. Answers within `10-5` of the actual answer will be accepted.

**Example 1:**

**Input:** quality = [10,20,5], wage = [70,50,30], k = 2
**Output:** 105.00000
**Explanation:** We pay 70 to 0th worker and 35 to 2nd worker.


Solution: First sort by cost/quality and use heap to keep track of min cost
- Ratio increasing in the loop so need to minimize total quality using heap (pop the largest)

```python
class Solution:
    def mincostToHireWorkers(self, quality: List[int], wage: List[int], k: int) -> float:
        
        wage_per_quality = [w/q for (w,q) in zip(wage, quality)]
        
        sorted_quality = sorted(zip(wage_per_quality, quality))
        
        heap = []
        min_cost = float(inf)
        total_quality = 0
        for ratio, q in sorted_quality:
            heapq.heappush(heap, -q)
            total_quality += q
            
            if len(heap) > k:
                total_quality += heapq.heappop(heap)
        
            if len(heap) == k:
                min_cost = min(min_cost, total_quality*ratio)
                
        return min_cost
```


## Odd Even Jump

You are given an integer array `arr`. From some starting index, you can make a series of jumps. The (1st, 3rd, 5th, ...) jumps in the series are called **odd-numbered jumps**, and the (2nd, 4th, 6th, ...) jumps in the series are called **even-numbered jumps**. Note that the **jumps** are numbered, not the indices.

You may jump forward from index `i` to index `j` (with `i < j`) in the following way:

- During **odd-numbered jumps** (i.e., jumps 1, 3, 5, ...), you jump to the index `j` such that `arr[i] <= arr[j]` and `arr[j]` is the smallest possible value. If there are multiple such indices `j`, you can only jump to the **smallest** such index `j`.
- During **even-numbered jumps** (i.e., jumps 2, 4, 6, ...), you jump to the index `j` such that `arr[i] >= arr[j]` and `arr[j]` is the largest possible value. If there are multiple such indices `j`, you can only jump to the **smallest** such index `j`.
- It may be the case that for some index `i`, there are no legal jumps.

A starting index is **good** if, starting from that index, you can reach the end of the array (index `arr.length - 1`) by jumping some number of times (possibly 0 or more than once).

Return _the number of **good** starting indices_.

