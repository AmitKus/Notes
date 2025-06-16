
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

