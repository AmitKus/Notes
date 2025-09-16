---
---

# Coding Interview Prep - Problem Patterns & Solutions

## 🎯 Sliding Window Pattern

### Core Technique
Maintain a window with two pointers (left/right) that expands and contracts based on conditions.

### Problems

#### **Longest Substring Without Repeating Characters**
- **Approach**: Track char indices, move left pointer when duplicate found
- **Key**: `left = max(left, lastIndex[char] + 1)`
- **Time**: O(n), **Space**: O(min(n, charset))

#### **Longest Substring with K Distinct Characters**
- **Approach**: Use frequency map, shrink window when distinct > k
- **Key**: Remove chars from left until valid again
- **Time**: O(n)

#### **Minimum Window Substring**
- **Approach**: Expand right to find valid window, contract left to minimize
- **Key**: Track `have` vs `need` character counts
- **Time**: O(|s| + |t|)

#### **Longest Repeating Character Replacement**
- **Approach**: Track max frequency char, window valid if `windowSize - maxFreq ≤ k`
- **Time**: O(n)

---

## 🌳 Tree Patterns

### Core Traversals
```python
# Inorder: Left → Root → Right (BST gives sorted)
# Preorder: Root → Left → Right
# Postorder: Left → Right → Root
```

### Key Problems

#### **Binary Tree Maximum Path Sum**
- **Pattern**: At each node, calculate path through node vs path to parent
- **Key**: `maxPath = node.val + left + right` but return `node.val + max(left, right)`
- **Time**: O(n)

#### **Diameter of Binary Tree**
- **Pattern**: Similar to max path sum, track max(left_depth + right_depth)
- **Time**: O(n)

#### **Validate BST**
- **Pattern**: Pass min/max bounds down recursively
- **Key**: `helper(node, minVal, maxVal)`
- **Time**: O(n)

#### **Lowest Common Ancestor (BST)**
- **Pattern**: Find split point where p.val ≤ node.val ≤ q.val
- **Time**: O(h)

#### **Serialize/Deserialize Binary Tree**
- **Pattern**: Level-order BFS with queue, use delimiter for nulls
- **Time**: O(n)

#### **Construct Tree from Traversals**
- **Pattern**: Use preorder for roots, inorder for left/right splits
- **Key**: First element in preorder is always root
- **Time**: O(n²) naive, O(n) with hashmap

---

## 🔗 Linked List Patterns

### Core Techniques
- **Fast/Slow Pointers**: Detect cycles, find middle
- **Dummy Head**: Simplify edge cases
- **Reversal**: Three pointers (prev, curr, next)

### Key Problems

#### **Reorder List (L0→Ln→L1→Ln-1...)**
1. Find middle using slow/fast pointers
2. Reverse second half
3. Merge alternately
- **Time**: O(n)

#### **Add Two Numbers**
- **Pattern**: Traverse both lists with carry
- **Key**: Handle different lengths and final carry
- **Time**: O(max(m,n))

#### **Merge K Sorted Lists**
- **Pattern**: Min heap of (value, list_index, node)
- **Time**: O(N log k) where N = total nodes

---

## 🏃 Two Pointers Pattern

### Key Problems

#### **3Sum / 4Sum**
- **Pattern**: Sort array, fix one/two elements, use two pointers for rest
- **Key**: Skip duplicates after finding valid triplet
- **Time**: O(n²) for 3Sum, O(n³) for 4Sum

#### **Trapping Rain Water**
- **Pattern**: Calculate left_max and right_max arrays
- **Key**: `water[i] = min(left_max[i], right_max[i]) - height[i]`
- **Time**: O(n), **Space**: O(n)

---

## 📊 Dynamic Programming Patterns

### Classic Patterns

#### **House Robber (Circular)**
- **Pattern**: Run linear robber twice: houses[0:n-1] and houses[1:n]
- **Recurrence**: `dp[i] = max(dp[i-1], nums[i] + dp[i-2])`

#### **Word Break**
- **Pattern**: Check if substring can be formed using dictionary
- **Key**: Use memoization or DP with word set
- **Time**: O(n² × m) where m = avg word length

#### **Jump Game**
- **Pattern**: Track if position is reachable
- **Key**: `dp[i] = any(dp[j] for j where j+nums[j] >= i)`
- **Time**: O(n²)

#### **Decode Ways**
- **Pattern**: Similar to climbing stairs with conditions
- **Recurrence**: Check 1-digit and 2-digit decodings
- **Time**: O(n)

#### **Longest Common Subsequence**
- **Pattern**: 2D DP matching characters
- **Recurrence**:
  - If match: `dp[i][j] = 1 + dp[i-1][j-1]`
  - Else: `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`

#### **Maximum Subarray (Kadane's)**
- **Pattern**: Track current sum vs starting fresh
- **Key**: `curr_sum = max(num, curr_sum + num)`
- **Time**: O(n)

#### **Maximum Product Subarray**
- **Pattern**: Track both min and max (negatives can flip)
- **Key**: Swap min/max when encountering negative
- **Time**: O(n)

#### **Longest Increasing Subsequence**
- **Pattern**: Binary search + patience sorting
- **Time**: O(n log n)

#### **Palindromic Substrings**
- **Pattern**: Expand around centers (odd and even length)
- **Time**: O(n²)

---

## 🔍 Binary Search Patterns

### Rotated Array Problems

#### **Search in Rotated Sorted Array**
- **Pattern**: Find sorted half, check if target in that half
- **Key**: Compare nums[left] with nums[mid] to find sorted portion

#### **Find Minimum in Rotated Sorted Array**
- **Pattern**: Compare mid with right
- **Key**: If `nums[mid] > nums[right]`, min is in right half

#### **Median of Two Sorted Arrays**
- **Pattern**: Binary search on smaller array for partition point
- **Time**: O(log(min(m,n)))

---

## 🗺️ Graph Patterns

### Core Algorithms

#### **Number of Islands**
- **Pattern**: DFS/BFS to mark visited cells
- **Key**: Modify grid or use visited set
- **Time**: O(m×n)

#### **Graph Valid Tree**
- **Pattern**: Tree has n-1 edges and all nodes connected
- **Key**: Check cycles and connectivity
- **Time**: O(n)

#### **Course Schedule (Topological Sort)**
- **Pattern**: Track in-degrees, process nodes with 0 in-degree
- **Key**: Use queue for BFS approach
- **Time**: O(V + E)

#### **Word Ladder**
- **Pattern**: BFS with pattern matching (h*t matches hot, hit)
- **Key**: Precompute pattern dictionary
- **Time**: O(M²×N) where M = word length

#### **Alien Dictionary**
- **Pattern**: Build graph from word ordering, topological sort
- **Key**: Compare adjacent words for character ordering
- **Time**: O(C) where C = total characters

---

## 🎨 Matrix Patterns

### Key Problems

#### **Rotate Image (90° clockwise)**
- **Pattern**: Transpose + reverse rows OR layer by layer rotation
- **Time**: O(n²)

#### **Spiral Matrix**
- **Pattern**: Four boundaries that shrink inward
- **Key**: Handle single row/column edge cases

#### **Word Search**
- **Pattern**: DFS with backtracking
- **Key**: Mark visited in-place with '#'
- **Time**: O(m×n×4^L) where L = word length

#### **Longest Increasing Path**
- **Pattern**: DFS with memoization from each cell
- **Key**: No need to track visited (increasing constraint prevents cycles)
- **Time**: O(m×n)

---

## 🔧 Data Structure Design

### LRU Cache
- **Pattern**: HashMap + Doubly Linked List OR OrderedDict
- **Key**: O(1) get and put operations

### Min Stack
- **Pattern**: Store (value, current_min) pairs
- **Key**: Each element tracks min at that point

### Trie (Prefix Tree)
- **Pattern**: Nested dictionaries with 'end' marker
- **Use Cases**: Autocomplete, word search, add/search with wildcards

### Find Median from Data Stream
- **Pattern**: Two heaps - max heap (lower half) + min heap (upper half)
- **Key**: Balance heaps so sizes differ by at most 1

---

## 🧩 Backtracking Patterns

### Core Template
```python
def backtrack(path, choices):
    if is_solution(path):
        results.append(path[:])
        return
    for choice in choices:
        if is_valid(choice):
            path.append(choice)
            backtrack(path, new_choices)
            path.pop()  # undo
```

### Key Problems

#### **Generate Parentheses**
- **Constraints**: open < n, close < open
- **Time**: O(4^n / √n) - Catalan number

#### **Combination Sum**
- **Pattern**: Include element multiple times or skip
- **Key**: Avoid duplicates by not going backwards

#### **Word Squares**
- **Pattern**: Build prefix dictionary, backtrack row by row
- **Key**: Next word must start with column letters

---

## 🔑 Key Insights & Tips

### Time Complexity Patterns
- Single pass with fixed operations: O(n)
- Nested loops: O(n²)
- Divide and conquer: O(n log n)
- Backtracking/subsets: O(2^n)
- Permutations: O(n!)

### Space Optimization
- Use input array for marking visited (DFS/BFS)
- Rolling array for DP (only need previous row)
- Morris traversal for trees (O(1) space)

### Common Tricks
- **Sort first**: Simplifies many problems (3Sum, intervals)
- **Dummy nodes**: Simplifies linked list edge cases
- **Precomputation**: Build lookup tables/prefix sums
- **Multiple passes**: Sometimes simpler than one complex pass
- **Binary search**: On answer space, not just sorted arrays

### Interview Tips
1. **Clarify**: Input constraints, edge cases, duplicates
2. **Approach**: Explain brute force, then optimize
3. **Code**: Write clean, modular code with clear variable names
4. **Test**: Walk through example, check edge cases
5. **Optimize**: Discuss time/space tradeoffs

---

## 📈 Complexity Cheat Sheet

| Pattern | Time | Space | When to Use |
|---------|------|-------|-------------|
| Sliding Window | O(n) | O(k) | Substring/subarray with condition |
| Two Pointers | O(n) | O(1) | Sorted array, palindrome |
| Fast/Slow Pointers | O(n) | O(1) | Cycle detection, middle element |
| Tree DFS | O(n) | O(h) | Path problems, subtree problems |
| Tree BFS | O(n) | O(w) | Level order, shortest path |
| Graph DFS/BFS | O(V+E) | O(V) | Connected components, shortest path |
| Dynamic Programming | O(n²) | O(n) | Optimal substructure |
| Binary Search | O(log n) | O(1) | Sorted/rotated array |
| Heap | O(n log k) | O(k) | K largest/smallest |
| Trie | O(m) | O(ALPHABET × m × n) | Prefix matching |
| Union Find | O(α(n)) | O(n) | Connected components |