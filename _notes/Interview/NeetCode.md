
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