import java.util.*;

public class LeetcodeList {
    //    给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
//
//    你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i])) {
                return new int[]{i, map.get(target - nums[i])};
            }
        }
        throw new IllegalArgumentException("No Data");
    }

    //    给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。
//    求在该柱状图中，能够勾勒出来的矩形的最大面积。
//    以上是柱状图的示例，其中每个柱子的宽度为 1，给定的高度为 [2,1,5,6,2,3]。
//    图中阴影部分为所能勾勒出的最大矩形面积，其面积为 10 个单位。
//    示例:
//    输入: [2,1,5,6,2,3]
//    输出: 10
//    public int largestRectangleArea(int[] heights) {
//        int [] area = new int[heights.length];
//        int res = 0;
//        for(int i=0;i<heights.length;i++){
//            int w = heights[i];
//            for(int j = i ;j<heights.length;j++){
//               w = Math.min(w,heights[j]);
//               area[i] = Math.max(w * (j-i+1),area[i]);
//            }
//        }
//        for(int a=0;a<area.length;a++){
//           res = Math.max(area[a],res);
//        }
//        return res;
//    }
    public int largestRectangleArea(int[] heights) {
        int n = heights.length;
        int[] left = new int[n];
        int[] right = new int[n];

        Stack<Integer> mono_stack = new Stack<>();
        for (int i = 0; i < n; ++i) {
            while (!mono_stack.isEmpty() && heights[mono_stack.peek()] >= heights[i]) {
                mono_stack.pop();
            }
            left[i] = (mono_stack.isEmpty() ? -1 : mono_stack.peek());
            mono_stack.push(i);
        }

        mono_stack.clear();
        for (int i = n - 1; i >= 0; --i) {
            while (!mono_stack.isEmpty() && heights[mono_stack.peek()] >= heights[i]) {
                mono_stack.pop();
            }
            right[i] = (mono_stack.isEmpty() ? n : mono_stack.peek());
            mono_stack.push(i);
        }

        int ans = 0;
        for (int i = 0; i < n; ++i) {
            ans = Math.max(ans, (right[i] - left[i] - 1) * heights[i]);
        }
        return ans;
    }

    //    给你一个数组 candies 和一个整数 extraCandies ，其中 candies[i] 代表第 i 个孩子拥有的糖果数目。
//    对每一个孩子，检查是否存在一种方案，将额外的 extraCandies 个糖果分配给孩子们之后，此孩子有 最多 的糖果。注意，允许有多个孩子同时拥有 最多 的糖果数目
//    示例 1：
//    输入：candies = [2,3,5,1,3], extraCandies = 3
//    输出：[true,true,true,false,true]
//    解释：
//    孩子 1 有 2 个糖果，如果他得到所有额外的糖果（3个），那么他总共有 5 个糖果，他将成为拥有最多糖果的孩子。
//    孩子 2 有 3 个糖果，如果他得到至少 2 个额外糖果，那么他将成为拥有最多糖果的孩子。
//    孩子 3 有 5 个糖果，他已经是拥有最多糖果的孩子。
//    孩子 4 有 1 个糖果，即使他得到所有额外的糖果，他也只有 4 个糖果，无法成为拥有糖果最多的孩子。
//    孩子 5 有 3 个糖果，如果他得到至少 2 个额外糖果，那么他将成为拥有最多糖果的孩子。
//    示例 2：
//    输入：candies = [4,2,1,1,2], extraCandies = 1
//    输出：[true,false,false,false,false]
//    解释：只有 1 个额外糖果，所以不管额外糖果给谁，只有孩子 1 可以成为拥有糖果最多的孩子。
//    示例 3：
//    输入：candies = [12,1,12], extraCandies = 10
//    输出：[true,false,true]
    public List<Boolean> kidsWithCandies(int[] candies, int extraCandies) {
        int max = 0;
        List<Boolean> res = new ArrayList<>();
        for (int candy : candies) {
            max = Math.max(max, candy);
        }
        for (int candy : candies) {
            if (candy + extraCandies >= max) {
                res.add(true);
            } else {
                res.add(false);
            }
        }
        return res;
    }

    //
//    求 1+2+...+n ，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
//    示例 1：
//    输入: n = 3
//    输出: 6
//    示例 2：
//    输入: n = 9
//    输出: 45
//    限制：1 <= n <= 10000
    public int sumNums(int n) {
        boolean flag = n > 0 && (n += sumNums(n - 1)) > 0;
        return n;
    }

    //    爱丽丝参与一个大致基于纸牌游戏 “21点” 规则的游戏，描述如下：
//    爱丽丝以 0 分开始，并在她的得分少于 K 分时抽取数字。 抽取时，她从 [1, W] 的范围中随机获得一个整数作为分数进行累计，其中 W 是整数。 每次抽取都是独立的，其结果具有相同的概率。
//    当爱丽丝获得不少于 K 分时，她就停止抽取数字。 爱丽丝的分数不超过 N 的概率是多少？
//    示例 1：
//    输入：N = 10, K = 1, W = 10
//    输出：1.00000
//    说明：爱丽丝得到一张卡，然后停止。
//    示例 2：
//    输入：N = 6, K = 1, W = 10
//    输出：0.60000
//    说明：爱丽丝得到一张卡，然后停止。
//    在 W = 10 的 6 种可能下，她的得分不超过 N = 6 分。
//    示例 3：
//    输入：N = 21, K = 17, W = 10
//    输出：0.73278
//    提示：
//            0 <= K <= N <= 10000
//            1 <= W <= 10000
//    如果答案与正确答案的误差不超过 10^-5，则该答案将被视为正确答案通过。
//    此问题的判断限制时间已经减少。
    public double new21Game(int N, int K, int W) {
//        if (K == 0) {
//            return 1.0;
//        }
//        double[] dp = new double[K + W + 1];
//        for (int i = K; i <= N && i < K + W; i++) {
//            dp[i] = 1.0;
//        }
//        for (int i = K - 1; i >= 0; i--) {
//            for (int j = 1; j <= W; j++) {
//                dp[i] += dp[i + j] / W;
//            }
//        }
//        return dp[0];
        double[] dp = new double[N + 1];
        double sum = 0;
        dp[0] = 1;
        if (K > 0) sum += 1;
        for (int i = 1; i <= N; i++) {
            dp[i] = sum / W;
            if (i < K) sum += dp[i];
            if (i >= W) sum -= dp[i - W];
        }
        double ans = 0;
        for (int i = K; i <= N; i++) ans += dp[i];
        return ans;
    }

    //    给你一个长度为 n 的整数数组 nums，其中 n > 1，返回输出数组 output ，其中 output[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积。
//    示例:
//    输入: [1,2,3,4]
//    输出: [24,12,8,6]
//    提示：题目数据保证数组之中任意元素的全部前缀元素和后缀（甚至是整个数组）的乘积都在 32 位整数范围内。
//    说明: 请不要使用除法，且在 O(n) 时间复杂度内完成此题。
//    进阶：
//    你可以在常数空间复杂度内完成这个题目吗？（ 出于对空间复杂度分析的目的，输出数组不被视为额外空间。）
    public int[] productExceptSelf(int[] nums) {

        int[] res = new int[nums.length];
        int r = 1;
        res[0] = 1;
        //从左向右遍历，以此乘以前一位的nums的值（第一个为1），得到左侧前缀
        for (int i = 1; i < nums.length; i++) {
            res[i] = res[i - 1] * nums[i - 1];
        }
        //从右向左遍历，结果数组乘右后缀为答案
        for (int j = nums.length - 1; j >= 0; j--) {
            res[j] = r * res[j];
            r = r * nums[j];
        }
        return res;
    }

    //    输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。
//    示例 1：
//    输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
//    输出：[1,2,3,6,9,8,7,4,5]
//    示例 2：
//    输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
//    输出：[1,2,3,4,8,12,11,10,9,5,6,7]
//    限制：
//            0 <= matrix.length <= 100
//            0 <= matrix[i].length <= 100
    public int[] spiralOrder(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return new int[0];
        }
        boolean[][] visited = new boolean[matrix.length][matrix[0].length];
        int total = matrix.length * matrix[0].length;
        int[] res = new int[total];
        int row = 0, column = 0;
        int[][] directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        int directionIndex = 0;
        for (int i = 0; i < total; i++) {
            res[i] = matrix[row][column];
            visited[row][column] = true;
            int nextRow = row + directions[directionIndex][0], nextColumn = column + directions[directionIndex][1];
            if (nextRow < 0 || nextRow >= matrix.length || nextColumn < 0 || nextColumn >= matrix[0].length || visited[nextRow][nextColumn]) {
                directionIndex = (directionIndex + 1) % 4;
            }
            row += directions[directionIndex][0];
            column += directions[directionIndex][1];
        }
        return res;
    }

//    给定一个未排序的整数数组，找出最长连续序列的长度。
//    要求算法的时间复杂度为 O(n)。
//    示例:
//    输入: [100, 4, 200, 1, 3, 2]
//    输出: 4
//    解释: 最长连续序列是 [1, 2, 3, 4]。它的长度为 4。
    public int longestConsecutive(int[] nums) {
        Set<Integer> resSet = new HashSet<>();
        int res = 0;
        for(int num: nums){
            resSet.add(num);
        }
        if(!resSet.isEmpty()){
            int target = 0;
            int temp = 1;
            for(int num: resSet){
                target = num + 1;
                while(resSet.contains(target)){
                    target = target + 1;
                    temp ++;
                }
                res = Math.max(temp,res);
                temp = 1;
            }
        }
        return res;
    }

//    给定两个单词（beginWord 和 endWord）和一个字典 wordList，找出所有从 beginWord 到 endWord 的最短转换序列。转换需遵循如下规则：
//    每次转换只能改变一个字母。
//    转换过程中的中间单词必须是字典中的单词。
//    说明:
//    如果不存在这样的转换序列，返回一个空列表。
//    所有单词具有相同的长度。
//    所有单词只由小写字母组成。
//    字典中不存在重复的单词。
//    你可以假设 beginWord 和 endWord 是非空的，且二者不相同。
//    示例 1:
//    输入:
//    beginWord = "hit",
//    endWord = "cog",
//    wordList = ["hot","dot","dog","lot","log","cog"]
//    输出:
//            [
//            ["hit","hot","dot","dog","cog"],
//              ["hit","hot","lot","log","cog"]
//            ]
//    示例 2:
//    输入:
//    beginWord = "hit"
//    endWord = "cog"
//    wordList = ["hot","dot","dog","lot","log"]
//    输出: []
//    解释: endWord "cog" 不在字典中，所以不存在符合要求的转换序列。
//    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
//
//    }

    //    给定两个字符串 s 和 t，它们只包含小写字母。
//    字符串 t 由字符串 s 随机重排，然后在随机位置添加一个字母。
//    请找出在 t 中被添加的字母。
    public char findTheDifference(String s, String t) {
        char diff = 'a';
        char[] sArr = s.toCharArray();
        char[] tArr = t.toCharArray();
        int sumSArr = 0;
        int sumTArr = 0;
        for (int i = 0; i < sArr.length; i++) {
            int sChar = sArr[i];
            int tChar = tArr[i];
            sumSArr += sChar;
            sumTArr += tChar;
        }
        diff = (char) (sumTArr - sumSArr + tArr[tArr.length - 1]);
        return diff;
    }

    //    数组的每个索引作为一个阶梯，第 i个阶梯对应着一个非负数的体力花费值 cost[i](索引从0开始)。
//    每当你爬上一个阶梯你都要花费对应的体力花费值，然后你可以选择继续爬一个阶梯或者爬两个阶梯。
//    您需要找到达到楼层顶部的最低花费。在开始时，你可以选择从索引为 0 或 1 的元素作为初始阶梯。
//    输入: cost = [10, 15, 20]
//    输出: 15
//    解释: 最低花费是从cost[1]开始，然后走两步即可到阶梯顶，一共花费15。
//    输入: cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
//    输出: 6
//    解释: 最低花费方式是从cost[0]开始，逐个经过那些1，跳过cost[3]，一共花费6。
//    注意：
//    cost 的长度将会在 [2, 1000]。
//    每一个 cost[i] 将会是一个Integer类型，范围为 [0, 999]。
    public int minCostClimbingStairs(int[] cost) {
        int len = cost.length;
        int prev = 0;
        int curr = 0;
        for (int i = 2; i <= len; i++) {
            int next = Math.min(prev + cost[i - 2], curr + cost[i - 1]);
            prev = curr;
            curr = next;
        }
        return curr;
    }

    //    4.给定两个大小为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的中位数
//    示例 1：
//    输入：nums1 = [1,3], nums2 = [2]
//    输出：2.00000
//    解释：合并数组 = [1,2,3] ，中位数 2
//    示例 2：
//    输入：nums1 = [1,2], nums2 = [3,4]
//    输出：2.50000
//    解释：合并数组 = [1,2,3,4] ，中位数 (2 + 3) / 2 = 2.5
//    示例 3：
//    输入：nums1 = [0,0], nums2 = [0,0]
//    输出：0.00000
//    示例 4：
//    输入：nums1 = [], nums2 = [1]
//    输出：1.00000
//    示例 5：
//    输入：nums1 = [2], nums2 = []
//    输出：2.00000
//    提示：
//    nums1.length == m
//    nums2.length == n
//    0 <= m <= 1000
//    0 <= n <= 1000
//    1 <= m + n <= 2000
//    -10^6 <= nums1[i], nums2[i] <= 10^6
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        double num = 0.00000;
        int m = nums1.length;
        int n = nums2.length;


        return num;
    }

    //  给你一个二叉树，请你返回其按 层序遍历 得到的节点值。 （即逐层地，从左到右访问所有节点）。
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> list = new ArrayList<>();
        traversalForTree(root, list, 0);
        return list;
    }

    private void traversalForTree(TreeNode root, List<List<Integer>> list, int level) {
        if (root == null) {
            return;
        }
        if (level == list.size()) {
            list.add(new ArrayList<Integer>());
        }
        list.get(level).add(root.val);
        traversalForTree(root.left, list, level + 1);
        traversalForTree(root.right, list, level + 1);
    }

    //  给定一个二叉树，返回其节点值的锯齿形层序遍历。（即先从左往右，再从右往左进行下一层遍历，以此类推，层与层之间交替进行）。
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> list = new ArrayList<>();
        traversal(root, list, 0);
        return list;
    }

    private void traversal(TreeNode root, List<List<Integer>> list, int level) {
        if (root == null) {
            return;
        }
        if (level == list.size()) {
            list.add(new ArrayList<Integer>());
        }
        if ((level & 1) == 1) {
            list.get(level).add(0, root.val);
        } else {
            list.get(level).add(root.val);
        }
        traversal(root.left, list, level + 1);
        traversal(root.right, list, level + 1);
    }

    //  387.给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回 -1
    public int firstUniqChar(String s) {
//        Map<Character,Integer> map = new HashMap<>();
//
//        for(int i = 0;i< s.length();i++){
//            if(map.get(s.charAt(i)) == null){
//                map.put(s.charAt(i),1);
//            }else{
//                map.put(s.charAt(i),map.get(s.charAt(i))+1 );
//            }
//        }
//        for(int i = 0;i< s.length();i++){
//          if(map.get(s.charAt(i)) == 1){
//              return i;
//          }
//        }
//        return -1;
        Queue<Pair> queue = new LinkedList<>();
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            if (!map.containsKey(s.charAt(i))) {
                map.put(s.charAt(i), i);
                queue.offer(new Pair(s.charAt(i), i));
            } else {
                map.put(s.charAt(i), -1);
                while (!queue.isEmpty() && map.get(queue.peek().ch) == -1) {
                    queue.poll();
                }
            }
        }
        return queue.isEmpty() ? -1 : queue.poll().pos;
    }

    //   135. 老师想给孩子们分发糖果，有 N 个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。
//    你需要按照以下要求，帮助老师给这些孩子分发糖果：
//    每个孩子至少分配到 1 个糖果。
//    相邻的孩子中，评分高的孩子必须获得更多的糖果。
//    那么这样下来，老师至少需要准备多少颗糖果呢？
    public int candy(int[] ratings) {
        List<Integer> list = new ArrayList<>();
        int len = ratings.length;
        int sum = 0;
        for (int i = 0; i < len; i++) {
            list.add(1);
        }
        for (int i = 1; i < len; i++) {
            if (ratings[i] > ratings[i - 1] && list.get(i) <= list.get(i - 1)) {
                list.set(i, list.get(i - 1) + 1);
            }

        }
        for (int i = len - 1; i >= 1; i--) {
            if (ratings[i - 1] > ratings[i] && list.get(i - 1) <= list.get(i)) {
                list.set(i - 1, list.get(i) + 1);
            }
        }
        for (int i = 0; i < list.size(); i++) {
            sum += list.get(i);
        }

        return sum;
    }

    //    34.给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。
//    如果数组中不存在目标值 target，返回 [-1, -1]。
//    进阶：
//    你可以设计并实现时间复杂度为 O(log n) 的算法解决此问题吗？
    public int[] searchRange(int[] nums, int target) {
        int[] res = new int[]{-1, -1};
        res[0] = binarySearch(nums, target, true);
        res[1] = binarySearch(nums, target, false);
        return res;
    }

    private int binarySearch(int[] nums, int target, boolean leftOrRight) {
        int res = -1;
        int left = 0, right = nums.length - 1, mid;
        while (left <= right) {
            mid = left + (right - left) / 2;
            if (target < nums[mid])
                right = mid - 1;
            else if (target > nums[mid])
                left = mid + 1;
            else {
                res = mid;
                //处理target == nums[mid]
                if (leftOrRight)
                    right = mid - 1;
                else
                    left = mid + 1;
            }
        }
        return res;
    }

    //    455.假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。
//    对每个孩子 i，都有一个胃口值 g[i]，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 j，都有一个尺寸 s[j] 。如果 s[j] >= g[i]，我们可以将这个饼干 j 分配给孩子 i ，这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。
    public int findContentChildren(int[] g, int[] s) {
        Arrays.parallelSort(g);
        Arrays.parallelSort(s);
        int numsOfChild = g.length;
        int numsOfCookie = s.length;
        int count = 0;
        for (int i = 0, j = 0; i < numsOfChild && j < numsOfCookie; i++, j++) {
            while (j < numsOfCookie && g[i] > s[j]) {
                j++;
            }
            if (j < numsOfCookie) {
                count++;
            }
        }
        return count;
    }
//    204.统计所有小于非负整数 n 的质数的数量。


//    188.给定一个整数数组 prices ，它的第 i 个元素 prices[i] 是一支给定的股票在第 i 天的价格。
//    设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。
//    注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）

    //    1046.有一堆石头，每块石头的重量都是正整数。
//    每一回合，从中选出两块 最重的 石头，然后将它们一起粉碎。假设石头的重量分别为 x 和 y，且 x <= y。那么粉碎的可能结果如下：
//    如果 x == y，那么两块石头都会被完全粉碎；
//    如果 x != y，那么重量为 x 的石头将会完全粉碎，而重量为 y 的石头新重量为 y-x。
//    最后，最多只会剩下一块石头。返回此石头的重量。如果没有石头剩下，就返回 0。
//    public int lastStoneWeight(int[] stones) {
//
//    }

}
