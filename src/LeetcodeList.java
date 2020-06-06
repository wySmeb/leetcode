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

        Stack<Integer> mono_stack = new Stack<Integer>();
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
}
