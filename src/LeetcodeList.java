import java.util.*;

public class LeetcodeList {
//    给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。
//
//    你可以假设每种输入只会对应一个答案。但是，你不能重复利用这个数组中同样的元素。
    public int[] twoSum(int[] nums, int target) {
         Map<Integer,Integer> map = new HashMap<Integer,Integer>();
         for(int i= 0 ; i< nums.length; i++){
            if(map.containsKey(target-nums[i])){
                return new int[] {i,map.get(target-nums[i])};
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
        for(int candy:candies){
            max = Math.max(max,candy);
        }
        for(int candy:candies){
            if(candy + extraCandies >= max){
                res.add(true);
            }else{
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
        boolean flag = n > 0 && (n += sumNums(n-1)) >0 ;
        return n;
    }



}
