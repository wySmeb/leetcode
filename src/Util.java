import java.math.BigDecimal;

public class Util {
    /**
     * @param num            待转换的double类型数据
     * @param reservedLength 二进制保留几位小数
     * @return 二进制字符串
     */
    public String doubleToBinaryString(double num, int reservedLength) {
        if (reservedLength < 0) {
            return "请保留正确位数";
        }
        StringBuffer sb = new StringBuffer();
        if (Double.toString(num).contains(".")) {
            //包含小数部分的处理逻辑
            String intStr = Double.toString(num).split("\\.")[0];
            String doubleStr = Double.toString(num).split("\\.")[1];
            Integer a = Integer.parseInt(intStr);
            String intBinStr = Integer.toBinaryString(a);
            sb.append(intBinStr);
            if (reservedLength > 0) {
                sb.append(".");
                BigDecimal bd = BigDecimal.valueOf(Double.parseDouble("0." + doubleStr));
                BigDecimal std = BigDecimal.valueOf(1);
                for (int i = 1; i <= reservedLength; i++) {
                    BigDecimal temp = bd.multiply(BigDecimal.valueOf(2));
                    if (temp.compareTo(std) == -1) {
                        //temp < 1
                        sb.append("0");
                        bd = temp;
                    } else {
                        sb.append("1");
                        bd = temp.subtract(std);
                    }
                }
            }
        } else {
            String intStr = Double.toString(num);
            Integer a = Integer.parseInt(intStr);
            String intBinStr = Integer.toBinaryString(a);
            sb.append(intBinStr);
        }
        return sb.toString();
    }


}
