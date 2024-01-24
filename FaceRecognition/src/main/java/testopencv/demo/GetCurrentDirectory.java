package testopencv.demo;

public class GetCurrentDirectory
{
    public static void main(String[] args) {
        // 获取当前工作目录
        String currentDirectory = System.getProperty("user.dir");
        System.out.println("Current Directory: " + currentDirectory+"\\src\\main\\java\\tesropencv.demo");
    }
}