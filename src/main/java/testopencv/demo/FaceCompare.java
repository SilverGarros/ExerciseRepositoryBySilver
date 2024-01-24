package testopencv.demo;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;

import java.util.Arrays;

public class FaceCompare {

    // 初始化人脸探测器
    static CascadeClassifier faceDetector;

    static {
        // 加载动态库
        System.load("C:\\Users\\Silver\\Desktop\\FaceRecognitionDemo\\src\\main\\resources\\lib\\opencv\\opencv_java490.dll");

        // 引入特征分类器配置文件：haarcascade_frontalface_alt.xml 文件路径
        String property = "G:\\JAVA\\FaceRecognition\\src\\main\\java\\testopencv\\demo\\haarcascade_frontalface_alt.xml";
        System.out.println(property);
        faceDetector = new CascadeClassifier(property);
    }

    public static void main(String[] args) {
        // 打开摄像头
        VideoCapture videoCapture = new VideoCapture(0);
        if (!videoCapture.isOpened()) {
            System.out.println("Error: Camera not opened.");
            return;
        }

        Mat frame = new Mat();
        videoCapture.read(frame);

        long start = System.currentTimeMillis();
        double compareHist = compare_camera(frame);
        System.out.println("time:" + (System.currentTimeMillis() - start));
        System.out.println(compareHist);

        // 释放资源
        videoCapture.release();
    }

    // 灰度化人脸
    public static Mat conv_Mat(Mat frame) {
        Mat image1 = new Mat();
        // 灰度化
        Imgproc.cvtColor(frame, image1, Imgproc.COLOR_BGR2GRAY);
        // 探测人脸
        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(image1, faceDetections);
        // rect中人脸图片的范围
        for (Rect rect : faceDetections.toArray()) {
            Mat face = new Mat(image1, rect);
            return face;
        }
        return null;
    }

    // 比较摄像头图像
    public static double compare_camera(Mat frame) {
        Mat mat_1 = conv_Mat(frame);

        // 检查 mat_1 是否为 null
        if (mat_1 == null) {
            System.out.println("No face detected in the current frame.");
            return 0.0;  // 或者根据需要返回其他默认值
        }

        // 读取下一帧
        VideoCapture videoCapture = new VideoCapture(0);
        Mat nextFrame = new Mat();
        videoCapture.read(nextFrame);
        videoCapture.release();

        Mat mat_2 = conv_Mat(nextFrame);
        Mat hist_1 = new Mat();
        Mat hist_2 = new Mat();
        // 颜色范围
        MatOfFloat ranges = new MatOfFloat(0f, 256f);
        // 直方图大小， 越大匹配越精确 (越慢)
        MatOfInt histSize = new MatOfInt(10000000);
        Imgproc.calcHist(Arrays.asList(mat_1), new MatOfInt(0), new Mat(), hist_1, histSize, ranges);

        // 检查 mat_2 是否为 null
        if (mat_2 == null) {
            System.out.println("No face detected in the next frame.");
            return 0.0;  // 或者根据需要返回其他默认值
        }

        Imgproc.calcHist(Arrays.asList(mat_2), new MatOfInt(0), new Mat(), hist_2, histSize, ranges);
        // CORREL 相关系数
        return Imgproc.compareHist(hist_1, hist_2, Imgproc.CV_COMP_CORREL);
    }



}
