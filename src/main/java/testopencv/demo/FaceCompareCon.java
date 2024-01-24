package testopencv.demo;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.opencv.highgui.HighGui;

import java.util.Arrays;

public class FaceCompareCon {

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

    // 持续比较摄像头图像
    public static void compare_camera_continuous() {
        VideoCapture videoCapture = new VideoCapture(0);

        // 检查摄像头是否打开
        if (!videoCapture.isOpened()) {
            System.out.println("Error: Camera not opened.");
            return;
        }

        Mat previousFrame = new Mat();
        Mat currentFrame = new Mat();

        // 创建窗口
        HighGui.namedWindow("Face Comparison", HighGui.WINDOW_NORMAL);

        while (true) {
            videoCapture.read(currentFrame);

            // 检查当前帧是否读取成功
            if (currentFrame.empty()) {
                System.out.println("Error: Couldn't read frame from camera.");
                break;
            }

            // 检测人脸并返回人脸区域
            Mat faceRegion = detect_face(currentFrame);

            if (!faceRegion.empty()) {
                // 进行人脸相似度比较
                double compareHist = compare_frames(previousFrame, faceRegion);
                System.out.println("Comparison result: " + compareHist);

                // 在窗口中显示当前帧
                HighGui.imshow("Face Comparison", currentFrame);
                HighGui.waitKey(1);  // 添加适当的延时，以允许窗口刷新
            } else {
                System.out.println("No face detected in the current frame.");
            }

            // 设置当前帧为下一轮的前一帧
            previousFrame.release();
            previousFrame = faceRegion.clone();

            try {
                // 添加适当的延时，以控制比较频率
                Thread.sleep(500); // 例如，每隔1秒比较一次
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        // 释放资源
        videoCapture.release();
        HighGui.destroyAllWindows();
    }

    // 检测人脸并返回人脸区域
    public static Mat detect_face(Mat frame) {
        Mat imageGray = new Mat();
        Imgproc.cvtColor(frame, imageGray, Imgproc.COLOR_BGR2GRAY);

        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(imageGray, faceDetections);

        if (!faceDetections.empty()) {
            Rect rect = faceDetections.toArray()[0]; // 取第一个检测到的人脸
            Mat face = new Mat(imageGray, rect);
            return face;
        } else {
            return new Mat();
        }
    }

    // 比较两帧图像
    public static double compare_frames(Mat frame1, Mat frame2) {
        Mat mat1 = conv_Mat(frame1);

        // 检查 mat1 是否为空
        if (mat1.empty()) {
            System.out.println("No face detected in the previous frame.");
            return 0.0;  // 或者根据需要返回其他默认值
        }

        Mat mat2 = conv_Mat(frame2);
        Mat hist1 = new Mat();
        Mat hist2 = new Mat();
        // 颜色范围
        MatOfFloat ranges = new MatOfFloat(0f, 256f);
        // 直方图大小， 越大匹配越精确 (越慢)
        MatOfInt histSize = new MatOfInt(10000000);
        Imgproc.calcHist(Arrays.asList(mat1), new MatOfInt(0), new Mat(), hist1, histSize, ranges);

        // 检查 mat2 是否为空
        if (mat2.empty()) {
            System.out.println("No face detected in the current frame.");
            return 0.0;  // 或者根据需要返回其他默认值
        }

        Imgproc.calcHist(Arrays.asList(mat2), new MatOfInt(0), new Mat(), hist2, histSize, ranges);
        // CORREL 相关系数
        return Imgproc.compareHist(hist1, hist2, Imgproc.CV_COMP_CORREL);
    }

    public static Mat conv_Mat(Mat frame) {
        if (frame.empty()) {
            System.out.println("Input frame is empty.");
            return new Mat();  // or handle it accordingly
        }

        Mat image1 = new Mat();

        // 检查通道数是否为 1，如果是，不进行颜色转换
        if (frame.channels() == 1) {
            image1 = frame.clone();
        } else {
            // 灰度化
            Imgproc.cvtColor(frame, image1, Imgproc.COLOR_BGR2GRAY);
        }

        // 探测人脸
        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(image1, faceDetections);

        // 检查是否检测到人脸
        if (!faceDetections.empty()) {
            Rect rect = faceDetections.toArray()[0]; // 取第一个检测到的人脸
            Mat face = new Mat(image1, rect);
            return face;
        } else {
            // 如果未检测到人脸，返回一个空的 Mat 对象
            return new Mat();
        }
    }

    public static void main(String[] args) {
        compare_camera_continuous();
    }
}
