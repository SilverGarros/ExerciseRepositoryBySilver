package testopencv.demo;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.VideoWriter;
import org.opencv.videoio.Videoio;

import java.awt.*;
import java.util.Arrays;
import java.util.Currency;

/**
 * Opencv实现人脸检测
 */
public class Bigc_Face_Compare {

    //初始化 Opencv 人脸探测器
    static CascadeClassifier faceDetector;
    static int i = 0;
    private static final String PATH_PRE = "G:";
    static String CurrentDirectory = System.getProperty("user.dir");
    private static final String PATH_haarcascade_frontalface_alt2 = CurrentDirectory + "/src/main/java/testopencv/demo/haarcascade_frontalface_alt2.xml";
    private static final String PATH_haarcascade_frontalface_alt = CurrentDirectory + "/src/main/java/testopencv/demo/haarcascade_frontalface_alt.xml";
    private static final String Path_OPENCV_Dll = CurrentDirectory + "/src/main/resources/lib/opencv/opencv_java490.dll";
    // private static final String Path_OPENCV_so=CurrentDirectory+"/src/main/resources/lib/opencv/opencv_java490.so";

    static {
        // 系统判断
        String os = System.getProperty("os.name");
        // 加载动态库
        if (os != null && os.toLowerCase().startsWith("windows")) {
            // For Windows
            // todo windows system load .dll file
            System.load(Path_OPENCV_Dll);
        } else if (os != null && os.toLowerCase().startsWith("Linux")) {
            // For Linux
            // todo Linux system load .so file
            //System.load(Path_OPENCV_so);
            System.out.println("Linux io 文件尚未导入，如果只做Windows程序删掉该段");
        }

        // 引入 特征分类器配置文件
        String property = PATH_haarcascade_frontalface_alt2;
        System.out.println(property);
        faceDetector = new CascadeClassifier(property);

    }

    public static Mat FaceDetection(Mat image) {
        // 读取人脸特征xml文件
        CascadeClassifier face_detector = new CascadeClassifier(PATH_haarcascade_frontalface_alt2);
        MatOfRect face = new MatOfRect();
        face_detector.detectMultiScale(image, face);
        Rect[] rects = face.toArray();
        System.out.println(Arrays.toString(rects));
        System.out.println("检测到" + rects.length + "个人脸");

        if (rects != null && rects.length >= 1) {

            for (int i = 0; i < rects.length; i++) {

                Imgproc.rectangle(image, new Point(rects[i].x, rects[i].y), new Point(rects[i].x + rects[i].width, rects[i].y + rects[i].height), new Scalar(0, 255, 0));
                Imgproc.putText(image, "Face", new Point(rects[i].x, rects[i].y), Imgproc.FONT_HERSHEY_SIMPLEX, 0.8, new Scalar(0, 255, 0), 2, Imgproc.LINE_AA, false);
            }
            i++;

            System.out.println(i);
            if (i == 3) {
                //Mat dst = image.clone();
                //Imgproc.resize(dst, dst, new Size(300, 300));
                Imgcodecs.imwrite(PATH_PRE + "/cache/face.png", image);
                i = 0;
            }
        }
        return image;
    }

    public static void getFaceFromCameraVideo() {
        VideoCapture capture = new VideoCapture(0);
        Mat video = new Mat();
        if (capture.isOpened()) {

            while (i < 3) {
                // 匹配成功三次退出
                capture.read(video);
                HighGui.imshow("人脸识别实时显示", FaceDetection(video));
                int indexESC = HighGui.waitKey(200);//0.2秒检测一次用户按键Esc判断是否退出
                if (indexESC == 27) {
                    capture.release();
                    break;
                }
            }
        } else {
            System.out.println("摄像头未开启");
        }
        try {
            //capture.release();
            Thread.sleep(200);
        } catch (InterruptedException e) {
            e.fillInStackTrace();
        }

    }

    public static void main(String[] args) {

        getFaceFromCameraVideo();


    }


}
