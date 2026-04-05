package org.firstinspires.ftc.teamcode;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;

import java.util.ArrayList;
import java.util.List;

public class BWBallPipeline extends OpenCvPipeline {

    private Mat hsv = new Mat();
    private Mat mask1 = new Mat();
    private Mat mask2 = new Mat();

    private List<MatOfPoint> contours1 = new ArrayList<>();
    private List<MatOfPoint> contours2 = new ArrayList<>();


    public Scalar lowerPurple = new Scalar(250, 20, 20);
    public Scalar upperPurple = new Scalar(360, 60, 70);

    public Scalar lowerGreen = new Scalar(50, 10, 10);
    public Scalar upperGreen = new Scalar(200, 100, 100);

    public int purpleCount = 0;
    public int greenCount = 0;

    @Override
    public Mat processFrame(Mat input) {


        Imgproc.cvtColor(input, hsv, Imgproc.COLOR_RGB2HSV);

        Core.inRange(hsv, lowerPurple, upperPurple, mask1);
        Core.inRange(hsv, lowerGreen, upperGreen, mask2);

        Mat combinedMask = new Mat();
        Core.bitwise_or(mask1, mask2, combinedMask);


        contours1.clear();
        contours2.clear();


        Imgproc.findContours(mask1, contours1, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        purpleCount = 0;
        for (MatOfPoint c : contours1) {
            if (Imgproc.contourArea(c) > 300) purpleCount++;
        }


        Imgproc.findContours(mask2, contours2, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        greenCount = 0;
        for (MatOfPoint c : contours2) {
            if (Imgproc.contourArea(c) > 300) greenCount++;
        }


        Mat bwFrame = Mat.zeros(input.size(), input.type());


        input.copyTo(bwFrame, combinedMask);

        return bwFrame;
    }
}