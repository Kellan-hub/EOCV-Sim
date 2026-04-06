package org.firstinspires.ftc.teamcode;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;

import java.util.ArrayList;
import java.util.List;

public class PurpleGreenPipeline extends OpenCvPipeline {

    private Mat hsv = new Mat();
    private Mat maskPurple = new Mat();
    private Mat maskGreen = new Mat();
    private Mat combinedMask = new Mat();
    private Mat output = new Mat();

    private List<MatOfPoint> contours = new ArrayList<>();


    public Scalar lowerPurple = new Scalar(120, 50, 70);
    public Scalar upperPurple = new Scalar(250, 250, 250);

    public Scalar lowerGreen = new Scalar(75, 180, 60);
    public Scalar upperGreen = new Scalar(108, 255, 230);

    public int totalCount = 0;

    @Override
    public Mat processFrame(Mat input) {


        Imgproc.cvtColor(input, hsv, Imgproc.COLOR_RGB2HSV);


        Imgproc.GaussianBlur(hsv, hsv, new Size(5,5), 0);


        Core.inRange(hsv, lowerPurple, upperPurple, maskPurple);
        Core.inRange(hsv, lowerGreen, upperGreen, maskGreen);


        Core.bitwise_or(maskPurple, maskGreen, combinedMask);


        contours.clear();
        Imgproc.findContours(combinedMask, contours, new Mat(),
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);


        output = Mat.zeros(input.size(), input.type());

        totalCount = 0;


        for (MatOfPoint contour : contours) {

            double area = Imgproc.contourArea(contour);

            if (area > 2000) {

                Imgproc.drawContours(
                        output,
                        contours,
                        contours.indexOf(contour),
                        new Scalar(255, 255, 255),
                        -1
                );

                totalCount++;
            }
        }

        return output;
    }
}