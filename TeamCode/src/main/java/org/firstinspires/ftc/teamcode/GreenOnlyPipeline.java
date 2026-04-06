package org.firstinspires.ftc.teamcode;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;

import java.util.ArrayList;
import java.util.List;

public class GreenOnlyPipeline extends OpenCvPipeline {

    private Mat hsv = new Mat();
    private Mat mask = new Mat();
    private Mat output = new Mat();

    private List<MatOfPoint> contours = new ArrayList<>();


    public Scalar lowerGreen = new Scalar(75, 180, 60);
    public Scalar upperGreen = new Scalar(108, 255, 230);

    public int purpleCount = 0;

    @Override
    public Mat processFrame(Mat input) {


        Imgproc.cvtColor(input, hsv, Imgproc.COLOR_RGB2HSV);


        Core.inRange(hsv, lowerGreen, upperGreen, mask);

        contours.clear();


        Imgproc.findContours(mask, contours, new Mat(),
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        purpleCount = 0;


        output = Mat.zeros(input.size(), input.type());


        for (MatOfPoint contour : contours) {

            double area = Imgproc.contourArea(contour);

            if (area > 300) {


                Imgproc.drawContours(output, contours,
                        contours.indexOf(contour),
                        new Scalar(255, 255, 255),
                        -1);

                purpleCount++;
            }
        }

        return output;
    }
}