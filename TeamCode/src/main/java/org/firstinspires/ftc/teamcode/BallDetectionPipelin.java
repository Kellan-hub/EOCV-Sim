package org.firstinspires.ftc.teamcode;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;

import java.util.ArrayList;
import java.util.List;

public class BallDetectionPipelin extends OpenCvPipeline {

    private Mat hsv = new Mat();
    private Mat mask1 = new Mat();
    private Mat mask2 = new Mat();

    private List<MatOfPoint> contours1 = new ArrayList<>();
    private List<MatOfPoint> contours2 = new ArrayList<>();

    // DETECTION COLORS - CHANGE THESE TO DETECT OTHER COLORS
    public Scalar lowerColor1 = new Scalar(130, 50, 50); // Purple HSV
    public Scalar upperColor1 = new Scalar(160, 255, 255);

    public Scalar lowerColor2 = new Scalar(40, 50, 50);  // Green HSV
    public Scalar upperColor2 = new Scalar(80, 255, 255);

    // COLORS TO DRAW ON DETECTED BALLS
    private Scalar drawColor1 = new Scalar(0, 255, 255);  // Cyan for purple balls
    private Scalar drawColor2 = new Scalar(255, 0, 255);  // Magenta for green balls

    public int color1Count = 0;
    public int color2Count = 0;

    @Override
    public Mat processFrame(Mat input) {

        // Convert frame to HSV
        Imgproc.cvtColor(input, hsv, Imgproc.COLOR_RGB2HSV);

        // Create masks
        Core.inRange(hsv, lowerColor1, upperColor1, mask1);
        Core.inRange(hsv, lowerColor2, upperColor2, mask2);

        contours1.clear();
        contours2.clear();

        // Find contours
        Imgproc.findContours(mask1, contours1, new Mat(),
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        Imgproc.findContours(mask2, contours2, new Mat(),
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        color1Count = 0;
        color2Count = 0;

        // Draw detected color1 balls in drawColor1
        for (MatOfPoint contour : contours1) {
            double area = Imgproc.contourArea(contour);
            if (area > 300) {
                Rect rect = Imgproc.boundingRect(contour);
                int cx = rect.x + rect.width/2;
                int cy = rect.y + rect.height/2;

                // Draw rectangle and center
                Imgproc.rectangle(input, rect, drawColor1, 3);
                Imgproc.circle(input, new Point(cx, cy), 5, drawColor1, -1);
                Imgproc.putText(input, "COLOR1", new Point(cx, cy-10),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, drawColor1, 2);

                color1Count++;
            }
        }

        // Draw detected color2 balls in drawColor2
        for (MatOfPoint contour : contours2) {
            double area = Imgproc.contourArea(contour);
            if (area > 300) {
                Rect rect = Imgproc.boundingRect(contour);
                int cx = rect.x + rect.width/2;
                int cy = rect.y + rect.height/2;

                Imgproc.rectangle(input, rect, drawColor2, 3);
                Imgproc.circle(input, new Point(cx, cy), 5, drawColor2, -1);
                Imgproc.putText(input, "COLOR2", new Point(cx, cy-10),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, drawColor2, 2);

                color2Count++;
            }
        }

        return input;
    }
}