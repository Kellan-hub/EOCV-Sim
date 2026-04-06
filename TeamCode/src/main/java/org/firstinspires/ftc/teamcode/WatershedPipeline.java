package org.firstinspires.ftc.teamcode;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.openftc.easyopencv.OpenCvPipeline;

public class WatershedPipeline extends OpenCvPipeline {

    private Mat hsv = new Mat();
    private Mat maskPurple = new Mat();
    private Mat maskGreen = new Mat();
    private Mat combinedMask = new Mat();
    private Mat sureBg = new Mat();
    private Mat distTransform = new Mat();
    private Mat sureFg = new Mat();
    private Mat markers = new Mat();

    public Scalar lowerPurple = new Scalar(120, 50, 70);
    public Scalar upperPurple = new Scalar(160, 255, 255);
    public Scalar lowerGreen = new Scalar(75, 180, 60);
    public Scalar upperGreen = new Scalar(108, 255, 230);

    public int ballCount = 0;
    public String bestQuadrant = "NONE";

    @Override
    public Mat processFrame(Mat input) {

        Imgproc.cvtColor(input, hsv, Imgproc.COLOR_RGB2HSV);
        Imgproc.GaussianBlur(hsv, hsv, new Size(5,5), 0);

        Core.inRange(hsv, lowerPurple, upperPurple, maskPurple);
        Core.inRange(hsv, lowerGreen, upperGreen, maskGreen);
        Core.bitwise_or(maskPurple, maskGreen, combinedMask);

        sureBg = combinedMask.clone();
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3,3));
        Imgproc.morphologyEx(sureBg, sureBg, Imgproc.MORPH_CLOSE, kernel);
        Imgproc.morphologyEx(sureBg, sureBg, Imgproc.MORPH_OPEN, kernel);
        Imgproc.dilate(sureBg, sureBg, kernel);

        Imgproc.distanceTransform(sureBg, distTransform, Imgproc.DIST_L2, 5);
        Core.normalize(distTransform, distTransform, 0, 1.0, Core.NORM_MINMAX);

        Imgproc.threshold(distTransform, sureFg, 0.2, 1.0, Imgproc.THRESH_BINARY);
        sureFg.convertTo(sureFg, CvType.CV_8U);

        Imgproc.connectedComponents(sureFg, markers);
        Core.add(markers, Scalar.all(1), markers);
        Imgproc.watershed(input, markers);

        ballCount = 0;
        int leftCount = 0, middleCount = 0, rightCount = 0;

        int width = input.cols();
        int height = input.rows();
        Rect leftRect = new Rect(0, 0, width/3, height);
        Rect middleRect = new Rect(width/3, 0, width/3, height);
        Rect rightRect = new Rect(2*width/3, 0, width/3, height);

        Imgproc.rectangle(input, leftRect.tl(), leftRect.br(), new Scalar(255,0,0), 2);
        Imgproc.rectangle(input, middleRect.tl(), middleRect.br(), new Scalar(0,255,0), 2);
        Imgproc.rectangle(input, rightRect.tl(), rightRect.br(), new Scalar(0,0,255), 2);

        double maxVal = Core.minMaxLoc(markers).maxVal;

        for (int i = 2; i <= maxVal; i++) {
            Mat mask = new Mat();
            Core.compare(markers, new Scalar(i), mask, Core.CMP_EQ);
            double area = Core.countNonZero(mask);
            if (area > 800) {
                Moments m = Imgproc.moments(mask);
                int cx = (int)(m.get_m10() / m.get_m00());
                int cy = (int)(m.get_m01() / m.get_m00());

                Imgproc.circle(input, new Point(cx, cy), 20, new Scalar(0, 255, 255), 2);
                Imgproc.putText(input, "Ball " + (ballCount+1),
                        new Point(cx - 20, cy - 25),
                        Imgproc.FONT_HERSHEY_SIMPLEX,
                        0.5, new Scalar(0,255,255), 2);

                if (leftRect.contains(new Point(cx, cy))) leftCount++;
                else if (middleRect.contains(new Point(cx, cy))) middleCount++;
                else rightCount++;

                ballCount++;
            }
        }

        if (leftCount >= middleCount && leftCount >= rightCount) bestQuadrant = "LEFT";
        else if (middleCount >= rightCount) bestQuadrant = "MIDDLE";
        else bestQuadrant = "RIGHT";

        Imgproc.putText(input, "Total Balls: " + ballCount,
                new Point(20, 40),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                1.0, new Scalar(0,255,255), 2);

        Imgproc.putText(input, "Best Quadrant: " + bestQuadrant,
                new Point(20, 80),
                Imgproc.FONT_HERSHEY_SIMPLEX,
                1.0, new Scalar(0,255,255), 2);

        return input;
    }
}