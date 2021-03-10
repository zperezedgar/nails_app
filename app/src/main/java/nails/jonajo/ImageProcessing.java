package nails.jonajo;

import android.graphics.Bitmap;
import android.util.Log;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;

import static org.opencv.core.Core.add;
import static org.opencv.core.Core.bitwise_and;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2RGB;
import static org.opencv.imgproc.Imgproc.COLOR_RGB2GRAY;
import static org.opencv.imgproc.Imgproc.COLOR_RGB2HSV;
import static org.opencv.imgproc.Imgproc.FONT_HERSHEY_SIMPLEX;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY_INV;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.getRotationMatrix2D;
import static org.opencv.imgproc.Imgproc.putText;
import static org.opencv.imgproc.Imgproc.rectangle;
import static org.opencv.imgproc.Imgproc.resize;
import static org.opencv.imgproc.Imgproc.threshold;
import static org.opencv.imgproc.Imgproc.warpAffine;
import static org.opencv.photo.Photo.MONOCHROME_TRANSFER;
import static org.opencv.photo.Photo.illuminationChange;
import static org.opencv.photo.Photo.seamlessClone;

public class ImageProcessing {
    // class contains image processing algorithms

    /**
     * This function implements image segmentation to img by using the watershed algorithm.
     *
     * @param img     //The image in RGB format to which the algorithm will be applied.
     * @return //The image obtained with the watershed algorithm.
     *          In this image the sure background of img has been identified and colored black.
     *          The rest of the image remains the same.
     */
    public static Mat Watershed(Mat img){
        //watershed algorithm

        Mat gray = new Mat();
        Mat thresh = new Mat();
        double ret;
        Mat  kernel;
        Mat opening = new Mat();
        Mat sure_bg = new Mat();
        Point anchor = new Point(-1, -1); // default value
        Mat dist_transform = new Mat();
        Mat sure_fg = new Mat();
        Mat unknown = new Mat();
        Mat markers = new Mat();
        Scalar sc = new Scalar(-1);

        Imgproc.cvtColor(img, gray, COLOR_RGB2GRAY);
        ret = Imgproc.threshold(gray, thresh, 0 ,255, THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

        // remove noise
        kernel = Mat.ones(3, 3, CvType.CV_8UC(1));
        Imgproc.morphologyEx(thresh, opening, Imgproc.MORPH_OPEN, kernel, anchor, 1);

        // Find the sure background region
        Imgproc.dilate(opening, sure_bg, kernel, anchor, 8);

        // Find the sure foreground region
        Imgproc.distanceTransform(opening, dist_transform,Imgproc.DIST_L2, 5);
        Core.MinMaxLocResult mmr= Core.minMaxLoc(dist_transform);
        ret = Imgproc.threshold(dist_transform, sure_fg, 0.7 * mmr.maxVal ,255, 0);
        sure_fg.assignTo(sure_fg, CvType.CV_8UC(sure_fg.channels()));
        //Log.i("value", String.valueOf(0.7 * mmr.maxVal));

        // Find the unknown region
        Core.subtract(sure_bg, sure_fg, unknown);

        // Label the foreground objects
        ret = Imgproc.connectedComponents(sure_fg, markers);

        // Add one to all labels so that sure background is not 0, but 1
        Core.subtract(markers, sc, markers);
        //for(int i=0; i <markers.rows(); i++){

          //  for(int j=0; j < markers.cols(); j++){
            //    markers.put(i, j, markers.get(i, j)[0] + 1);
            //}
        //}

        // Label the unknown region as 0
        for(int i=0; i <markers.rows(); i++){

            for(int j=0; j < markers.cols(); j++){
                if(unknown.get(i, j)[0] == 255){
                    markers.put(i, j, 0);
                }
            }
        }

        Imgproc.watershed(img, markers);

        for(int i=0; i <markers.rows(); i++){

            for(int j=0; j < markers.cols(); j++){
                if(markers.get(i, j)[0] == 1){
                    img.put(i, j, 0,0,0);
                }
            }
        }

        return img;
    }

    /**
     * This function converts images in RGB format from Mat to BitMap
     *
     * @param input     //The image in RGB format to be converted from Mat to BitMap
     * @return //The image converted to BitMap
     */
    public static Bitmap convertMatToBitMap(Mat input){
        Bitmap bmp = null;
        //Mat rgb = new Mat();
        //Imgproc.cvtColor(input, rgb, Imgproc.COLOR_BGR2RGB);

        Mat rgb = input;

        try {
            bmp = Bitmap.createBitmap(rgb.cols(), rgb.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(rgb, bmp);
        }
        catch (CvException e){
            Log.d("Exception",e.getMessage());
        }
        return bmp;
    }

    public static void rotateImage(Mat image, Mat dst, float angle){
        Point imgCenter = new Point(image.size().width/2, image.size().height/2);
        Mat rotMat = getRotationMatrix2D(imgCenter, angle, 1.0);
        Mat rotImage = new Mat();
        warpAffine(image, image, rotMat, image.size(), Imgproc.INTER_LINEAR);

        image.copyTo(dst);
    }

    public static float[] getBetaAlpha(float mean){
        //Log.i("Mean= ", String.valueOf(mean));
        float alpha = 2 - 2*(mean / 255.f);
        float beta = 2*(mean / 255.f);

        return new float[]{alpha, beta};
    }

    public static float meanMat(Mat mat){
        int columns = (int) mat.size().width;
        int rows = (int) mat.size().height;
        float mean=0;
        double[] pixel;

        for(int i=0; i<rows; i++){
            for(int j=0; j<columns; j++){
                pixel = mat.get(i, j);
                mean += pixel[0];
            }
        }

        mean /= (columns*rows);

        return mean;
    }

    public static Mat paintNail(Mat frame, Mat color, Mat nail_mask, Mat rthumb_mask, Mat lthumb_mask,
                                   float[][] boxespred, float[] scorespred, float[] angles, float[][] point2coords,
                                   Boolean isLeftHand, String nailsize, float[] orgsize, int  nailpatches,
                                   int illum_offset, int size, float minscore, Boolean showBoxes){

        Size modelsize = new Size(size, size);
        Mat image = new Mat();
        int ymin_raw, xmin_raw, ymax_raw, xmax_raw, xminoff, yminoff, xmaxoff, ymaxoff, ymin, ymax, xmin, xmax;
        Boolean fingerfound;
        float angle=0;
        int finger=0;
        float factor = size/224.f; // landmarks are obtained with images of size 224X224

        if(frame.size() != modelsize){
            resize(frame, image, modelsize);
        } else{
            image = frame.clone();
        }

        for(int i=0; i<boxespred.length; i++){
            for(int j=0; j<boxespred[i].length; j++){
                boxespred[i][j] = boxespred[i][j] * size;
            }
        }

        //cvtColor(color, color, COLOR_BGR2RGB);

        for(int pred=0; pred<scorespred.length; pred++){
            if(scorespred[pred] > minscore){
                ymin_raw = Math.round(boxespred[pred][0]);
                xmin_raw = Math.round(boxespred[pred][1]);
                ymax_raw = Math.round(boxespred[pred][2]);
                xmax_raw = Math.round(boxespred[pred][3]);

                /////////////////////////
                // check is nail's size is normal, medium or large
                if(nailsize.equals("medium")){
                    xminoff=10;
                    yminoff=16;
                    xmaxoff=10;
                    ymaxoff=12;
                    ymin = ymin_raw - yminoff;
                    ymax = ymax_raw + ymaxoff;
                    xmin = xmin_raw - xminoff;
                    xmax = xmax_raw + xmaxoff;
                } else if(nailsize.equals("large")){
                    xminoff=8;
                    yminoff=20;
                    xmaxoff=8;
                    ymaxoff=7;
                    ymin = ymin_raw - yminoff;
                    ymax = ymax_raw + ymaxoff;
                    xmin = xmin_raw - xminoff;
                    xmax = xmax_raw + xmaxoff;
                } else{
                    xminoff=2;
                    yminoff=0;
                    xmaxoff=2;
                    ymaxoff=1;
                    ymin = ymin_raw - yminoff;
                    ymax = ymax_raw + ymaxoff;
                    xmin = xmin_raw - xminoff;
                    xmax = xmax_raw + xmaxoff;
                }

                // assert we get correct values
                if(ymin<0){
                    ymin=0;
                } else if(xmin<0){
                    xmin=0;
                } else if(ymax > image.size().height){
                    ymax = (int) Math.round(image.size().height);
                } else if(xmax > image.size().width){
                    xmax = (int) Math.round(image.size().width);
                }
                /////////////////////////

                /////////////////////////
                // find the corresponding rot angle
                // use horizontal coordinates to find the finger number
                fingerfound = false;
                for(int i=0; i<angles.length; i++){
                    if((point2coords[i][0]*factor >= xmin) && (point2coords[i][0]*factor <= xmax)){
                        angle = angles[i];
                        if(isLeftHand){
                            finger = i;
                        } else{
                            finger = 4-i;
                        }
                        fingerfound = true;
                        break;
                    } else{
                        angle = 0;
                        finger = 666; // not thumb finger
                    }
                }
                // If finger was not found using horizontal coordinates, use vertical coordinates instead
                if(!fingerfound){
                    for(int i=0; i<angles.length; i++){
                        if((point2coords[i][1]*factor >= ymin) && (point2coords[i][1]*factor <= ymax)){
                            angle = angles[i];
                            if(isLeftHand){
                                finger = i;
                            } else{
                                finger = 4-i;
                            }
                            fingerfound = true;
                            break;
                        } else{
                            angle = 0;
                            finger = 666; // not thumb finger
                        }
                    }
                }
                /////////////////////////

                Mat nail = image.submat(ymin, ymax, xmin, xmax);
                Mat nail_raw = image.submat(ymin_raw, ymax_raw ,xmin_raw, xmax_raw);

                /////////////////////////
                // create a mask to add color
                Mat mask = new Mat();
                Mat nail_rawmask = new Mat();
                Size maskSize = new Size(Math.round((float) nail.size().width), Math.round((float) nail.size().height));
                Size rawmaskSize = new Size(Math.round((float) nail_raw.size().width), Math.round((float) nail_raw.size().height));
                if((isLeftHand && finger!=4) || ((!isLeftHand) && finger!=0)){ //not thumb
                    resize(nail_mask, mask, maskSize);
                    resize(nail_mask, nail_rawmask, rawmaskSize);
                } else if(isLeftHand && finger==4){ // left hand thumb
                    resize(lthumb_mask, mask, maskSize);
                    resize(lthumb_mask, nail_rawmask, rawmaskSize);
                } else if((!isLeftHand) && (finger==0)){ // right hand thumb
                    resize(rthumb_mask, mask, maskSize);
                    resize(rthumb_mask, nail_rawmask, rawmaskSize);
                } else{
                    resize(nail_mask, mask, maskSize);
                    resize(nail_mask, nail_rawmask, rawmaskSize);
                }
                Mat rotmask = new Mat();
                Mat rotnail_rawmask = new Mat();
                rotateImage(mask, mask,angle-90);
                rotateImage(nail_rawmask, nail_rawmask, angle-90);
                cvtColor(mask, mask, COLOR_RGB2GRAY);
                cvtColor(nail_rawmask, nail_rawmask, COLOR_RGB2GRAY);
                threshold(mask, mask, 0, 255, THRESH_BINARY_INV);
                threshold(nail_rawmask, nail_rawmask, 0, 255, THRESH_BINARY_INV);
                Mat mask_inv = new Mat();
                Core.bitwise_not(mask, mask_inv);
                Mat nail_rawmask_inv = new Mat();
                Core.bitwise_not(nail_rawmask, nail_rawmask_inv);
                Mat nailmasked = new Mat();
                bitwise_and(nail, nail, nailmasked, mask);
                /////////////////////////

                /////////////////////////
                // Add COLOR
                Mat glamcolor = color.clone();
                rotateImage(glamcolor, glamcolor,angle-90);
                Size newcolorSize = new Size(nail.size().width, nail.size().height);
                Mat newcolor = new Mat();
                resize(glamcolor, newcolor, newcolorSize);

                ///////////
                // Adjust illumination
                Mat nailmaskedinv = new Mat();
                bitwise_and(nail,nail, nailmaskedinv, mask_inv);
                Mat nailmaskedinvHSV = new Mat();
                cvtColor(nailmaskedinv, nailmaskedinvHSV, COLOR_RGB2HSV);
                int heightHSV = (int) nailmaskedinvHSV.size().height;
                int widthHSV = (int) nailmaskedinvHSV.size().width;
                int areawidth = Math.round(widthHSV/nailpatches);
                int areaheight = Math.round(heightHSV/nailpatches);

                for(int i=0; i<nailpatches; i++){
                    // use vertical bands to adjust illumination
                    int widthstart = i*(areawidth);
                    int widthend = (i+1)*(areawidth) + illum_offset;
                    if(widthend > widthHSV){
                        widthend = widthHSV;
                    }

                    //use quadrants to adjust illumination
                    for(int j=0; j<nailpatches; j++){
                        int heightstart = j*(areaheight);
                        int heightend = (j+1)*(areaheight) + illum_offset;
                        if(heightend > heightHSV){
                            heightend = heightHSV;
                        }
                        Mat nailpatch = nailmaskedinvHSV.submat(heightstart, heightend, widthstart, widthend); // quadrants
                        Mat colorpatch = newcolor.submat(heightstart, heightend, widthstart, widthend); // quadrants
                        ArrayList<Mat> channels = new ArrayList<Mat>();
                        Core.split(nailpatch, channels);
                        float mean = meanMat(channels.get(2));
                        float[] alphaBeta = getBetaAlpha(mean);
                        Mat maskcolor = Mat.ones((int) colorpatch.size().height,(int) colorpatch.size().width, CvType.CV_8UC(1));
                        Mat newcolorpatch = new Mat();
                        illuminationChange(colorpatch, maskcolor, newcolorpatch, alphaBeta[0], alphaBeta[1]); // alpha controls the color, beta=0 does nothing
                        newcolorpatch.copyTo(newcolor.submat(heightstart, heightend, widthstart, widthend));
                        //newcolor.submat(heightstart, heightend, widthstart, widthend).copyTo(newcolorpatch); // quadrants
                    }
                }
                Mat finalcolor = new Mat();
                bitwise_and(newcolor, newcolor, finalcolor, mask_inv);
                Mat nail_rawmaskedinv = new Mat();
                bitwise_and(nail_raw, nail_raw, nail_rawmaskedinv, nail_rawmask_inv);
                Point point = new Point(Math.round(nail_raw.size().width/2)+xminoff, Math.round(nail_raw.size().height/2)+yminoff);
                seamlessClone(nail_raw, finalcolor, nail_rawmaskedinv, point, finalcolor, MONOCHROME_TRANSFER);

                //////////
                Mat mixed = new Mat();
                add(finalcolor, nailmasked, mixed);
                //////////

                mixed.copyTo(image.submat(ymin, ymax, xmin, xmax));

                ////////////////
                // Show raw bounding boxes and rot angles
                if(showBoxes){
                    Point startpoint = new Point(xmin_raw, ymin_raw);
                    Point endpoint = new Point(xmax_raw, ymax_raw);
                    Scalar rectcolor = new Scalar(0, 255, 0);
                    rectangle(image, startpoint, endpoint, rectcolor, 1);
                    Scalar textcolor = new Scalar(255, 255, 0);
                    putText(image, String.valueOf(angle), endpoint, FONT_HERSHEY_SIMPLEX,
                            0.5, textcolor, 1);
                }
                ////////////////
            }
        }
        // resize to orIginal ratio
        Mat finalimage = new Mat();
        Size finalsize = new Size(orgsize[0], orgsize[1]);
        resize(image, finalimage, finalsize);

        return finalimage;
    }
}
