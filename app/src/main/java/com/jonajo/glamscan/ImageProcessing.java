package com.jonajo.glamscan;

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
import static org.opencv.core.Core.bitwise_not;
import static org.opencv.core.Core.flip;
import static org.opencv.imgproc.Imgproc.COLOR_RGB2GRAY;
import static org.opencv.imgproc.Imgproc.COLOR_RGB2HSV;
import static org.opencv.imgproc.Imgproc.FONT_HERSHEY_SIMPLEX;
import static org.opencv.imgproc.Imgproc.GaussianBlur;
import static org.opencv.imgproc.Imgproc.THRESH_BINARY_INV;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.getRotationMatrix2D;
import static org.opencv.imgproc.Imgproc.medianBlur;
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

    /**
     * This function rotates image by angle.
     *
     * @param image     //Image to be rotated counter-clockwise.
     * @param dst     //Destination mat.
     * @param angle     //Angle to rotate image in degrees. Negative values mean clockwise rotation.
     */
    public static void rotateImage(Mat image, Mat dst, float angle){
        Point imgCenter = new Point(image.size().width/2, image.size().height/2);
        Mat rotMat = getRotationMatrix2D(imgCenter, angle, 1.0);
        Mat rotImage = new Mat();
        warpAffine(image, image, rotMat, image.size(), Imgproc.INTER_LINEAR);

        image.copyTo(dst);
    }

    /**
     * This function computes from a mean value the alpha and beta values needed to perform illumination change.
     *
     * @param mean    //Mean value used to compute alpha and beta.
     * @return //Array containing the alpha and beta values computed.
     */
    public static float[] getBetaAlpha(float mean){
        //Log.i("Mean= ", String.valueOf(mean));
        float alpha = 2 - 2*(mean / 255.f);
        float beta = 2*(mean / 255.f);

        return new float[]{alpha, beta};
    }

    /**
     * This function computes the mean value of mat.
     *
     * @param mat     //1D mat.
     * @return //Mean value of mat.
     */
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

    /**
     * This function performs image processing that paint nails.
     *
     * @param frame     //Image in RGB format containing a single hand whose nails we'll add color.
     *            The wrist of the hand needs to be visible for the algorithm to work properly.
     * @param color //Image in RGB format containing the color we want to add to the nails.
     * @param nail_mask //A mask in RGB format containing the generic shape of a nail.
     * @param rthumb_mask //A mask in RGB format containing the generic shape of a right thumb.
     * @param lthumb_mask //A mask in RGB format containing the generic shape of a left thumb.
     * @param boxespred //The coordinates predicted for the bounding box of each nail.
     * @param scorespred //The scores of the bounding boxes predicted.
     * @param angles //The rotation angles for each finger.
     * @param point2coords //The coordinates of the point P2=(x2,y2) used to compute the angles.
     * @param isLeftHand //Indicates if the image to be processed is a left or right hand.
     * @param nailsize //String indicating the desired size of the final nail (can be "small", "medium", or "large").
     * @param orgsize //Original size of frame before processing.
     * @param nailpatches //Number of local patches used to adjust illumination.
     * @param illum_offset //This value increases the number of pixels used to average illumination.
     * @param resolution //Resolution used during the coloring process, it can be the size of the square image used in the prediction
     *                   of the bounding boxes or any other value >0.
     *                   This parameter exists because it is more efficient to perform image processing
     *                   with a low resolution version of frame and at the end resize the result to the original size
     *                   and paste it into frame so frame is never resized.
     * @param minscore //Minimum score needed to consider that the bounding box predicted has a nail in it.
     * @param showBoxes //Whether we want to display or not the bounding boxes and the angles provided for the image.
     * @param houtlinemask // mask to show a predefined hand outline
     * @param showOutline // wheter we want to show hand's outline
     * @return //The processed image with nails painted.
     */
    public static Mat paintNail(Mat frame, Mat color, Mat nail_mask, Mat rthumb_mask, Mat lthumb_mask,
                                   float[][] boxespred, float[] scorespred, float[] angles, float[][] point2coords,
                                   Boolean isLeftHand, String nailsize, float[] orgsize, int  nailpatches,
                                   int illum_offset, int resolution, float minscore, Boolean showBoxes, Mat houtlinemask,
                                Boolean showOutline){

        Size modelsize = new Size(resolution, resolution);
        Mat image = new Mat();
        int ymin_raw, xmin_raw, ymax_raw, xmax_raw, xminoff, yminoff, xmaxoff, ymaxoff, ymin, ymax, xmin, xmax;
        Boolean fingerfound;
        float angle=0;
        int finger=0;
        float factor = resolution/224.f; // landmarks are obtained with images of size 224X224

        Mat imageOrg = frame.clone(); // original image never resize so the its resolution is preserved
        image = frame.clone(); // create a copy of the image to resize (so original image is never resized)

        if(frame.size() != modelsize){
            resize(frame, image, modelsize);
        }

        for(int i=0; i<boxespred.length; i++){
            for(int j=0; j<boxespred[i].length; j++){
                boxespred[i][j] = boxespred[i][j] * resolution;
            }
        }

        // resize hand outline contour
        Mat handContour = new Mat();
        Size finalsize = new Size(orgsize[0], orgsize[1]);
        if(showOutline){
            resize(houtlinemask, handContour, finalsize);
            if(isLeftHand){
                flip(handContour, handContour, 1);
            }
        }

        for(int pred=0; pred<scorespred.length; pred++){
            if(scorespred[pred] > minscore){
                ymin_raw = Math.round(boxespred[pred][0]);
                xmin_raw = Math.round(boxespred[pred][1]);
                ymax_raw = Math.round(boxespred[pred][2]);
                xmax_raw = Math.round(boxespred[pred][3]);

                /////////////////////////
                // check is nail's size is normal, medium or large
                if(nailsize.equals("medium")){
                    xminoff=30;
                    yminoff=15;
                    xmaxoff=30;
                    ymaxoff=19;
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
                if(ymin<0){ //check values
                    ymin=0;
                } if(xmin<0){
                    xmin=0;
                } if(ymax > image.size().height){
                    ymax = (int) Math.round(image.size().height);
                } if(xmax > image.size().width){
                    xmax = (int) Math.round(image.size().width);
                } if(ymin_raw<0){ // check raw values
                    ymin_raw=0;
                } if(xmin_raw<0){
                    xmin_raw=0;
                } if(ymax_raw > image.size().height){
                    ymax_raw = (int) Math.round(image.size().height);
                } if(xmax_raw > image.size().width){
                    xmax_raw = (int) Math.round(image.size().width);
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

                // Check submat size will not equal image size
                if(((int) Math.round(image.size().width))<=(xmax-xmin) || ((int) Math.round(image.size().height)<=(ymax-ymin))){
                    break;
                }
                if(((int) Math.round(image.size().width))<=(xmax_raw-xmin_raw) || ((int) Math.round(image.size().height)<=(ymax_raw-ymin_raw))){
                    break;
                }
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
                    if(widthend - widthstart <=0){
                        break;
                    }

                    //use quadrants to adjust illumination
                    for(int j=0; j<nailpatches; j++){
                        int heightstart = j*(areaheight);
                        int heightend = (j+1)*(areaheight) + illum_offset;
                        if(heightend > heightHSV){
                            heightend = heightHSV;
                        }
                        if(heightend - heightstart <=0){
                            break;
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
                //seamlessClone(nail_raw, finalcolor, nail_rawmaskedinv, point, finalcolor, MONOCHROME_TRANSFER);

                //////////
                Mat mixed = new Mat();
                add(finalcolor, nailmasked, mixed);
                // Apply Antialiasing to eliminate jagged results
                /*Size mixedSize = mixed.size();
                Size mixedSizeLarge = new Size(4000, 4000);
                resize(mixed, mixed, mixedSizeLarge);
                resize(mixed, mixed, mixedSize);*/
                medianBlur(mixed, mixed,5);
                Size kSize = new Size(5, 5);
                GaussianBlur(mixed, mixed, kSize ,0);
                // copy illum info
                if(Math.round(nail_raw.size().width/2)+xminoff >=nail_raw.size().width || Math.round(nail_raw.size().height/2)+yminoff >=nail_raw.size().height){
                    Point point = new Point(Math.round(nail_raw.size().width/2), Math.round(nail_raw.size().height/2));
                    seamlessClone(nail_raw, mixed, nail_rawmaskedinv, point, mixed, MONOCHROME_TRANSFER);
                } else{
                    Point point = new Point(Math.round(nail_raw.size().width/2)+xminoff, Math.round(nail_raw.size().height/2)+yminoff);
                    seamlessClone(nail_raw, mixed, nail_rawmaskedinv, point, mixed, MONOCHROME_TRANSFER);
                }
                //////////

                //mixed.copyTo(image.submat(ymin, ymax, xmin, xmax));
                double xfactor = frame.size().width/image.size().width;
                double yfactor = frame.size().height/image.size().height;
                // Warning: round error may cause the copyTo function to not work!!
                Size fullmixed = new Size((int)(Math.round(xmax*xfactor) - Math.round(xmin*xfactor)), (int)(Math.round(ymax*yfactor) - Math.round(ymin*yfactor)));
                resize(mixed, mixed, fullmixed);
                mixed.copyTo(imageOrg.submat((int) Math.round(ymin*yfactor),
                        (int) Math.round(ymax*yfactor),
                        (int) Math.round(xmin*xfactor),
                        (int) Math.round(xmax*xfactor)));

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

        Mat finalimage = new Mat();

        if(showOutline){
            // resize to orIginal ratio
            //resize(image, finalimage, finalsize);
            // add hand outline
            Mat finalimageoutline = new Mat();
            add(imageOrg, handContour, finalimageoutline);

            return finalimageoutline;

        } else{
            //resize(image, finalimage, finalsize);

            return imageOrg;
        }

    }
}
