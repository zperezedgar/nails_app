package com.jonajo.glamscan;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.os.Build;
import android.widget.Toast;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

public class handLandmarkModel {

    private Bitmap mimageBitmap;
    private Context mContext;
    private Boolean misLeftHand;
    private Boolean mopenCVConvention;
    private float[][] mpredCoords = new float[1][63];
    private float[] mangles = new float[5];
    private float[][] mcoords = new float[5][2];
    private Boolean museGPU;

    // Constructor
    public handLandmarkModel(Bitmap imageBitmap, Context context, Boolean isLeftHand, Boolean openCVConvention, Boolean useGPU){
        this.mimageBitmap = imageBitmap;
        this.mContext = context; // pass getApplicationContext() when instantiating the class
        this.misLeftHand = isLeftHand;
        this.mopenCVConvention = openCVConvention;
        this.museGPU = useGPU;
    }

    // return angles computed to the user
    public float[] getAngles(){
        return mangles;
    }

    // return coords of point 2 used to get the angles
    public float[][] getCoords(){
        return mcoords;
    }

    // return a Mat showing the landmarks predicted and the rotation angles computed
    public Mat getMatResults(){

        Mat mat = new Mat();
        Bitmap bmp32 = mimageBitmap.copy(Bitmap.Config.RGB_565, true);
        Utils.bitmapToMat(bmp32, mat);
        Mat mat8UC = new Mat(mat.size(), CvType.CV_8UC(3));
        Imgproc.cvtColor(mat, mat8UC, Imgproc.COLOR_RGBA2RGB);

        float[] coords = mpredCoords[0];
        int key = 0;
        float x=0, y=0, z=0;
        Scalar color = new Scalar(0, 255, 0);
        float xFactor = mat8UC.width() / 224.f; //landmarks are obtained with images of size 224X224
        float yFactor = mat8UC.height() / 224.f;

        for(int keypoint = 1; keypoint <= coords.length; keypoint++){
            if(keypoint == (1 + (3*key))){
                x = Math.round(coords[keypoint - 1]) * xFactor;
            } else if(keypoint == (2 + (3*key))){
                y = Math.round(coords[keypoint - 1]) * yFactor;
            } else{
                z = Math.round(coords[keypoint - 1]);
            }

            if(keypoint%3 == 0){
                Point center = new Point(x, y);
                Imgproc.circle(mat8UC, center, 100,  color, -1);
                key+=1;
            }

        }

        return  mat8UC;
    }

    // run the model and compute the angles
    public void predictAngles(){

        //new runLandmarkModelBackground().execute(); //run the model in a new thread (in background)
        runLandmarkModel();
    }

    // Algorithm to compute rotation angles
    private void computeAngles(float[][] keypoints, Boolean isLeftHand, Boolean openCVConvention){

        // Rotation angle for each finger
        float thumbAngle = 0;
        float indexAngle = 0;
        float middleAngle = 0;
        float ringAngle = 0;
        float pinkyAngle = 0;

        float[] angle = {thumbAngle, indexAngle, middleAngle, ringAngle, pinkyAngle};

        // Knuckle coordinates for each finger
        float[] thumbKcoord = {keypoints[0][3*3], keypoints[0][(3*3) + 1], keypoints[0][(3*3) + 2], keypoints[0][(3*3) + 3], keypoints[0][(3*3) + 4], keypoints[0][(3*3) + 5]};
        float[] indexKcoord = {keypoints[0][7*3], keypoints[0][(7*3) + 1], keypoints[0][(7*3) + 2], keypoints[0][(7*3) + 3], keypoints[0][(7*3) + 4], keypoints[0][(7*3) + 5]};
        float[] middleKcoord = {keypoints[0][11*3], keypoints[0][(11*3) + 1], keypoints[0][(11*3) + 2], keypoints[0][(11*3) + 3], keypoints[0][(11*3) + 4], keypoints[0][(11*3) + 5]};
        float[] ringKcoord = {keypoints[0][15*3], keypoints[0][(15*3) + 1], keypoints[0][(15*3) + 2], keypoints[0][(15*3) + 3], keypoints[0][(15*3) + 4], keypoints[0][(15*3) + 5]};
        float[] pinkyKcoord = {keypoints[0][19*3], keypoints[0][(19*3) + 1], keypoints[0][(19*3) + 2], keypoints[0][(19*3) + 3], keypoints[0][(19*3) + 4], keypoints[0][(19*3) + 5]};

        float[][] kCoords = {pinkyKcoord, ringKcoord, middleKcoord, indexKcoord, thumbKcoord};
        float[][] point2Coords = new float[5][2];

        for(int finger = 0; finger<angle.length; finger++){
            float x1 = kCoords[finger][0];
            float y1 = kCoords[finger][1];
            float x2 = kCoords[finger][3];
            float y2 = kCoords[finger][4];

            // Get angles in degrees between -90° to 90°
            angle[finger] = (float) (Math.atan((y2 - y1)/(x2 - x1)) * 360/(2*Math.PI));

            // Get angles ranging from 0 to 360
            if(openCVConvention){
                if((x1 < x2) && (y2 < y1)){
                    angle[finger] = -angle[finger];

                } else if((x2 < x1) && (y2 < y1)){
                    angle[finger] = 180 - angle[finger];

                } else if((x2 < x1) && (y1 < y2)){
                    angle[finger] = 180 + Math.abs(angle[finger]);

                } else if((x1 < x2) && (y1 < y2)){
                    angle[finger] = 360 - angle[finger];

                }
            } else{
                if((x2 < x1) && (y2 < y1)){
                    angle[finger] = 180 + Math.abs(angle[finger]);

                } else if(x2 < x1){
                    angle[finger] = 90 + Math.abs(angle[finger]);

                } else if(y2 < y1){
                    angle[finger] = 270 + Math.abs(angle[finger]);

                } else{
                    angle[finger] = angle[finger];

                }
            }
            point2Coords[finger][0] = x2;
            point2Coords[finger][1] = y2;
        }

        if(!isLeftHand){

        }

        mangles = angle;
        mcoords = point2Coords;
    }

    // Initialise the model
    private MappedByteBuffer loadModelFile(String modelName) throws IOException {
        String MODEL_ASSETS_PATH = modelName;
        AssetFileDescriptor assetFileDescriptor = mContext.getAssets().openFd(MODEL_ASSETS_PATH) ;
        FileInputStream fileInputStream = new FileInputStream( assetFileDescriptor.getFileDescriptor() ) ;
        FileChannel fileChannel = fileInputStream.getChannel() ;
        long startoffset = assetFileDescriptor.getStartOffset() ;
        long declaredLength = assetFileDescriptor.getDeclaredLength() ;
        return fileChannel.map( FileChannel.MapMode.READ_ONLY , startoffset , declaredLength ) ;
    }

    // run the model in current thread
    private void runLandmarkModel(){
        String modelName = "hand_landmark.tflite";

        //To convert the image into the tensor format required by the TensorFlow Lite interpreter, create a TensorImage to be used as input:
        // Initialization code

        // Create an ImageProcessor with all ops required.
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        // Center crop the image to the largest square possible
                        //.add(new ResizeWithCropOrPadOp(size, size))
                        .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                        //.add(new Rot90Op(360/90))
                        .add(new NormalizeOp(0, 255)) // pixels must be in range 0 to 1
                        .build();

        // Create a TensorImage object, this creates the tensor the TensorFlow Lite interpreter needs
        TensorImage tImage = new TensorImage(DataType.FLOAT32); // set correct data type for the model

        // Analysis code for every frame
        // Preprocess the image
        tImage.load(mimageBitmap);
        tImage = imageProcessor.process(tImage);

        //Log.i("PIXEL = ", String.valueOf(imageBitmap.getPixel(350,350)));
        //Log.i("CONFIG = ", String.valueOf(imageBitmap.getConfig()));

        // Initialize interpreter with GPU delegate
        Interpreter.Options options = new Interpreter.Options();

        //It seems we only can use GPU delegates with android sdk > 21
        if((android.os.Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) && museGPU){

            CompatibilityList compatList = new CompatibilityList();

            if(compatList.isDelegateSupportedOnThisDevice()){
                // if the device has a supported GPU, add the GPU delegate
                GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
                GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
                options.addDelegate(gpuDelegate);
            } else {
                // if the GPU is not supported, run on 4 threads
                options.setNumThreads(4);
                Toast.makeText(mContext, "GPU Not Supported!", Toast.LENGTH_LONG).show();
            }
        } else {
            //Do not use GPU delegates, use 4 threads instead
            options.setNumThreads(4);
            if(museGPU){
                Toast.makeText(mContext, "GPU Not Supported!", Toast.LENGTH_LONG).show();
            }
        }

        try (Interpreter interpreter = new Interpreter(loadModelFile(modelName), options)) {

            ////////////////////OUTPUT TENSORS /////////////////////////
            Map outputProbabilityBuffers = new HashMap<>();
            float[][] output0 = new float[1][63];
            float[][] output1 = new float[1][1];
            float[][] output2 = new float[1][1];
            outputProbabilityBuffers.put(0, output0);
            outputProbabilityBuffers.put(1, output1);
            outputProbabilityBuffers.put(2, output2);

            //{your_regular_input }
            Object[] inputs = {tImage.getBuffer()};

            //get the final prediction
            interpreter.runForMultipleInputsOutputs(inputs, outputProbabilityBuffers);

            mpredCoords = (float[][]) (outputProbabilityBuffers.get(0));

            //interpreter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

        computeAngles(mpredCoords, misLeftHand, mopenCVConvention);

    }

    // run the model in a new thread
    private class runLandmarkModelBackground extends AsyncTask {
        @Override
        protected void onPreExecute() {
            super.onPreExecute();


        }

        @Override
        protected Object doInBackground(Object[] objects) {

            String modelName = "hand_landmark.tflite";

            //To convert the image into the tensor format required by the TensorFlow Lite interpreter, create a TensorImage to be used as input:
            // Initialization code

            // Create an ImageProcessor with all ops required.
            ImageProcessor imageProcessor =
                    new ImageProcessor.Builder()
                            // Center crop the image to the largest square possible
                            //.add(new ResizeWithCropOrPadOp(size, size))
                            .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                            //.add(new Rot90Op(360/90))
                            .add(new NormalizeOp(0, 255)) // pixels must be in range 0 to 1
                            .build();

            // Create a TensorImage object, this creates the tensor the TensorFlow Lite interpreter needs
            TensorImage tImage = new TensorImage(DataType.FLOAT32); // set correct data type for the model

            // Analysis code for every frame
            // Preprocess the image
            tImage.load(mimageBitmap);
            tImage = imageProcessor.process(tImage);

            //Log.i("PIXEL = ", String.valueOf(imageBitmap.getPixel(350,350)));
            //Log.i("CONFIG = ", String.valueOf(imageBitmap.getConfig()));

            // Initialize interpreter with GPU delegate
            Interpreter.Options options = new Interpreter.Options();

            //It seems we only can use GPU delegates with android sdk > 21
            if(android.os.Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP){

                CompatibilityList compatList = new CompatibilityList();

                if(compatList.isDelegateSupportedOnThisDevice()){
                    // if the device has a supported GPU, add the GPU delegate
                    GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
                    GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
                    options.addDelegate(gpuDelegate);
                } else {
                    // if the GPU is not supported, run on 4 threads
                    options.setNumThreads(4);
                }
            } else {
                //Do not use GPU delegates, use 4 threads instead
                options.setNumThreads(4);
            }

            try (Interpreter interpreter = new Interpreter(loadModelFile(modelName), options)) {

                ////////////////////OUTPUT TENSORS /////////////////////////
                Map outputProbabilityBuffers = new HashMap<>();
                float[][] output0 = new float[1][63];
                float[][] output1 = new float[1][1];
                float[][] output2 = new float[1][1];
                outputProbabilityBuffers.put(0, output0);
                outputProbabilityBuffers.put(1, output1);
                outputProbabilityBuffers.put(2, output2);

                //{your_regular_input }
                Object[] inputs = {tImage.getBuffer()};

                //get the final prediction
                interpreter.runForMultipleInputsOutputs(inputs, outputProbabilityBuffers);

                mpredCoords = (float[][]) (outputProbabilityBuffers.get(0));

                //interpreter.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

            return null;
        }

        @Override
        protected void onPostExecute(Object o) {
            super.onPostExecute(o);

            computeAngles(mpredCoords, misLeftHand, mopenCVConvention);

        }
    }
}
