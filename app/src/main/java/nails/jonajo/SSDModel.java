package nails.jonajo;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.Build;
import android.widget.Toast;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

public class SSDModel {

    private Bitmap mimageBitmap;
    private Context mContext;
    private float[][][] mpredBoxes = new float[1][10][4]; // BBoxes
    float[][] mpredClasses = new float[1][10];  // classes
    float[][] mpredScores = new float[1][10];  // Scores
    float[] mpredNumBoxes = new float[1];  // number of detected boxes
    private Boolean museGPU;
    private String mssdVersion;

    // Constructor
    public SSDModel(Bitmap imageBitmap, Context context, Boolean useGPU, String ssdVersion){
        this.mimageBitmap = imageBitmap;
        this.mContext = context; // pass getApplicationContext() when instantiating the class
        this.museGPU = useGPU;
        this.mssdVersion = ssdVersion;
    }

    // run the model
    public void predictBoxes(){

        runSSDModel();
    }

    // return coordinates of the predicted bounding boxes
    public float[][] getBoxes(){
        return mpredBoxes[0];
    }

    // return scores of the predicted bounding boxes
    public float[] getScores(){
        return mpredScores[0];
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
    private void runSSDModel(){
        String modelName = mssdVersion;
        int size;
        if(mssdVersion.equals("ssd_mobilenet_v2_fingernails_quant.tflite")){
            size=320;
        } else{
            size=640;
        }

        //To convert the image into the tensor format required by the TensorFlow Lite interpreter, create a TensorImage to be used as input:
        // Initialization code

        // Create an ImageProcessor with all ops required.
        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        // Center crop the image to the largest square possible
                        //.add(new ResizeWithCropOrPadOp(size, size))
                        .add(new ResizeOp(size, size, ResizeOp.ResizeMethod.BILINEAR))
                        //.add(new Rot90Op(360/90))
                        //.add(new NormalizeOp(0, 255))
                        .build();

        // Create a TensorImage object, this creates the tensor the TensorFlow Lite interpreter needs
        TensorImage tImage = new TensorImage(DataType.UINT8); // set correct data type for the model

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
                Toast.makeText(mContext, "GPU Not Supported!", Toast.LENGTH_SHORT).show();
            }
        } else {
            //Do not use GPU delegates, use 4 threads instead
            options.setNumThreads(4);
            if(museGPU){
                Toast.makeText(mContext, "GPU Not Supported!", Toast.LENGTH_SHORT).show();
            }
        }

        try (Interpreter interpreter = new Interpreter(loadModelFile(modelName), options)) {

            ////////////////////OUTPUT TENSORS /////////////////////////
            Map outputProbabilityBuffers = new HashMap<>();
            float[][][] output0 = new float[1][10][4]; // BBoxes
            float[][] output1 = new float[1][10];  // classes
            float[][] output2 = new float[1][10];  // Scores
            float[] output3 = new float[1];  // number of detected boxes
            outputProbabilityBuffers.put(0, output0);
            outputProbabilityBuffers.put(1, output1);
            outputProbabilityBuffers.put(2, output2);
            outputProbabilityBuffers.put(3, output3);

            //{your_regular_input }
            Object[] inputs = {tImage.getBuffer()};

            //get the final prediction
            interpreter.runForMultipleInputsOutputs(inputs, outputProbabilityBuffers);

            mpredBoxes = (float[][][]) (outputProbabilityBuffers.get(0));
            mpredClasses = (float[][]) (outputProbabilityBuffers.get(1));
            mpredScores = (float[][]) (outputProbabilityBuffers.get(2));
            mpredNumBoxes = (float[]) (outputProbabilityBuffers.get(3));

            //interpreter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
