package com.jonajo.glamscan;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.PopupMenu;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.FileNotFoundException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    static Bitmap imageBitmap;
    static Bitmap color;
    static Bitmap nailmask;
    static Bitmap rthumbmask;
    static Bitmap lthumbmask;
    public ImageView Iv;
    public ImageView Iv2;
    public TextView tv1;
    public TextView tv2;
    public TextView tv3;
    public TextView tv4;
    public TextView tvpatches;
    public TextView tvSSD;
    public Switch gpuSwitch;
    public Switch leftSwitch;
    public Button runButton;
    public Mat mat8UC;
    public Mat colormat8UC;
    public Mat nailmaskmat8UC;
    public Mat rthumbmaskmat8UC;
    public Mat lthumbmaskmat8UC;
    double startTime;
    double endTime;
    double totalstartTime;
    double totalendTime;
    int illum_patches=5;
    static Uri photoURI;
    public String ssdVersion = "tflite_nail_640.tflite";
    float minscore = 0.81f;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (!OpenCVLoader.initDebug()) {
            // Handle initialization error
        }

        tv1 = (TextView) findViewById(R.id.textView);
        tv2 = (TextView) findViewById(R.id.textView2);
        tv3 = (TextView) findViewById(R.id.textView3);
        tv4 = (TextView) findViewById(R.id.textView4);
        tvpatches = (TextView) findViewById(R.id.tvpatches);
        tvSSD= (TextView) findViewById(R.id.tvSSD);
        Iv = (ImageView) findViewById(R.id.imageView);
        //Iv2 = (ImageView) findViewById(R.id.imageView2);

        tvpatches.setText("Illum patches: " + illum_patches);
        tvSSD.setText("SSD 640X640");

        ////int drawableID = this.getResources().getIdentifier("finger", "drawable", getPackageName());
        imageBitmap = BitmapFactory.decodeResource(this.getResources(), R.drawable.handmodel);
        //Iv.setImageBitmap(imageBitmap);

        ///////////////////////////////////////////////////////////////////////////////////
        /////////////////////// Testing Watershed /////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////
        //Mat mat = new Mat();
        //Bitmap bmp32 = imageBitmap.copy(Bitmap.Config.RGB_565, true);
        ////bmp32.setConfig(Bitmap.Config.RGB_565);
        //Utils.bitmapToMat(bmp32, mat);
        ////Log.i("tipo mat", String.valueOf(mat.type()));
        //Mat mat8UC = new Mat(mat.size(), CvType.CV_8UC(3));
        //Imgproc.cvtColor(mat, mat8UC, Imgproc.COLOR_RGBA2RGB);
        ////mat.convertTo(mat8UC, CvType.CV_8U);
        ////Log.i("despues tipo mat", String.valueOf(mat8UC.type()));
        ////Log.i("buen tipo mat", String.valueOf(Mat.ones(640, 640, CvType.CV_8UC(3)).type()));
        ////Log.i("channels", String.valueOf(mat8UC.channels()));

        ////mat = ImageProcessing.Watershed(Mat.ones(640, 640, CvType.CV_8UC(3)));
        //mat8UC = ImageProcessing.Watershed(mat8UC);
        //Iv2.setImageBitmap(ImageProcessing.convertMatToBitMap(mat8UC));

        ////try {
            ////mat = Utils.loadResource(this, R.drawable.finger, CvType.CV_8UC4);
        ////} catch (IOException e) {
            ////e.printStackTrace();
        ////}

        ////mat = Imgcodecs.imread(getResources().getDrawable(R.drawable.finger).toString());
        ///////////////////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////////////////
        ////////////////////// Testing HandLandmark ///////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////
        /*startTime = System.nanoTime();
        handLandmarkModel landmarkModel = new handLandmarkModel(imageBitmap, getApplicationContext(), true, true, true);
        landmarkModel.predictAngles();
        float[] angles = landmarkModel.getAngles();
        float[][] point2coords = landmarkModel.getCoords();
        endTime = System.nanoTime();
        tv1.setText("Landmark pred Time = " + (endTime - startTime)*0.000001 + "ms");*/
        /*Mat resultMat = landmarkModel.getMatResults();

        Size size = new Size(240,240);

        Mat mat = new Mat();
        Bitmap bmp32 = imageBitmap.copy(Bitmap.Config.RGB_565, true);
        Utils.bitmapToMat(bmp32, mat);
        Mat mat8UC = new Mat(mat.size(), CvType.CV_8UC(3));
        Imgproc.cvtColor(mat, mat8UC, Imgproc.COLOR_RGBA2RGB);
        Imgproc.resize(mat8UC, mat8UC, size);
        Bitmap orgImage = ImageProcessing.convertMatToBitMap(mat8UC);
        Iv.setImageBitmap(orgImage);

        Imgproc.resize(resultMat, resultMat, size);
        ImageProcessing.rotateImage(resultMat, 45);
        Bitmap resultBitmap = ImageProcessing.convertMatToBitMap(resultMat);
        Iv2.setImageBitmap(resultBitmap);

        Log.i("results", ": ");
        for(int i = 0; i<angles.length; i++){
            Log.i("angle =", String.valueOf(angles[i]));
        }*/
        ///////////////////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////////////////
        ////////////////////// Testing SSD Model //////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////
        /*startTime = System.nanoTime();
        SSDModel ssdModel = new SSDModel(imageBitmap, getApplicationContext(), true);
        ssdModel.predictBoxes();
        float[][] boxes = ssdModel.getBoxes();
        float[] scores = ssdModel.getScores();
        endTime = System.nanoTime();
        tv2.setText("SSD pred Time = " + (endTime - startTime)*0.000001 + "ms");*/
        /*Log.i("results", ": ");
        for(int i = 0; i<scores.length; i++){
            Log.i("score =", String.valueOf(scores[i]));
        }*/
        ///////////////////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////////////////
        ////////////////////// Testing Nails painting //////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////
        // image
        Mat mat = new Mat();
        Bitmap bmp32 = imageBitmap.copy(Bitmap.Config.RGB_565, true);
        Utils.bitmapToMat(bmp32, mat);
        mat8UC = new Mat(mat.size(), CvType.CV_8UC(3));
        Imgproc.cvtColor(mat, mat8UC, Imgproc.COLOR_RGBA2RGB);

        // color
        color = BitmapFactory.decodeResource(this.getResources(), R.drawable.bright_red);
        Mat colormat = new Mat();
        Bitmap colorbmp32 = color.copy(Bitmap.Config.RGB_565, true);
        Utils.bitmapToMat(colorbmp32, colormat);
        colormat8UC = new Mat(colormat.size(), CvType.CV_8UC(3));
        Imgproc.cvtColor(colormat, colormat8UC, Imgproc.COLOR_RGBA2RGB);

        // nails mask
        nailmask = BitmapFactory.decodeResource(this.getResources(), R.drawable.newnail_model);
        Mat nailmaskmat = new Mat();
        Bitmap nailmaskbmp32 = nailmask.copy(Bitmap.Config.RGB_565, true);
        Utils.bitmapToMat(nailmaskbmp32, nailmaskmat);
        nailmaskmat8UC = new Mat(nailmaskmat.size(), CvType.CV_8UC(3));
        Imgproc.cvtColor(nailmaskmat, nailmaskmat8UC, Imgproc.COLOR_RGBA2RGB);

        // right thumb mask
        rthumbmask = BitmapFactory.decodeResource(this.getResources(), R.drawable.rthumb_modelv3);
        Mat rthumbmaskmat = new Mat();
        Bitmap rthumbmaskbmp32 = rthumbmask.copy(Bitmap.Config.RGB_565, true);
        Utils.bitmapToMat(rthumbmaskbmp32, rthumbmaskmat);
        rthumbmaskmat8UC = new Mat(rthumbmaskmat.size(), CvType.CV_8UC(3));
        Imgproc.cvtColor(rthumbmaskmat, rthumbmaskmat8UC, Imgproc.COLOR_RGBA2RGB);

        // left thumb mask
        lthumbmask = BitmapFactory.decodeResource(this.getResources(), R.drawable.lthumb_modelv3);
        Mat lthumbmaskmat = new Mat();
        Bitmap lthumbmaskbmp32 = lthumbmask.copy(Bitmap.Config.RGB_565, true);
        Utils.bitmapToMat(lthumbmaskbmp32, lthumbmaskmat);
        lthumbmaskmat8UC = new Mat(lthumbmaskmat.size(), CvType.CV_8UC(3));
        Imgproc.cvtColor(lthumbmaskmat, lthumbmaskmat8UC, Imgproc.COLOR_RGBA2RGB);

        /*Boolean isleftHand = false;
        String nailsize = "medium";
        float[] orgsize = {(float) mat8UC.size().width, (float) mat8UC.size().height};
        int nailpatches = 4;
        int illum_ofset = 4;
        int size = 640;
        float minscore = 0.81f;
        Boolean showBoxes = false;

        startTime = System.nanoTime();
        Mat result = ImageProcessing.paintNail(mat8UC, colormat8UC, nailmaskmat8UC, rthumbmaskmat8UC, lthumbmaskmat8UC,
                boxes, scores, angles, point2coords, isleftHand, nailsize, orgsize, nailpatches, illum_ofset,
                size, minscore, showBoxes);
        endTime = System.nanoTime();
        tv3.setText("Coloring Time = " + (endTime - startTime)*0.000001 + "ms");*/

        //Size resultsize = new Size(240,240);
        //Size resultsize = new Size(Iv.getMeasuredWidth(),Iv.getMeasuredHeight());
        //Log.i("width", String.valueOf(Iv.getMeasuredWidth()));
        //Log.i("height", String.valueOf(Iv.getMeasuredHeight()));
        //Imgproc.resize(result, result, resultsize);
        /*Bitmap resultBitmap = ImageProcessing.convertMatToBitMap(result);
        Iv.setImageBitmap(resultBitmap);*/
        ///////////////////////////////////////////////////////////////////////////////////

        re_runModel(false, false, ssdVersion);

        ///////////////////////////////////////////////////////////////////////////////////
        ////////////////////// Testing Listener //////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////
        gpuSwitch = (Switch) findViewById(R.id.switch1);
        leftSwitch = (Switch) findViewById(R.id.switch2);
        runButton = (Button) findViewById(R.id.button);
        runButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                re_runModel(gpuSwitch.isChecked(), leftSwitch.isChecked(), ssdVersion);
            }
        });
        ///////////////////////////////////////////////////////////////////////////////////

        tvpatches.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                patchesMenu(v);
            }
        });

        tvSSD.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                ssdMenu(v);
            }
        });
    }

    public void re_runModel(Boolean useGPU, Boolean leftHand, String ssdVersion){
        totalstartTime = System.nanoTime();

        // landmark model
        startTime = System.nanoTime();
        handLandmarkModel landmarkModel = new handLandmarkModel(imageBitmap, getApplicationContext(), true, true, useGPU);
        landmarkModel.predictAngles();
        float[] angles = landmarkModel.getAngles();
        float[][] point2coords = landmarkModel.getCoords();
        endTime = System.nanoTime();
        tv1.setText("Landmark prediction Time = " + (endTime - startTime)*0.000001 + "ms");

        //SSD model
        startTime = System.nanoTime();
        SSDModel ssdModel = new SSDModel(imageBitmap, getApplicationContext(), useGPU, ssdVersion);
        ssdModel.predictBoxes();
        float[][] boxes = ssdModel.getBoxes();
        float[] scores = ssdModel.getScores();
        endTime = System.nanoTime();
        tv2.setText("SSD prediction Time = " + (endTime - startTime)*0.000001 + "ms");

        Boolean isleftHand = leftHand;
        String nailsize = "medium";
        float[] orgsize = {(float) mat8UC.size().width, (float) mat8UC.size().height};
        int nailpatches = illum_patches;
        int illum_ofset = 6;
        int resolution = 1280;
        Boolean showBoxes = false;

        startTime = System.nanoTime();
        Mat result = ImageProcessing.paintNail(mat8UC, colormat8UC, nailmaskmat8UC, rthumbmaskmat8UC, lthumbmaskmat8UC,
                boxes, scores, angles, point2coords, isleftHand, nailsize, orgsize, nailpatches, illum_ofset,
                resolution, minscore, showBoxes);
        endTime = System.nanoTime();
        tv3.setText("Coloring Time = " + (endTime - startTime)*0.000001 + "ms");

        totalendTime = System.nanoTime();
        tv4.setText("Total Time = " + (totalendTime - totalstartTime)*0.000001 + "ms");

        Bitmap resultBitmap = ImageProcessing.convertMatToBitMap(result);
        Iv.setImageBitmap(resultBitmap);
    }

    public void patchesMenu(View v) {
        PopupMenu popup = new PopupMenu(getApplicationContext(), v);

        popup.setOnMenuItemClickListener(new PopupMenu.OnMenuItemClickListener() {
            @Override
            public boolean onMenuItemClick(MenuItem item) {
                if (item.getItemId() == R.id.patch1) {
                    illum_patches=1;
                    tvpatches.setText("Illum patches: " + illum_patches);
                    return true;
                } else if (item.getItemId() == R.id.patch2) {
                    illum_patches=2;
                    tvpatches.setText("Illum patches: " + illum_patches);
                    return true;
                } else if (item.getItemId() == R.id.patch3) {
                    illum_patches=3;
                    tvpatches.setText("Illum patches: " + illum_patches);
                    return true;
                } else if (item.getItemId() == R.id.patch4) {
                    illum_patches=4;
                    tvpatches.setText("Illum patches: " + illum_patches);
                    return true;
                } else if (item.getItemId() == R.id.patch5) {
                    illum_patches=5;
                    tvpatches.setText("Illum patches: " + illum_patches);
                    return true;
                } else{
                    return false;
                }
            }
        });
        popup.inflate(R.menu.patches_menu);
        popup.show();

    }

    public void ssdMenu(View v) {
        PopupMenu popup = new PopupMenu(getApplicationContext(), v);

        popup.setOnMenuItemClickListener(new PopupMenu.OnMenuItemClickListener() {
            @Override
            public boolean onMenuItemClick(MenuItem item) {
                if (item.getItemId() == R.id.ssd320) {
                    ssdVersion = "ssd_mobilenet_v2_fingernails_quant.tflite";
                    tvSSD.setText("SSD 320X320");
                    minscore = 0.36f;
                    return true;
                } else if (item.getItemId() == R.id.ssd640) {
                    ssdVersion = "tflite_nail_640.tflite";
                    tvSSD.setText("SSD 640X640");
                    minscore = 0.81f;
                    return true;
                }  else{
                    return false;
                }
            }
        });
        popup.inflate(R.menu.ssd_version_menu);
        popup.show();

    }

    private static final int RESULT_LOAD_IMG = 1;
    private static final int PERMISSION_REQUEST_CODE_GALERIA = 1000;
    public void Gallery(View v){

        if (ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            //ask permissions
            ActivityCompat.requestPermissions(MainActivity.this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, PERMISSION_REQUEST_CODE_GALERIA);

        } else{
            Intent photoPickerIntent = new Intent(Intent.ACTION_PICK);
            photoPickerIntent.setType("image/*");
            startActivityForResult(photoPickerIntent, RESULT_LOAD_IMG);
        }
    }

    @Override
    protected void onActivityResult(int reqCode, int resultCode, Intent data) {
        super.onActivityResult(reqCode, resultCode, data);

        // Check which request we're responding to
        //Gallery request
        if(reqCode == RESULT_LOAD_IMG ){
            if (resultCode == RESULT_OK) {
                try {
                    photoURI = data.getData();
                    final InputStream imageStream = getContentResolver().openInputStream(photoURI);
                    imageBitmap = BitmapFactory.decodeStream(imageStream);
                    Iv.setImageBitmap(imageBitmap);

                    Mat mat = new Mat();
                    Bitmap bmp32 = imageBitmap.copy(Bitmap.Config.RGB_565, true);
                    Utils.bitmapToMat(bmp32, mat);
                    mat8UC = new Mat(mat.size(), CvType.CV_8UC(3));
                    Imgproc.cvtColor(mat, mat8UC, Imgproc.COLOR_RGBA2RGB);

                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                    Toast.makeText(this, "Something went wrong", Toast.LENGTH_LONG).show();
                }
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions,
                                           int[] grantResults) {
        switch (requestCode) {
            case PERMISSION_REQUEST_CODE_GALERIA:
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0 &&
                        grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    // Permission is granted. Continue the action or workflow
                    // in your app.

                    Intent photoPickerIntent = new Intent(Intent.ACTION_PICK);
                    photoPickerIntent.setType("image/*");
                    startActivityForResult(photoPickerIntent, RESULT_LOAD_IMG);

                }  else {
                    // Explain to the user that the feature is unavailable because
                    // the features requires a permission that the user has denied.
                    // At the same time, respect the user's decision. Don't link to
                    // system settings in an effort to convince the user to change
                    // their decision.
                    Toast.makeText(this, "You didn't give permissions!", Toast.LENGTH_LONG).show();
                }
                return;
        }
        // Other 'case' lines to check for other
        // permissions this app might request.
    }
}