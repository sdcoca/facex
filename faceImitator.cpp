/**********************************************************************************
* (c) Copyright 2010-2011, Sergio Diaz-Miguel Coca
*
* Files: faceImitator.cpp
* Author: Sergio Diaz-Miguel Coca <sdcoca@gmail.com>
*
* This is an example of how to use the "Face" C/C++ class for geometrical facial features
* extraction (characteristic points).
*
* It can be used as a template for face expression imitation (AVATAR) by just
* including the imitation algorith in the section of main where it is indicated.
*
* Good Luck.
*
***********************************************************************************/

#define CV_NO_BACKWARD_COMPATIBILITY

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <string>

//For OpenCV types.
#include "cv.h"
#include "highgui.h"

//Our Face Points Extraction Library.
#include "face.h"


#ifdef _EiC
#define WIN32
#endif


using namespace std;
using namespace cv;




String cascadeName =
"./haarcascade_frontalface_alt.xml";


const int FRAME_WIDTH = 240;
const int FRAME_HEIGHT = 180;



bool rotationEnabled = true;
bool blurEnabled = true;
bool faceDetected = false;
double extractionTime;



CvCapture* capture = 0;
Mat image;
const String scaleOpt = "--scale=";
size_t scaleOptLen = scaleOpt.length();
const String cascadeOpt = "--cascade=";
size_t cascadeOptLen = cascadeOpt.length();
String inputName;


double scale = 3;

double timeToWait;
FILE* f;

bool showPointsEnabled = true;

int parseArgs ( int argc, const char** argv);
int initImageStream ();
int closeImageStream();
int getNextImage (Mat& img);
inline bool manageKeyInputMenuAndWait ();
void drawFacePoints (Mat& img, const Face face);
void sendToAvatar (Face& face);




int main( int argc, const char** argv )
{
    CascadeClassifier cascade;
    int error;

    //Parsing Arguments.
    error = parseArgs(argc, argv);
    if (0 != error)
        return error;
    

    //Initialize the stream of images. We abstract from whether it was from camera, video file, database or image file. We will get next image by calling "getNextImage()"
    initImageStream();


    //Loading file for face detection.
    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        cerr << "Usage: faceImitator [--cascade=\"<cascade_path>\"]\n"
            "   [--scale=<image scale>]\n"
            "   [filename|camera_index]\n" ;
        return -1;
    }


    cvMoveWindow("Captured from Cam", 900,50);
    cvMoveWindow("Preprocessed Image", 1000,300);


    Mat img;
    bool quit = false;
    double nextTilt=0;
    Point nextRotCenter;
    double t;
    while (!quit &&  getNextImage(img) == 0)
    {
        Mat draftImg;
        img.copyTo(draftImg);

        //Face detection and points extraction.
        t = (double)cvGetTickCount(); //To count extraction time.
        Face face (img, cascade, scale);


        //If autorotation for tracking
        if (nextTilt != 0 && rotationEnabled)
        {
            face.extractCharacteristicPoints(scale,nextTilt,nextRotCenter);
        }
        else
            face.extractCharacteristicPoints();

        t = (double)cvGetTickCount() - t;
        extractionTime = t/((double)cvGetTickFrequency()*1000.0); //in ms

        //Only for autorotation
        if (rotationEnabled)
        {
            if (face.faceFound)
            {
                nextTilt = -(face.tilt);
                nextRotCenter = face.cleye; //We center rotation on left eye.
            }else
                nextTilt = 0;
        }


        /***************************************************************************************************
        *
        *   Here it goes your imitation algorithms (e.g: Neural Network). Example:
        *---------------------------------------------------------------------------------------------------
        *
        *   //Process the extracted face points
        *   myNeuralNetwork.step (face);
        *
        *
        *   //Command the avatar with the neurally computed outputs.
        *   sendToAvatar (myNeuralNetwork.outputs);
        *
        *
        *---------------------------------------------------------------------------------------------------
        *   All those functions were invented. But this one below is an example for straight imitation:
        */
        sendToAvatar(face);
        /*
        ****************************************************************************************************/



        //Drawing the detected points on the image if required.
        if (showPointsEnabled)
            drawFacePoints (draftImg, face);

        cv::imshow( "Captured from Cam", draftImg );
        cv::imshow("Preprocessed Image", face.processedImg);

        quit = manageKeyInputMenuAndWait();

    }


    closeImageStream();

    cvDestroyWindow( "Captured from Cam");
    cvDestroyWindow( "Preprocessed Image");


    return 0;
}








//------------------------------------------------------------ Util Functions to Simplify Main() ------------------------------------//


int parseArgs ( int argc, const char** argv)
{
    for( int i = 1; i < argc; i++ )
    {
        if( cascadeOpt.compare( 0, cascadeOptLen, argv[i], cascadeOptLen ) == 0 )
        {
            cascadeName.assign( argv[i] + cascadeOptLen );
        }
        else if( scaleOpt.compare( 0, scaleOptLen, argv[i], scaleOptLen ) == 0 )
        {
            if( !sscanf( argv[i] + scaleOpt.length(), "%lf", &scale ) || scale < 1 )
                scale = 2;
        }
        else if( argv[i][0] == '-' )
        {
            cerr << "WARNING: Unknown option %s" << argv[i] << endl;
        }
        else
            inputName.assign( argv[i] );
    }

    return 0;

}

//Scans arguments so it can start capturing from camera (and configure it), or capturing from video file (avi), or image database (see below), or just an image filename.
int initImageStream ()
{
    if( inputName.empty() || (isdigit(inputName.c_str()[0]) && inputName.c_str()[1] == '\0') )
    {

        capture = cvCaptureFromCAM( inputName.empty() ? 0 : inputName.c_str()[0] - '0' );

        //This is not working with our Logitech HD Pro Webcam C910.
        cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH);
        cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT);

    }
    else if( inputName.size() )
    {
        image = imread( inputName, 1 );
        if( image.empty() )
            capture = cvCaptureFromAVI( inputName.c_str() );

        if( !capture && !inputName.empty() )
        {
            /* assume it is a text file containing the
            list of the image filenames to be processed - one per line */

            if( !(f = fopen( inputName.c_str(), "rt" )) )
            {
                cerr << "ERROR: Could not open file \"" << inputName.c_str() << "\"."<< endl;

                cerr << "Usage: faceImitator [--cascade=\"<cascade_path>\"]\n"
                    "   [--scale=<image scale>]\n"
                    "   [filename|camera_index]\n" ;

                return -1;
            }
        }
    }
    else
    {
        cerr << "ERROR: Could not load image or image database.\n"
                "It should be a text file containing the list of the image filenames to be processed - one per line"
        << endl;

        cerr << "Usage: faceImitator [--cascade=\"<cascade_path>\"]\n"
            "   [--scale=<image scale>]\n"
            "   [filename|camera_index]\n" ;
        return -1;
    }

    return 0; //OK
}


int closeImageStream()
{
    if (capture)
        cvReleaseCapture( &capture );

    if ( f != NULL )
        fclose(f);

    return 0;

}


int getNextImage (Mat& img)
{
    //If it is a video sequence (from file or camera).
    if( capture )
    {

        Mat frame;

        IplImage* iplImg = cvQueryFrame( capture );
        frame = iplImg;

        if( frame.empty() )
            return -1; //No frame found.

        if( iplImg->origin == IPL_ORIGIN_TL )
            frame.copyTo( img );
        else
            flip( frame, img, 0 );

        //The image in the screen looks like a mirror.
        flip( frame, img, 1 );

    }
    else
    {
        if( !image.empty() )
        {
            img = image;
        }
        else if( !inputName.empty() )
        {
            /* assume it is a text file containing the
            list of the image filenames to be processed - one per line */
            if( f )
            {
                char buf[1000+1];
                if( fgets( buf, 1000, f ) )
                {
                    int len = (int)strlen(buf);

                    while( len > 0 && isspace(buf[len-1]) )
                        len--;
                    buf[len] = '\0';

                    image = imread( buf, 1 );
                    if( !image.empty() )
                    {
                        img = image;
                    }
                }
                else {
                    fclose(f); //End of file reached.
                    return -3; //Indicates there are no more images left. Stream is ended.
                }
            }
            else{
                return -2; //Error reading database file. NULL pointer.
            }
        }
    }

    return 0; //Ok. Next image is available in "img".
}

inline bool manageKeyInputMenuAndWait ()
{
    char c;

    c = cv::waitKey(30);

    if ( c == 'q' || c == 'Q')   //QUIT
        return true;
    else if ( c == 'd' || c == 'D')   //Draw points
        showPointsEnabled = !showPointsEnabled;


    return false;

}

void drawFacePoints (Mat& img, const Face face)
{
    double scale = face.scale;

    //Lips
    Face::rescaledCircle(img,face.llip, 1, scale, CV_RGB(255,170,120),-1,CV_AA);
    Face::rescaledCircle(img,face.rlip, 1, scale, CV_RGB(255,170,120),-1,CV_AA);
    Face::rescaledCircle(img,face.ulip, 1, scale, CV_RGB(255,170,120),-1,CV_AA);
    Face::rescaledCircle(img,face.dlip, 1, scale, CV_RGB(255,170,120),-1,CV_AA);

    //Eyebrows
    for (int i = 0; i< 4; i++)
    {
        //Left
        Face::rescaledCircle(img,face.leb[i], 1, scale,CV_RGB(255,0,0),-1,CV_AA);

        //Right
        Face::rescaledCircle(img,face.reb[i], 1, scale,CV_RGB(255,0,0),-1,CV_AA);
    }


    //Eyes
    //Center Points
    Face::rescaledCircle(img,face.creye, 1, scale,CV_RGB(255,255,255),-1,CV_AA);
    Face::rescaledCircle(img,face.cleye, 1, scale,CV_RGB(255,255,255),-1,CV_AA);


    //Nosestrills
    Face::rescaledCircle(img,face.lnstrl, 1, scale, CV_RGB(0,255,0),-1,CV_AA);
    Face::rescaledCircle(img,face.rnstrl, 1, scale, CV_RGB(0,255,0),-1,CV_AA);



    //Tilt Angle
    // Drawing tilted lines on face limits
    Point uline1, uline2, dline1, dline2, lline1, lline2, rline1, rline2;
    Point puline1, puline2;
    Point p1,p2;

    Rect fr; //Face frame
    Face::transformPoint(face.location, p1, face.rMat);
    fr.x = p1.x;
    fr.y = p1.y;
    fr.width = face.size.width;
    fr.height = face.size.height;


    lline1.x = fr.x;
    lline2.x = lline1.x - 10;
    rline1.x = fr.x + fr.width - 1;
    rline2.x = rline1.x + 10;
    lline1.y = lline2.y = rline1.y = rline2.y = fr.y + std::max(0, (fr.height/2) - 1);

    uline1.x = uline2.x =  dline1.x = dline2.x = fr.x + std::max(0, (fr.width/2) - 1);
    uline1.y =  fr. y;
    uline2.y = uline1.y - 10;
    dline1.y =  fr. y + fr.height -1;
    dline2.y = dline1.y + 10;

    puline1 = uline1;
    puline2 = uline2;


    Face::transformPoint(uline1 , uline1, face.irMat);
    Face::transformPoint(uline2 , uline2, face.irMat);
    Face::transformPoint(dline1 , dline1, face.irMat);
    Face::transformPoint(dline2 , dline2, face.irMat);
    Face::transformPoint(lline1 , lline1, face.irMat);
    Face::transformPoint(lline2 , lline2, face.irMat);
    Face::transformPoint(rline1 , rline1, face.irMat);
    Face::transformPoint(rline2 , rline2, face.irMat);



    //Scaling to big img
    Face::rescaledLine(img, puline1, puline2, scale, CV_RGB(255,255, 255), 3, CV_AA);
    //Face::rescaledLine(img2, rotCenter, puline1, scale, CV_RGB(255,255, 255), 1, CV_AA);
    //Face::rescaledLine(img2, rotCenter, uline1, scale, CV_RGB(255,255, 255), 1, CV_AA);
    Face::rescaledLine(img, uline1, uline2, scale, CV_RGB(255,170,120), 3, CV_AA);
    Face::rescaledLine(img, dline1, dline2, scale, CV_RGB(255,170,120), 3, CV_AA);
    Face::rescaledLine(img, lline1, lline2, scale, CV_RGB(255,170,120), 3, CV_AA);
    Face::rescaledLine(img, rline1, rline2, scale, CV_RGB(255,170,120), 3, CV_AA);


    // Some Textual Info
    std::stringstream ss;
    ss << "Face tilt: " << setiosflags(ios::fixed) << setprecision(2) << face.tilt << " deg.";
    putText(img, ss.str(), Point (0,15), FONT_HERSHEY_PLAIN, 1, CV_RGB(255,255, 255), 1,CV_AA, false);

    ss.str("");
    ss << "Img. Proc. Time: " << setiosflags(ios::fixed) << setprecision(2) << extractionTime << " ms.";
    putText(img, ss.str(), Point (0,30), FONT_HERSHEY_PLAIN, 1, CV_RGB(255,255, 255), 1,CV_AA, false);

    ss.str("");
    ss << "Auto-rotation: " << (rotationEnabled? "ON": "OFF");
    putText(img, ss.str(), Point (0,45), FONT_HERSHEY_PLAIN, 1, CV_RGB(255,255, 255), 1,CV_AA, false);
}



clock_t commandsTimer = 0;
double angle = 0;
double filteredTilt = 0;
double filteredRFrown = 0;

void sendToAvatar (Face& face) {

    if (face.faceFound)
    {
        commandsTimer = (double)cvGetTickCount() - commandsTimer;
        double elapsedTime = commandsTimer/((double)cvGetTickFrequency()*1000.0); //in ms

        //SMILE
        double newAngle = double((face.llip.y - face.lnstrl.y)) / double(std::max((face.lnstrl.x - face.llip.x), 1));
        newAngle = std::abs(std::atan(newAngle) == NAN ? 90.0 : std::atan(newAngle) * (180.0/ M_PI));
        newAngle -= face.tilt;
        newAngle = newAngle == NAN ? 1 : newAngle;

        angle = (0.2*newAngle + 0.8*angle);

        filteredTilt = (0.2*face.tilt + 0.8*filteredTilt );




        //BROWS
        //Rotation of the image.
        Mat rotMat = getRotationMatrix2D(face.cleye, -face.tilt, 1);
        Point ruFrownp;
        Point rrFrownp, rlFrownp;
        Face::transformPoint(face.reb[3],ruFrownp, rotMat);
        Face::transformPoint(face.reb[1],rrFrownp, rotMat);
        Face::transformPoint(face.reb[0],rlFrownp, rotMat);


        Point luFrownp;
        Point lrFrownp, llFrownp;

        Face::transformPoint(face.leb[3],luFrownp, rotMat);
        Face::transformPoint(face.leb[1],lrFrownp, rotMat);
        Face::transformPoint(face.leb[0],llFrownp, rotMat);


        Point lnstrl,rnstrl;
        Face::transformPoint(face.lnstrl, lnstrl, rotMat);
        Face::transformPoint(face.rnstrl, rnstrl, rotMat);

        double noseMid = ((rnstrl.y + lnstrl.y) /2);
        double ruFrown =  noseMid - ruFrownp.y;
        double luFrown =  noseMid - luFrownp.y;


        ruFrown = ruFrown / std::max((rrFrownp.x-rlFrownp.x), 1); //Normalizing with brow length.
        ruFrown = (ruFrown-0.7) / 1.0  ;
        ruFrown *= 1.2; //Range of Frown param from -0.2 to 1.8
        ruFrown -= 0.5;

        luFrown = luFrown / std::max((lrFrownp.x-llFrownp.x), 1); //Normalizing with brow length.
        luFrown = (luFrown-0.7) / 1.0  ;
        luFrown *= 1.2; //Range of Frown param from -0.2 to 1.8
        luFrown -= 0.5;

        double bothBrows = (luFrown + ruFrown)/2;
        filteredRFrown = (0.2*bothBrows+ 0.8*filteredRFrown );


        //DEBUG
        //cout<<ruFrown<<endl;
        //cout<<lFrown<<endl;


        if (elapsedTime > 10) {

                double smile;
                smile = 1-((angle-25) / 40 ) ;
                smile *= 2.2;


            cout << "EXPR(0, tilt(" << filteredTilt << ") jaw(1) purse(0) smile(" <<
                    setiosflags(ios::fixed) << setprecision(1) <<  smile << ") brows("
                    << setiosflags(ios::fixed) << setprecision(1) << filteredRFrown << ") sneer(0) pout(0) cheek(.3))" << endl;

            commandsTimer = (double)cvGetTickCount() ;

        }
    }
}
