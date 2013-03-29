/**********************************************************************************
* (c) Copyright 2010-2011, Sergio Diaz-Miguel Coca
*
* Files: face.h
* Author: Sergio Diaz-Miguel Coca <sdcoca@gmail.com>
*
* This API declaration of the "Face" C/C++ class for geometrical facial features
* extraction (characteristic points).
*
* Good Luck.
*
***********************************************************************************/

#ifndef FACE_H
#define FACE_H

#include "cv.h"


using namespace cv; //This will let us name OpenCv types as Mat, Point, etc. instead of cv::Mat, cv::Point, etc.

class Face
{
public:

    Mat& originalImg;
    Mat processedImg;  //Algorithms will work with this version of the image (rescaled, equalized, color transformed, rotated, etc.).


    //Image preprocesing parameters. To generate "processedImg" from originalImg.
    Mat rMat;       //Matrices de rotação direta e inversa.Usado junto com a função transformPoint pode calcular o ponto rotado.
    Mat irMat;
    double rotationAngle;   //Degrees.
    Point rotationCenter;
    double scale;
    bool bluringEnabled;

    //For face detection. Haar classifier.
    CascadeClassifier& cascade;
    String cascadeFilename;
    bool faceFound; //Indicates whether a face was found in originalImg. It makes no sense to read this value before calling "extracCharacteristicPoints()".


    //Clarifications:
    // - Left means the left part of the image (not of the face) from your poitn of view. So left eye will have lower values of X coordinate (coloumns) than the right eye, or eyebrow, nostrill and so on.
    // - Every data (coordinates, tilt angle, etc.) is given in absolute values with respect to the original image (originalImg).
    // - So if you want a transformed and normalized sequence of numbers version (e.g: tilt and scale corrected version) of the points you will have to do it yourself.
    // - To do that, you only have to scale the points and then use transformPoint() with rMat;

    //Face tilt in degrees. It is the angle of the line formed by left and rigth eyes.
    //TODO: global tilt. I believe there is an error in implementation of this. "roll" should be computed after transform of eye centers and not by adding with previous rotation (since rotation centers where different).
    Point location;
    Size size;
    double tilt;

    //Mouth points: Rigth, Left, Upper and Down lips.
    Point rlip, llip, ulip, dlip;

    //Nostrills: Left and Right
    Point lnstrl, rnstrl;

    //Eyebrows: Left and Right.
    //Each array contains 4 bounding points. Respectively from 0 to 3 indexes: Left, Right, Up and Down points. E.g: Upper Point of Left Eyebrow --> leb[2]
    Point leb[4], reb[4];

    //Eye carachteristic points. Respectively from 0 to 4 indexes: Left and Right eye corners, Up and Down eyelids, and Center of IRIS (not of the whole eye).
    Point lefcps[5], refcps[5];

    //Eyes' Centers. They are just aliases of lefcps[4] and refcps[4] for easyness. Using one or the other makes no diference at all.
    Point& cleye, &creye;


    //Constructor. Requires an Image where to find a face to be processed. This image will be final and will remain untouched. If you want to process another image you will have to create another instance of this class (Face).
    Face(Mat& faceImg, CascadeClassifier& cascade, double scale = 1, double rotationAngle = 0, Point rotationCenter = Point (0,0), bool blur = true);

    //This is basically the method the class was created for. It scans the image in order to find the biggest face, processes it and fills up this object attributes with the corresponding values of rlip,llip, leb, rleb, etc.
    //It will return the final value of the faceFound atribute.
    bool extractCharacteristicPoints ();
    bool extractCharacteristicPoints (double scale, double rotationAngle = 0, Point rotationCenter = Point (0,0), bool blur = true);


    //Useful static functions. Useful for any image processing, not just for face. This is make public just for this reason.
    static void transformPoint(Point orig, Point& dstn, Mat tmat);
    static void project(const Mat& mat, Mat* hp, Mat* vp = NULL);
    static bool findBoundingPoints(const vector<Point>& vp,Point* leftmost, Point* rightmost, Point* umost, Point* dmost);

    static void rescaledRectangle (Mat& img, const Rect& r, double scale, const Scalar& color, int thickness=1, int lineType=8, int shift=0);
    static void rescaledCircle (Mat& img, const Point& p, double radius, double scale, const Scalar& color, int thickness=1, int lineType=8, int shift=0);
    static void rescaledLine (Mat& img, Point p1, Point p2, double scale, const Scalar& color, int thickness = 1, int lineType = 8, int shift = 0);
    static void rescaledEllipse(Mat& img, Point center, Size& axes, double angle, double startAngle, double endAngle, double scale, Scalar color, int thickness = 1, int lineType = 8);


protected:
    bool initialized; //Indicates whether the init() method was already called.

    //It processes the original image to generate "processedImg" the class will work with.
    //It applies: rescaling, rotation, equalization, filtering and color transformations.
    //Should be called before extracCharacteristicPoints(). If not, it will be automatically called.
    inline bool init ();
    inline bool init (double scale, double rotationAngle = 0, Point rotationCenter = Point (0,0), bool blur = true);


    static inline bool extractMouthPoints (Mat mouthImg, Point& llip, Point& rlip, Point& ulip, Point& dlip);
    static inline bool extractNostrils (Mat noseImg, Point& lnstrl, Point& rnstrl);
    static inline bool extractEyebrows(Mat eyeRegionImg, bool doBluring, Point leb[4], Point reb[4], Point& cleye, Point& creye, Point lefcps [5], Point refcps [5]);
    static inline bool extractEyeCorners (Mat eyeImg, Point efcps [5], Point ceye);


};

#endif // FACE_H
