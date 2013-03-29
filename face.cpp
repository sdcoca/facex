/**********************************************************************************
* (c) Copyright 2010-2011, Sergio Diaz-Miguel Coca
*
* Files: face.cpp
* Author: Sergio Diaz-Miguel Coca <sdcoca@gmail.com>
*
* This is "Face" C/C++ class for geometrical facial features
* extraction (characteristic points).
*
* Good Luck.
*
***********************************************************************************/

#include "face.h"
#include "cv.h"



Face::Face(Mat& faceImg, CascadeClassifier& pCascade, double scale, double rotationAngle, Point rotationCenter, bool blur) : originalImg(faceImg), cascade(pCascade), cleye(lefcps[4]), creye(refcps[4])
{
    this->scale = scale;
    this->rotationAngle = rotationAngle;
    this->rotationCenter = rotationCenter;
    this->bluringEnabled = blur;

    this->faceFound = false;
    this->initialized = false;

    this->tilt = 0;

}

//This method should be always called BEFORE any other method call. It is not called during constructor for efficiency.
//It processes the original image to generate "processedImg" which is the actual image the class will work with.
//It applies: rescaling, rotation, equalization, filtering and color transformations.
//Should be called before extracCharacteristicPoints(). If not, it will be automatically called.
inline bool Face::init ()
{
    Mat& img = this->originalImg;

    Mat gray, dstnImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );

    //Color conversion to grayscale.
    cvtColor( img, gray, CV_BGR2GRAY );

    //Resizing with the given scale.
    resize( gray, dstnImg, dstnImg.size(), 0, 0, INTER_LINEAR );

    //Equalizing the histogram.
    equalizeHist( dstnImg, dstnImg );

    //Gaussian Filter to get rid of noise or excesive definition. Bluring.
    if (this->bluringEnabled)
        GaussianBlur( dstnImg, dstnImg, Size(3, 3), 2, 2);


    //Rotation of the image.
    this->rMat = getRotationMatrix2D(this->rotationCenter, this->rotationAngle, 1);
    Mat smallImg2;
    dstnImg.copyTo(smallImg2);
    warpAffine(smallImg2, dstnImg, this->rMat, dstnImg.size());

    //Generation of inverse rotation matrix for future uses. See transformPoint().
    invertAffineTransform(this->rMat,this->irMat);

    //Storing the final preprocessed image ready to be analized for face search and inspection.
    this->processedImg = dstnImg;

    //We indicate the object has been initialized so the rest of its API is available.
    this->faceFound = false; //We have to set this to false since the "processedImg" has changed.
    return (this->initialized = true);
}

//Lets initialize the object with new parameters.
inline bool Face::init (double scale, double rotationAngle, Point rotationCenter, bool blur)
{
    this->scale = scale;
    this->rotationAngle = rotationAngle;
    this->rotationCenter = rotationCenter;
    this->bluringEnabled = blur;

    return this->init();
}


//This is basically the method the class was created for. It scans the image in order to find the biggest face, processes it and fills up this object attributes with the corresponding values of rlip,llip, leb, rleb, etc.
//It will return the final value of the faceFound atribute.
bool Face::extractCharacteristicPoints ()
{
    Mat& pImg = this->processedImg; //Alias for easyness.

    vector<Rect> faces;
    vector<Rect>::const_iterator r;

    Mat mouthImg, noseImg, eyeRegionImg ;
    Rect mr, er, nr;

    Point p1, p2;
    Rect box;
    Mat imToFill;


    //Initializing if required.
    if (!this->initialized)
        this->init();


    // Starting Face Detection
    cascade.detectMultiScale( pImg, faces,
        1.1, 2, 0
        |CV_HAAR_FIND_BIGGEST_OBJECT                    //DETECT THE BIGGEST FACE. Only one.
        //|CV_HAAR_DO_ROUGH_SEARCH
        |CV_HAAR_SCALE_IMAGE
        ,
        Size(30, 30) );

    //End of detection


    // If a face was detected ..
    r = faces.begin();
    if (r != faces.end())
    {
        //Setting Face Atributes
        this->location.x = r->x;
        this->location.y = r->y;
        this->size.width = r->width;
        this->size.height= r->height;


        //Starting Face Points Extraction

        //MOUTH
        //A gente so vai procurar parte baixa da face.
        mr.x = r->x + (r->width*0.20) ;
        mr.y = r->y + (r->height*0.5);
        mr.width = r->width * 0.6 ;
        mr.height = r->height * 0.5;

        // Now we crop the mouth and extract its feature points.
        mouthImg = pImg(mr);
        extractMouthPoints (mouthImg, llip, rlip, ulip, dlip);


        //NOSE
        nr.x = r->x + (r->width*0.3) ;
        nr.y = r->y + (r->height*0.50);
        nr.width = r->width * 0.4 ;
        nr.height =  std::min((ulip.y + mr.y) - nr.y + 1, int(r->height * 0.20));
        noseImg = pImg(nr);
        extractNostrils (noseImg, lnstrl, rnstrl);

        rnstrl += nr.tl();
        lnstrl += nr.tl();

        //EYES and EYEBROWS
        er.x = r->x + (r->width*0.1) ;
        er.y = r->y + (r->height*0.1);
        er.width = r->width * 0.8 ;
        er.height =  std::min((nr.y - er.y + 1), int(r->height * 0.5));
        eyeRegionImg = pImg(er);


        extractEyebrows(eyeRegionImg, !(this->bluringEnabled), leb, reb, cleye, creye, lefcps, refcps);


        //Relocating coordinates to the original image (rotated, rescaled)

        for (int i = 0; i< 4; i++){
            leb[i] += er.tl();
            reb[i] += er.tl();
        }

        for (int i = 0; i< 5; i++){
            lefcps[i] += er.tl();
            refcps[i] += er.tl();
        }

        llip += mr.tl();
        rlip += mr.tl();
        ulip += mr.tl();
        dlip += mr.tl();

        //Inverting Rotation if required.
        if (this->rotationAngle != 0 )
        {

            transformPoint(llip , llip , irMat);
            transformPoint(rlip , rlip , irMat);
            transformPoint(ulip , ulip , irMat);
            transformPoint(dlip , dlip , irMat);

            transformPoint(lnstrl, lnstrl, irMat);
            transformPoint(rnstrl, rnstrl, irMat);

            for (int i = 0; i< 5; i++){
                if (i<4){
                transformPoint(leb[i], leb[i], irMat);
                transformPoint(reb[i], reb[i], irMat);
                }

                transformPoint(lefcps[i], lefcps[i], irMat);
                transformPoint(refcps[i], refcps[i], irMat);
            }

            transformPoint(this->location, this->location, irMat);

            //this->tilt = this->rotationAngle + rollInc; //This is wrong. Shouldn be computed here. "rollInc should be computed after theses transformation to give the real tilt.
            //cout << "Global Tilt Angle eyes aprox: " << (roll) << " deg. centered on left eye." <<  endl;
        }
        //else
        //    this->tilt = rollInc; //Not necessary .. wrong too.

        //Computing global face tilt angle using eyes'axis. Left eye considered the center.
        double angle = double((cleye.y - creye.y)) / double(std::max((creye.x - cleye.x),1));
        angle = std::atan(angle) == NAN ? 90.0 : std::atan(angle) * (180.0/ M_PI);
        this->tilt = angle == NAN ? 0 : angle;

        //cout << "Local Tilt Angle eyes aprox: " << (*froll) << " deg. centered on left eye." <<  endl;

        return (this->faceFound = true);
    }
    else
        return (this->faceFound=false);

}


bool Face::extractCharacteristicPoints (double scale, double rotationAngle, Point rotationCenter , bool blur )
{
    this->scale = scale;
    this->rotationAngle = rotationAngle;
    this->rotationCenter = rotationCenter;
    this->bluringEnabled = blur;

    this->faceFound = false;
    this->initialized = false;

    return extractCharacteristicPoints();
}


inline bool Face::extractMouthPoints (Mat mouthImg, Point& llips, Point& rlips, Point& ulips, Point& dlips)
{
    Mat mouthImgTh, mouthSobel, kernel(1,5,CV_8UC1,1);

    //Sobel filter. Horizontal, 2nd order, single channel 255 bit-depth, 3x3 kernel.
    Sobel(mouthImg, mouthSobel, CV_8UC1, 0, 2, 3);


    //Thresholding so only hard edges stay.Automatic threshold estimated with Otsu algorithm. Both numbers 255 are ignored (dummy).
    threshold(mouthSobel, mouthImgTh, 255,255, THRESH_BINARY | THRESH_OTSU);

    Mat mouthAdapTh;
    adaptiveThreshold(mouthSobel,mouthAdapTh,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,((mouthSobel.size().width) % 2 == 1? (mouthSobel.size().width) : (mouthSobel.size().width) + 1 ),-15);


    vector<vector<Point> > contours;
    Rect bigBox;
    Mat imContours;

    mouthAdapTh.copyTo(imContours);
    findContours(imContours,contours,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);

    int cind;
    for( uint  j = 0 ; j < contours.size(); j++)
    {

        Mat matcnt (contours[j]);
        Rect bbox = boundingRect (matcnt);

        if (bbox.width > bigBox.width) {
            // O novo pasa a maior
            bigBox = bbox;
            cind = j;
        }
    }


    if (contours.size()>0)
        Face::findBoundingPoints(contours.at(cind),&llips,&rlips,&ulips, &dlips);
    else
        return false; //The image got totally black. No mouth detected.



    //Finding Upper lip y coordinate. The longest horizontal edge within the brightest.
    //integral(mouthSobel, mouthIntegral);
    Mat mouthEroded;
    erode(mouthAdapTh, mouthEroded, kernel);
    Mat lip2 = mouthEroded(bigBox);
    Mat vPrj (lip2.size().height, 1, CV_32FC1); //One coloumn
    Point p1,p2;
    double pv1,pv2;

    Face::project(lip2, NULL, &vPrj);

    minMaxLoc(vPrj,&pv2, &pv1,&p2, &p1);

    ulips.y = p1.y + bigBox.tl().y;

    //Adjust de upper and lower lip 'x' coordinates to the middle. Otherwise they will be moving with noise.
    ulips.x = int((rlips.x + llips.x)/2);




    /* Detecting Lower Lip (dlip) */
    Mat lipmap;
    Mat mouthComp = Mat(mouthImg.size(),mouthImg.type(), CV_RGB(255,255,255)) - mouthImg  ;
    addWeighted( mouthSobel, 0.5, mouthComp, 0.5, 0, lipmap);
    Mat mouthAdapTh3;
    adaptiveThreshold(lipmap, mouthAdapTh3,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,((mouthSobel.size().width) % 2 == 1? (mouthSobel.size().width) : (mouthSobel.size().width) + 1 ),-10);


    mouthAdapTh3.copyTo(imContours);
    //mouthImgTh.copyTo(imContours);
    findContours(imContours,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);

    //Sorting by area and darkness. TODO: improve heuristics to sort by (e.g: use area or width/mean value (max area and darkest)*relative down position)
    const int CAREAS_SZ = 500;
    int cAreasIdx [CAREAS_SZ];
    int cAreas [CAREAS_SZ];
    /*Computing areas and sorting (bubble sort .. improve)*/
    uint j;
    for( j = 0 ; j < contours.size() && j < CAREAS_SZ; j++)
    {
        int aux1, aux2, auxIdx1, auxIdx2;
        Mat matcnt (contours[j]);
        Rect bbox = boundingRect(matcnt);

        //Cost funtion to maximize: max(area * darkest)  // this does not work lowest located in 'y' coordinate
        double costf = bbox.width; //sum(lipmap(bbox)) [0] ;// (sum(eyeNeg(halfsideBox))[0]))  + ((bbox.y + (bbox.height/2)) / halfImContours.size().height ;

        //cout << "Area " << j << ": " << area << endl;

        //Sorting
        int i = 0;
        while (i < j && costf <= cAreas[i]) i++;

        aux1 = costf;
        auxIdx1 = j;
        while (i <= j)
        {
            aux2 = cAreas[i];
            cAreas[i] = aux1;
            aux1 = aux2;

            auxIdx2 = cAreasIdx[i];
            cAreasIdx[i] = auxIdx1;
            auxIdx1 = auxIdx2;

            i++;
        }
    }

    if (contours.size() >= 1) {
        Mat matcnt (contours.at(cAreasIdx[0]));
        bigBox = boundingRect(matcnt);
        Face::findBoundingPoints(contours.at(cAreasIdx[0]),NULL, NULL, NULL, &dlips);
        //Adjust de upper and lower lip 'x' coordinates to the middle. Otherwise they will be moving with noise.
        ulips.x = int((rlips.x + llips.x)/2);
        dlips.x = int((rlips.x + llips.x)/2);

    }

    return true;
}

inline bool Face::extractNostrils (Mat noseImg, Point& lnstrl, Point& rnstrl)
{

    Mat noseSobel;

    //Sobel filter. Horizontal, 2nd order, single channel 255 bit-depth, 3x3 kernel.
    Sobel(noseImg, noseSobel, CV_8UC1, 0, 2 , 3);

    /* Finding tighter bounding box for nose*/
    /*See "Precise detailed detection of facial features" 2008 IEEE.*/
    Rect nBox;
    Mat vPrj (noseImg.size().height, 1, CV_32FC1); //One coloumn
    Mat hPrj (1, noseImg.size().width, CV_32FC1); //One coloumn
    Point p1,p2;
    double pv1,pv2;

    Face::project(noseSobel,&hPrj, &vPrj);


    float vTh, hTh; /*Thresholding values from projections*/
    minMaxLoc(vPrj,&pv2, &pv1,&p2, &p1);
    vTh = pv1/2;
    minMaxLoc(hPrj,&pv2, &pv1,&p2, &p1);
    hTh = pv1/2;

    /*Finding vertical range*/
    int r = 0,cl;

    for (r=0;r < vPrj.size().height; r++)
        if (vPrj.at<float>(r,0) > vTh) break;
    nBox.y = r;
    for (r = vPrj.size().height-1; r >= nBox.y; r--)
        if (vPrj.at<float>(r,0) > vTh) break;
    nBox.height = r - nBox.y + 1;

    /*Finding horizontal range*/
    for (cl=0;cl < hPrj.size().width; cl++)
        if (hPrj.at<float>(0,cl) > hTh) break;
    nBox.x = cl;
    for (cl = hPrj.size().width -1; cl >= nBox.x; cl--)
        if (hPrj.at<float>(0,cl) > hTh) break;
    nBox.width = cl - nBox.x + 1;

//DEBUG
//    rectangle(noseImg,nBox.tl(),nBox.br(),CV_RGB (255,0,0),1);

    /*Darkest point in each left and right half of the nose image (tight box)*/
    Mat noseStrImg;
    Rect halfNoseBox;


    halfNoseBox = nBox;
    if (halfNoseBox.width !=0 && halfNoseBox.height !=0){

        //Left half
        halfNoseBox.width = std::max(0.0, std::ceil(halfNoseBox.width * 0.5)); //TODO : ver cuando es 1.
        noseImg(halfNoseBox).copyTo(noseStrImg);
        minMaxLoc(noseStrImg,&pv2, &pv1,&p2, &p1);
        p2.x = p2.x + halfNoseBox.x;
        p2.y = p2.y + halfNoseBox.y;

        lnstrl = p2;

        //Right half
        halfNoseBox.x += std::max(0, halfNoseBox.width-1); //TODO BUG: aquí width se hace cero cuando nBox.widht es 1.
        noseImg(halfNoseBox).copyTo(noseStrImg);
        minMaxLoc(noseStrImg,&pv2, &pv1,&p2, &p1);
        p2.x = p2.x + halfNoseBox.x;
        p2.y = p2.y + halfNoseBox.y;
        //cout << "Nose Mi2 [x,y]:" << p2.x << ","<< p2.y<< endl;

        rnstrl = p2;
    }
/* DEBUG
    circle (noseImg, lnstrl, 2, CV_RGB(255,255,255),1);
    circle (noseImg, rqnstrl, 2, CV_RGB(255,255,255),1);
*/

    return true;
}

inline bool Face::extractEyebrows(Mat eyeRegionImg, bool doBluring, Point leb[4], Point reb[4], Point& cleye, Point& creye, Point lefcps [5], Point refcps [5])
{
    // Alias for eyebrows'points (left, right, up, down).
    Point&  lleb = leb[0];
    Point&  rleb = leb[1];
    Point&  uleb = leb[2];
    Point&  dleb = leb[3];
    Point&  lreb = reb[0];
    Point&  rreb = reb[1];
    Point&  ureb = reb[2];
    Point&  dreb = reb[3];


    //We will get the four widest areas of the contours extracted from the Otsu Thresholded image of the Sobel transform of the gaussian blured version (to get rid of foredhead wrinkles, hair, etc) of the original image.
    vector<vector<Point> > contours;
    Rect bigBox1, bigBox2;
    Mat imContours;

    Mat blur, blurSobel, blurSTh;

    eyeRegionImg.copyTo(blur);
    if (doBluring) GaussianBlur( eyeRegionImg , blur, Size(5, 5), 2, 2);

    //Sobel filter. Horizontal, 2nd order, single channel 255 bit-depth, 3x3 kernel.
    Sobel(blur, blurSobel,CV_8UC1, 0, 2 , 3);

    //Thresholding so only hard edges stay.Automatic threshold estimated with Otsu algorithm. Both numbers 255 are ignored (dummy).
    threshold(blurSobel, blurSTh, 255,255, THRESH_BINARY | THRESH_OTSU);
    //adaptiveThreshold(blurSobel, blurSTh,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,/*((eyesSobel.size().width) % 2 == 1? (eyesSobel.size().width) : (eyesSobel.size().width) + 1 )*/ 3,-2);

    blurSTh.copyTo(imContours);

/*
    cv::imshow( "SThEClose", imContours);
    cvMoveWindow("SThEClose", 1200, 300 );

    cv::imshow( "Blur", blur);
    cvMoveWindow("Blur", 1200, 500 );

    cv::imshow( "BlurSobel", blurSobel);
    cvMoveWindow("BlurSobel", 1200, 700 );
*/


    //Dividing eye region in left and right halfs. This made the algorithm more robust to face rotation.
    Rect halfsideBox;
    Mat eyeNeg =  Mat(eyeRegionImg.size(),eyeRegionImg.type(), CV_RGB(255,255,255)) - eyeRegionImg  ;
    for (int iteration = 0; iteration < 2; iteration++)
    {

        //Left half
        if ( iteration == 0 )
        {
            halfsideBox.x = 0;
            halfsideBox.y = 0;
            halfsideBox.width = imContours.size().width * 0.5;
            halfsideBox.height = imContours.size().height;
        }
        else
        {
            halfsideBox.x = std::floor(imContours.size().width * 0.5);
            halfsideBox.y = 0;
            halfsideBox.width = imContours.size().width * 0.5 - 1;
            halfsideBox.height = imContours.size().height;

        }

        Mat halfImContours = imContours(halfsideBox);

        findContours(halfImContours,contours,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);

        //Sorting by area and darkness. TODO: improve heuristics to sort by (e.g: use area or width/mean value (max area and darkest)*relative down position)
        const int CAREAS_SZ = 500;
        int cAreasIdx [CAREAS_SZ];
        int cAreas [CAREAS_SZ];
        /*Computing areas and sorting (bubble sort .. improve)*/
        uint j;
        for( j = 0 ; j < contours.size() && j < CAREAS_SZ; j++)
        {
            int aux1, aux2, auxIdx1, auxIdx2;
            Mat matcnt (contours[j]);
            Rect bbox = boundingRect(matcnt);

            //Cost funtion to maximize: max(area * darkest)
            double costf = sum(eyeNeg(halfsideBox)(bbox)) [0] ; //Other things tryed: (sum(eyeNeg(halfsideBox))[0]))  + ((bbox.y + (bbox.height/2)) / halfImContours.size().height ; //bbox.area();

            //cout << "Area " << j << ": " << area << endl;

            //Sorting
            int i = 0;
            while (i < j && costf <= cAreas[i]) i++;

            aux1 = costf;
            auxIdx1 = j;
            while (i <= j)
            {
                aux2 = cAreas[i];
                cAreas[i] = aux1;
                aux1 = aux2;

                auxIdx2 = cAreasIdx[i];
                cAreasIdx[i] = auxIdx1;
                auxIdx1 = auxIdx2;

                i++;
            }
        }


        if (contours.size() >= 2) {

            //Position 0 will have the eyebrow and 1 the eye.
            if (boundingRect(contours[cAreasIdx[1]]).y < boundingRect(contours[cAreasIdx[0]]).y)
            {
                int aux = cAreasIdx[0];
                cAreasIdx[0] = cAreasIdx[1];
                cAreasIdx[1] = aux;
            }

            if (iteration == 0)
            {
                findBoundingPoints(contours.at(cAreasIdx[0]),&lleb,&rleb,&uleb, &dleb);
                lleb += halfsideBox.tl();
                rleb += halfsideBox.tl();
                uleb += halfsideBox.tl();
                dleb += halfsideBox.tl();
            }
            else
            {
                findBoundingPoints(contours.at(cAreasIdx[0]),&lreb,&rreb,&ureb, &dreb);
                lreb += halfsideBox.tl();
                rreb += halfsideBox.tl();
                ureb += halfsideBox.tl();
                dreb += halfsideBox.tl();
            }

            //Extracting Eye Center. Considered the darkest point in eye contour. TODO: Try to improve with a geometric centroid of the darkest region.
            Rect eyeBox= boundingRect(Mat(contours[cAreasIdx[1]]));
            Point p1,p2;
            double pv1,pv2;

            minMaxLoc(blur(eyeBox),&pv2, &pv1,&p2, &p1);
            p2.x = p2.x + eyeBox.x;
            p2.y = p2.y + eyeBox.y;

            if (iteration == 0 )
                cleye = p2 + halfsideBox.tl(); // Center of Left Eye
            else
                creye = p2 + halfsideBox.tl(); // Center of Right Eye

            //DEBUG
            for (int i = 0; i < 2 && i < contours.size(); i++ )
            {
                Point laux ;
                Point raux ;
                Point uaux ;
                Point daux ;

                findBoundingPoints(contours[cAreasIdx[i]],&laux,&raux,&uaux, &daux);

                Rect br;
                br.x = laux.x;
                br.y = uaux.y;
                br.width = raux.x - br.x + 1;
                br.height = daux.y - br.y + 1;

                //DEBUG
                Scalar colorR;

                colorR = i >= 1? CV_RGB (120,120,120) : CV_RGB (255,255,255);

                rectangle(eyeNeg, br.tl() + halfsideBox.tl() ,br.br() + halfsideBox.tl(),colorR,1);

            }


            //DEBUG
            //findBoundingPoints(contours.at(cAreasIdx[1]),&lefcps[0],&lefcps[1],&lefcps[2], &lefcps[3]);
            //lefcps[0] += halfsideBox.tl();
            //lefcps[1] += halfsideBox.tl();
            //lefcps[2] += halfsideBox.tl();
            //lefcps[3] += halfsideBox.tl();
            //if (iteration == 0){
            //    debuglefcps = lefcps[3];
            //}
            //line(eyeRegionImg,Point(iteration * eyeRegionImg.size().width/2, lefcps[2].y), Point(eyeRegionImg.size().width/(2-iteration) -1, lefcps[2].y),  CV_RGB (255,255,255), 1);
            //circle (eyeRegionImg, lefcps[2], 0, CV_RGB (255,255,255),-1);
            //line(eyeRegionImg,Point(0, lefcps[3].y), Point(eyeRegionImg.size().width/2 -1, lefcps[3].y),  CV_RGB (255,255,255), 1);
            //circle (eyeRegionImg, lefcps[3], 0, CV_RGB (255,255,255),-1);



        }else
            return false; //Some eyebrow or eye was not found

        //DEBUG
        //cv::imshow( "NegEyes", eyeNeg );
        //cvMoveWindow("NegEyes", 1400, 300 );

    }

    //Finding right eye corners
    Rect ler; //Left Eye Region
    ler.x = 0;
    ler.y = std::max (0, std::min(eyeRegionImg.size().height-1, dleb.y + 1)); //Not including eyebrow point.
    ler.width = std::max(0, int(std::ceil(eyeRegionImg.size().width / 2.0)));
    ler.height = std::max(0, eyeRegionImg.rows - ler.y);

    //Left Eye facial Characteristic Points: left and right corners, up and down eyelids, and center  respectively from 0 to 4.
    if (cleye.y < ler.y) cleye.y = ler.y; //To be sure that eyecenter is in left eye region (ler) and below eyebrow.
    extractEyeCorners (eyeRegionImg(ler), lefcps, cleye - ler.tl());



     //Finding right eye corners
    Rect rer; //Left Eye Region
    rer.x = std::max(0, int(std::ceil(eyeRegionImg.size().width / 2.0)) - 1);
    rer.y = std::max (0, std::min(eyeRegionImg.size().height-1, dreb.y + 1)); //Not including eyebrow point.
    rer.width = std::max(0, eyeRegionImg.size().width - rer.x);
    rer.height = std::max(0, eyeRegionImg.rows - rer.y);

     //Right Eye facial Characteristic Points: left and right corners, up and down eyelids, and center  respectively from 0 to 4.
    if (creye.y < rer.y) creye.y = rer.y; //To be sure that eyecenter is in left eye region (ler) and below eyebrow.
    extractEyeCorners (eyeRegionImg(rer), refcps, creye - rer.tl());


    //Relocating points
    int i=5;
    while (0 < i--)
    {
        lefcps [i] += ler.tl();
        refcps [i] += rer.tl();
    }


/*
    cv::imshow( "EBlur", blured);
    cvMoveWindow("EBlur", 1200, 500 );


    cv::imshow( "SEyes", eyesSobel);
    cvMoveWindow("SEyes", 700, 500 );

    cv::imshow( "ThEyes", eyesTh );
    cvMoveWindow("ThEyes", 700,300);
    cv::imshow( "SThEyes", eyesSTh);
    cvMoveWindow("SThEyes", 1000, 300 );
*/

    return true; //OK
}

inline bool Face::extractEyeCorners (Mat eyeImg, Point efcps [5], Point ceye)
{
    Mat eyesSobel, eyesTh, eyesSTh;
    Mat eyeBlur;

    //equalizeHist( eyeImg, eyeImg );

    GaussianBlur( eyeImg , eyeBlur, Size(3,3), 2, 2 );

    //Sobel filter. Horizontal, 2nd order, single channel 255 bit-depth, 3x3 kernel.
    Sobel(eyeBlur, eyesSobel,CV_8UC1, 0, 2 , 3, 1, 0, BORDER_REPLICATE);

    //Thresholding so only hard edges stay.Automatic threshold estimated with Otsu algorithm. Both numbers 255 are ignored (dummy).
    //threshold(eyesSobel, eyesSTh, 255,255, THRESH_BINARY | THRESH_OTSU);
    adaptiveThreshold(eyesSobel, eyesSTh,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY,/*((eyesSobel.size().width) % 2 == 1? (eyesSobel.size().width) : (eyesSobel.size().width) + 1 )*/ 3,-2);


    //The extraction will be based on:
    //  - Sorting contours by area*darkness
    //  - Considering only the 4 first of those.
    //  - The first of them will be the upper eyelid contour.
    //  - The lowest located of them will be the lower eyelid.
    //  - Upper and lower eyelids contours could be the same one.
    //  - The bounding points of the union of these two contours are the 4 characteristic points of the eye.
    //  - The eye center is not extracted here. It is keeped the value from previous algorithms.
    vector<vector<Point> > contours;
    Mat imContours;

    eyesSTh.copyTo(imContours);
    findContours(imContours,contours,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);

    //Sorting by area and darkness.
    const int CAREAS_SZ = 500;
    int cAreasIdx [CAREAS_SZ];
    int cAreas [CAREAS_SZ];
    /*Computing areas and sorting (bubble sort .. improve)*/
    uint j;
    for( j = 0 ; j < contours.size() && j < CAREAS_SZ; j++)
    {
        int aux1, aux2, auxIdx1, auxIdx2;
        Mat matcnt (contours[j]);
        Rect bbox = boundingRect(matcnt);

        //Cost funtion to maximize: max(area * darkest)
        double costf = sum(eyesSTh(bbox)) [0] ; //Other things tryed: (sum(eyeNeg(halfsideBox))[0]))  + ((bbox.y + (bbox.height/2)) / halfImContours.size().height ; //bbox.area();

        //cout << "Area " << j << ": " << area << endl;

        //Sorting
        int i = 0;
        while (i < j && costf <= cAreas[i]) i++;

        aux1 = costf;
        auxIdx1 = j;
        while (i <= j)
        {
            aux2 = cAreas[i];
            cAreas[i] = aux1;
            aux1 = aux2;

            auxIdx2 = cAreasIdx[i];
            cAreasIdx[i] = auxIdx1;
            auxIdx1 = auxIdx2;

            i++;
        }
    }

    //We will get consider only the 4 first contours. Then the lowest located of those, will be the low eyelid.
    //The upper eyelid will be the first in sort.
    int uIdx = -1 , lIdx = -1;
    if (contours.size() > 0)
    {
        uIdx = cAreasIdx[0]; //Upper eyelid will be the widest and darkest

        //Low eyelid will be the lowest located among the 4 wider.
        int maxy = 0;
        for( j = 0 ; j < contours.size() && j < 4; j++)
        {
            Mat matcnt (contours[cAreasIdx[j]]);
            Rect bbox = boundingRect(matcnt);

            if (bbox.br().y > maxy )
            {
                maxy = bbox.br().y;
                lIdx = cAreasIdx[j];
            }
        }
    }



    Point laux, raux, uaux, daux;

    if (uIdx >=0 && lIdx >= 0)
    {
        findBoundingPoints(contours[uIdx],&efcps[0], &efcps[1], &efcps[2], &efcps[3]);
        findBoundingPoints(contours[lIdx],&laux, &raux, &uaux, &daux);

        //Getting leftmost and rightmost
        efcps[0] = (laux.x < efcps[0].x) ? laux : efcps[0];
        efcps[1] = (raux.x > efcps[1].x) ? raux : efcps[1];

        //Getting uppermost and downmost
        efcps[2] = (uaux.y < efcps[2].y) ? uaux : efcps[2];
        efcps[3] = (daux.y > efcps[3].y) ? daux : efcps[3];

        //Centering the upper and lower points
        efcps[2].x = (efcps[1].x + efcps[0].x) / 2;
        efcps[3].x = efcps[2].x;

        //Centero of the eye will be the same. Maybe improved in the future.
        efcps[4] = ceye;



    }else
        return false; //cerr << "Exception: NO EYE CORNERS DETECTED !! " << endl;

/*
    //DEBUG
            circle (eyeBlur, efcps[0], 0, CV_RGB (255,255,255),2);
            circle (eyeBlur, efcps[1], 0, CV_RGB (255,255,255),2);
            circle (eyeBlur, efcps[2], 0, CV_RGB (255,255,255),2);
            circle (eyeBlur, efcps[3], 0, CV_RGB (255,255,255),2);


    cv::imshow( "SEyes", eyesSobel);
    cvMoveWindow("SEyes", 700, 500 );

    cv::imshow( "SThEyes", eyesSTh);
    cvMoveWindow("SThEyes", 1000, 300 );

    cv::imshow( "EyeImg", eyeBlur);
    cvMoveWindow("EyeImg", 1100, 300 );
*/
    return 0;
}


void Face::transformPoint(Point orig, Point& dstn, Mat tmat)
{
    double x = orig.x;
    double y = orig.y;


    dstn.x = tmat.at<double>(Point (0,0))* x + tmat.at<double>(Point (1,0)) * y + tmat.at<double>(Point (2,0));
    dstn.y = tmat.at<double>(Point (0,1))* x + tmat.at<double>(Point (1,1)) * y + tmat.at<double>(Point (2,1));
}



void Face::project(const Mat& mat, Mat* hpp, Mat* vpp)
{
    Mat &hp = (*hpp), &vp = (*vpp); //To make easy use of vpp and hpp
    int r,cl; //row and coloumn indexes.


    if (&vp != NULL)
        for (r=0;r < vp.size().height; r++)
            vp.at<float>(r,0) = sum(mat.row(r))[0];

    if (&hp != NULL)
        for (cl=0;cl < hp.size().width; cl++)
            hp.at<float>(0,cl) = sum(mat.col(cl))[0];
}

bool Face::findBoundingPoints(const vector<Point>& vp,Point* leftmost, Point* rightmost, Point* umost, Point* dmost)
{

    int maxxi = 0; // max and min 'x' index
    int minxi = 0;
    int maxyi = 0; // max and min 'y' index
    int minyi = 0;

    for( uint  j = 0 ; j < vp.size(); j++)
    {
        const Point& p = vp.at(j);

        if (p.x < vp.at(minxi).x)
            minxi = j;

        if (p.x > vp.at(maxxi).x)
            maxxi = j;

        if (p.y < vp.at(minyi).y)
            minyi = j;

        if (p.y > vp.at(maxyi).y)
            maxyi = j;


        //cout << "x,y: " << p.x << "," << p.y <<endl;
    }

    if (leftmost != NULL) *leftmost = vp.at(minxi);
    if (rightmost!= NULL) *rightmost= vp.at(maxxi);
    if (umost != NULL) *umost = vp.at(minyi);
    if (dmost != NULL) *dmost = vp.at(maxyi);

    return true;

}

void Face::rescaledRectangle (Mat& img, const Rect& r, double scale, const Scalar& color, int thickness, int lineType, int shift)
{
    Point p1,p2;

    p1 = r.tl();
    p2 = r.br();
    p1.x *= scale;
    p1.y *= scale;
    p2.x *= scale;
    p2.y *= scale;

    rectangle(img, p1, p2, color,thickness, lineType, shift);
}

void Face::rescaledLine (Mat& img, Point p1, Point p2, double scale, const Scalar& color, int thickness, int lineType, int shift)
{
    p1.x *= scale;
    p1.y *= scale;
    p2.x *= scale;
    p2.y *= scale;

    line (img, p1, p2, color,thickness, lineType, shift);
}

void Face::rescaledEllipse(Mat& img, Point center, Size& axes, double angle, double startAngle, double endAngle, double scale, Scalar color, int thickness, int linetype)
{
    center.x *= scale;
    center.y *= scale;
    axes.width *= scale;
    axes.height*= scale;

    ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness, linetype);
}

void Face::rescaledCircle (Mat& img, const Point& p, double radius, double scale, const Scalar& color, int thickness, int lineType, int shift)
{
    Point p1 = p;

    p1.x *= scale;
    p1.y *= scale;

    cv::circle(img, p1, radius,color,thickness, lineType, shift);
}





















