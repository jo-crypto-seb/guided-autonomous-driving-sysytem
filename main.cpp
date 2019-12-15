//
//  main.cpp
//  opencv_project
//

//

#include "knuros.h"

using namespace std;
using namespace cv;

MGSign gSign;

double angle_;
double distance_ = 10; // detect range
bool happy_flag = false;

boost::mutex mtx[2];
nav_msgs::Odometry g_odom;
float pre_dAngleTurned;
sensor_msgs::LaserScan g_scan;

int main(int argc, char * argv[])
{
    distance_ = 10;
    gSign = NIL;
    ros::init(argc, argv, "happy");
    ros::NodeHandle nh;

    // setting for open_cv dectection
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber subRGB = it.subscribe("/raspicam_node/image", 1, &postMessageRecievedRGB, ros::VoidPtr(), image_transport::TransportHints("compressed"));
    
    
    // setting for parking
    ros::Publisher vel_pub_=nh.advertise<geometry_msgs::Twist>("/cmd_vel", 1);
    ros::Subscriber scan_sub = nh.subscribe("/scan", 10, scanCallback);
    
    
    // setting for autodriving
    ros::Subscriber subOdom = nh.subscribe("/odom", 50, &odomMsgCallback);
    ros::Subscriber subScan = nh.subscribe("/scan", 10, &scanMsgCallback);
    ros::Publisher pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 100);
    
    ros::Rate loop_rate(500);
    
    /* main code */
    
    // 2. if found PARKING_AREA, break
    // if(gSign == PARKING_AREA) return;
    
    // 3. auto race to find PARKING_SIGN
    autodriving(pub);
    
    // 4. if found PARKING_SIGN, break
    // if(gSign == PARKING_SIGN) return;
    
    // 5. parking function

    cout << "Let's Parking!" << endl;
    sleep(3);
    happy_flag = true;
    parking(vel_pub_);
   
    
    ros::spin();
    
    return 0;
}
