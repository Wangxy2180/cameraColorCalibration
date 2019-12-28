// myCameraCalibrate.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>
#include "kinect.h"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;
//这个假设已经有图像了
int main()
{
	/*
	std::string colorPath = "C:\\Users\\王\\Desktop\\KinectSDK\\pictureHere\\color";
	std::string infraredPath = "C:\\Users\\王\\Desktop\\KinectSDK\\pictureHere\\infrared";
	std::string index = "1.bmp";
	colorPath = colorPath + index;
	infraredPath = infraredPath + index;
	*/
	vector<string> filePath;
	for (int count = 0; count < 16; count++) {

		string infraredPath = "C:\\Users\\王\\Desktop\\KinectSDK\\KinectColor\\color";
		string index = to_string(count) + ".bmp";
		infraredPath = infraredPath + index;
		cout << infraredPath << endl;
		filePath.push_back(infraredPath);
	}
	cout << filePath.size() << endl;
	cout << filePath.data()[1] << endl;
	int iIWidth = 1920;
	int iIHeight = 1080;

////////////////////////////////////////////////////////////////////////////////////////////
	const double patternLen = 30;//每个格子的大小 30mm
	Size patternSize = Size(7, 5);//拍的图
	//Size patternSize = Size(6, 4);//这是一个7*5的方块
	//顺序无关，哪个在前都可以
	//Size patternSize = Size(6, 9);//这是一个5*7的方块
////////////////////////////////////////////////////////////////////////////////////////////

	Size infraredImgSize(iIWidth, iIHeight);
	int imageCount = 0;

	vector<Point2f> outColorCorners;//缓存检测到的角点
	vector<vector<Point2f>> outInfraredCorners_seq;//所有检测到的缓存序列
	for (int i = 0; i <filePath.size(); i++) {
		imageCount++;
		Mat imageInput = imread(filePath[i]);
		
		//未转化灰度版//这里的代码是正确示例
		bool patternColorFound = findChessboardCorners(imageInput, patternSize, outColorCorners);
		if (patternColorFound == false)
		{
			cout << "彩色图获取角点失败" << endl;
			//To do
		}
		cout << "彩色图" << i << "检测出" << outColorCorners.size() << "个角点" << endl;

		string drawName = "drawColor" + to_string(i);
		//
		
		Mat image2Gray;
		cvtColor(imageInput, image2Gray, COLOR_RGB2GRAY);
		cornerSubPix(image2Gray, outColorCorners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 20, 0.01));
		drawChessboardCorners(image2Gray, patternSize, outColorCorners, true);
		
		outInfraredCorners_seq.push_back(outColorCorners);
		drawChessboardCorners(image2Gray, patternSize, outColorCorners, true);
		imshow(drawName, image2Gray);
		
		//waitKey(500); 
	}
	waitKey(0);

	vector<vector<Point3f>> image3DCorners_seq;
	for (int i = 0; i < imageCount; i++) {
		vector<Point3f> image3DCorners;
		for (int i = 0; i < patternSize.height; i++)
		{
			for (int j = 0; j < patternSize.width; j++)
			{
				Point3f realCorner;
				/* 假设标定板放在世界坐标系中z=0的平面上 */
				realCorner.x = i * patternLen;
				realCorner.y = j * patternLen;
				realCorner.z = 0;
				image3DCorners.push_back(realCorner);
			}
		}
		image3DCorners_seq.push_back(image3DCorners);
	}
	//相机内参矩阵和畸变，别问，这么写就行
	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
	Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));
	vector<Mat> rvecsMat;
	vector<Mat> tvecsMat;

	Size imageSize = Size(512, 424);
	double retErr=calibrateCamera(image3DCorners_seq, outInfraredCorners_seq, imageSize, cameraMatrix,distCoeffs,rvecsMat,tvecsMat);
	cout << "标定函数结束" << endl;
	cout << "retErr is " << retErr << endl;
	//参数列表
	//1.objectPoint:世界坐标点 vector<vector<Point3f>>
	//2.imagePoint vector< vector< Point2f > >
	//3.imageSize Size()
	//4.cameraMatrix 内参数矩阵 Mat cameraMatrix;
	//5.distCoeffs 畸变矩阵 Mat camreaMatrix
	//6.rvecs 旋转向量
	//7.tvecs 位移向量
	//8.stdDeviationsIntrinsics:内参的输出向量。顺序为(fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4,τx,τy)，如果不估计其中某一个参数，值等于0
	//9.stdDeviationsIntrinsics:外参
	//10.perViewErrors 每个标定图片的重投影均方根误差的输出向量
	//11.int flag=0 标定时所采用的模型 ！！！！！重点
	//12.criteria： 迭代优化算法的终止准则

	cout << "开始评价标定结果………………";
	double total_err = 0.0;            // 所有图像的平均误差的总和 
	double err = 0.0;                  // 每幅图像的平均误差
	double totalErr = 0.0;
	double totalPoints = 0.0;
	vector<Point2f> image_points_pro;     // 保存重新计算得到的投影点

	for (int i = 0; i < imageCount; i++)
	{
		projectPoints(image3DCorners_seq[i], rvecsMat[i], tvecsMat[i], cameraMatrix, distCoeffs, image_points_pro);   //通过得到的摄像机内外参数，对角点的空间三维坐标进行重新投影计算

		err = norm(Mat(outInfraredCorners_seq[i]), Mat(image_points_pro), NORM_L2);

		totalErr += err * err;
		totalPoints += image3DCorners_seq[i].size();

		err /= image3DCorners_seq[i].size();
		//fout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
		total_err += err;
	}
	cout << "重投影误差2：" << sqrt(totalErr / totalPoints) << "像素" << endl << endl;
	cout << "重投影误差3：" << total_err / imageCount << "像素" << endl << endl;


	//保存定标结果    
	cout << "开始保存定标结果………………" << endl;
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */
	cout << "相机内参数矩阵：" << endl;
	cout << cameraMatrix << endl << endl;
	cout << "畸变系数：\n";
	cout << distCoeffs << endl << endl << endl;
	for (int i = 0; i < imageCount; i++)
	{
		cout << "第" << i + 1 << "幅图像的旋转向量：" << endl;
		cout << rvecsMat[i] << endl;

		/* 将旋转向量转换为相对应的旋转矩阵 */
		Rodrigues(rvecsMat[i], rotation_matrix);
		cout << "第" << i + 1 << "幅图像的旋转矩阵：" << endl;
		cout << rotation_matrix << endl;
		cout << "第" << i + 1 << "幅图像的平移向量：" << endl;
		cout << tvecsMat[i] << endl << endl;
	}
	cout << "定标结果完成保存！！！" << endl;
	


	std::cout << "Hello World!\n";
}


/*
	//cornerSubPix(mInfraredImage, outInfraredCorners, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 20, 0.01));
	//别忘了还有个infrared图
	//1.单通道图像
	//2.存放的角点
	//3.画出角点的圈的半径 这个表示直径23的圆
	//4.零区域，此处表示没有这样的区域
	//5.条件阈值，一旦满足某一条，立刻停止 ，此处表示：终止条为件COUNT+EPS，次数或精度，最多迭代20次，精度0.01
*/