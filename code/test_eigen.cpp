#include "eigen_serialize.hpp"
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "Eigen/Sparse"
#include "Eigen/SparseCore"
#include "Eigen/Dense"
#include "Eigen/Eigen"

#include <map>
#include <iostream>
#include <fstream>
#include <stdint.h>
#include <vector>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/math/special_functions/sign.hpp> //just for sign..... jesus C++ sometimes

using namespace std; //because we can
using namespace Eigen; //fuck yeah pollute it up

typedef Eigen::Matrix<double, Dynamic, Dynamic, RowMajor> MatrixD;
typedef Eigen::SparseMatrix<int8_t, RowMajor> SparseMatI;
typedef Eigen::SparseMatrix<double, RowMajor> SparseMatD;

//global num of features for collaborative filter
const int NUMFEATURES = 10;
int numFeatures = NUMFEATURES;
//global learning rate for gradient descent
//const double GAMMA = 0.010; <--pre sudent-kc biases
const double GAMMA = 0.010;
double _gamma = GAMMA;
//const double GAMMALEARN = 0.05; <-- pre student-kc biases
const double GAMMALEARN = 0.05;
double _gammalearn = GAMMALEARN;
//global regularization constant
//const double LAMBDA = 0.15; <-- this was good before adding student bias
const double LAMBDA = 0.15;
double lambda = LAMBDA;
const double OVERALL_MEAN = 0.261519;
//const double STUDENT_LAMBDA_MULTIPLIER = 500;

const int THREADCOUNT = 4;

ofstream trainlogfile;
ofstream lambdalogfile;
ofstream featurelogfile;
ofstream gammalearnlogfile;

bool logtrain = false;
bool loglambda = false;
bool logfeats = false;
bool loggamma = false;

//global mapping of student ids to row in matrix
map<int, int> studentMap;
SparseMatI train;
SparseMatD trainKC;
MatrixD    trainKC_PCA;
SparseMatD trainSubtracks;

MatrixD U, Q;
//MatrixD KCBias;
MatrixD QuestionBias;

struct perThread
{
	MatrixD KMat;
	double overall_bias = OVERALL_MEAN;
};

perThread threadVars[THREADCOUNT];
perThread avgVars;

void aggregateVars()
{
	avgVars = threadVars[0];
	if(THREADCOUNT > 1)
	{
		for(int c = 1; c < THREADCOUNT; c++)
		{
			avgVars.KMat += threadVars[c].KMat;
			avgVars.overall_bias += threadVars[c].overall_bias;
		}
		avgVars.KMat /= THREADCOUNT;
		avgVars.overall_bias /= THREADCOUNT;
		for(int c = 0; c < THREADCOUNT; c++)
		{
			threadVars[c] = avgVars;
		}
	}
}

void getMap(string filename)
{
	ifstream results;
	results.open(filename.c_str());
	string line;
	std::getline(results, line);
	string token;
	istringstream ss(line);
	std::getline(ss, token, ',');
	int numUsers = atoi(token.c_str());
	
	for(int i = 0; i < numUsers; i++)
	{
		std::getline(results,line);
		//assume good line
		istringstream ss(line);
		
		std::getline(ss, token, ',');
		//token is userid here
		int user = atoi(token.c_str());
		//map [userid] = rowid
		studentMap[user] = i;
	}
}

//read the question feature data into trainKC
void read_question_features(string filename)
{
	ifstream results;
	results.open(filename.c_str());
	
	string line;
	
	std::getline(results,line);
	string token;
	istringstream ss(line);
	std::getline(ss, token, ',');
	int numQuestions = atoi(token.c_str());
	std::getline(ss, token, ',');
	int numFeatures = atoi(token.c_str());
	
	trainKC.resize(numQuestions, numFeatures);
	
	for(int i = 0; i < numQuestions; i++)
	{
		uint32_t c = 0;
		std::getline(results,line);
		//assume good line
		istringstream ss(line);
		
		std::getline(ss, token, ',');
		//token is question id here
		int question = atoi(token.c_str());
		
		if( question != i )
		{
			//we know at least one question is missing
			//just leave empty feature vector
			cout << i << " Missing!" << endl;
			i = question;
			cout << "Setting i to: " << i << endl;
		}
		
		while(std::getline(ss, token, ','))
		{
			if(token.length() == 0)
				c++;
			else
			{
				int r = atoi(token.c_str());
				if(r != 0)
				{
					trainKC.insert(i,c) = r;
				}
				c++;
			}
		}
		
		if(i % 1000 == 0)
			cout << "question: " << i << endl;
	}
}

//reads in the question features post-PCA
void read_PCA_question_features(string filename)
{
	ifstream results;
	results.open(filename.c_str());
	
	string line;
	
	std::getline(results,line);
	string token;
	istringstream ss(line);
	std::getline(ss, token, ',');
	int numQuestions = atoi(token.c_str());
	std::getline(ss, token, ',');
	int numFeatures = atoi(token.c_str());
	
	trainKC_PCA.resize(numQuestions, numFeatures);
	
	for(int i = 0; i < numQuestions; i++)
	{
		uint32_t c = 0;
		std::getline(results,line);
		//assume good line
		istringstream ss(line);
		
		if( i == 2207 )
		{
			//question 2207 is missing
			i++;
		}
		
		while(std::getline(ss, token, ','))
		{
			if(token.length() == 0)
				c++;
			else
			{
				int r = atof(token.c_str());
				if(r != 0)
				{
					trainKC_PCA(i,c) = r;
				}
				c++;
			}
		}
		
		if(i % 1000 == 0)
			cout << "question: " << i << endl;
	}
}

//read training data into train
void read_results_file(string filename)
{	
	ifstream results;
	results.open(filename.c_str());
	
	string line;
	
	std::getline(results,line);
	string token;
	istringstream ss(line);
	std::getline(ss, token, ',');
	int numUsers = atoi(token.c_str());
	std::getline(ss, token, ',');
	int numQuestions = atoi(token.c_str());
	
	//cout << numQuestions << " " << numUsers;
	train.resize(numUsers, numQuestions);
	
	for(int i = 0; i < numUsers; i++)
	{
		uint32_t c = 0;
		std::getline(results,line);
		//assume good line
		istringstream ss(line);
		
		std::getline(ss, token, ',');
		//token is userid here
		int user = atoi(token.c_str());
		//map [userid] = rowid
		studentMap[user] = i;
		
		while(std::getline(ss, token, ','))
		{
			if(token.length() == 0)
				c++;
			else
			{
				int r = atoi(token.c_str());
				if(r != 0)
				{
					train.insert(i,c) = r;
				}
				c++;
			}
		}
		
		if(i % 1000 == 0)
			cout << "user: " << i << endl;
	}
}

void SGD(float t_gamma, size_t startU, size_t numU, size_t startQ, size_t numQ, double &squaredErr, int ID)
{	
	size_t endQ = min(startQ + numQ, size_t(train.cols()));
	size_t endU = min(startU + numU, size_t(train.rows()));
	
	size_t question;
	int8_t answer;
	
	squaredErr = 0.f;
	
	MatrixD qKCBias;
	MatrixD ksbias;

	//Run SGD on the data in one stratum of the training matrix
	for( size_t user = startU; user < endU; ++user )
	{
		//train on the examples from a single student
		for( SparseMatI::InnerIterator it(train,user); it; ++it )
		{
			//cout << "gg" << endl;
			question = it.index();
			
			//only train on the questions in the stratum
			if(question < startQ || question >= endQ)
				continue;
			
			answer = it.value(); 
			
			/*double gg = (trainKC.row(question) * threadVars[ID].KMat * U.row(user).transpose())(0,0) / trainKC.row(question).sum();
			
			cout << gg << endl;
			if(std::isnan(gg))
			{
				cout << gg << endl << endl;
				cout << trainKC.row(question) * threadVars[ID].KMat << endl << endl;
				cout << U.row(user) << endl << endl;
				cout << Q.row(question) << endl << endl;
				cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
			}*/
			
			//cout << trainKC.row(question) * KMat << endl << endl;
			//cout << trainKC.row(question) * KMat * U.row(user).transpose() << endl;
			
			double est = (U.row(user) * Q.row(question).transpose())(0,0) + QuestionBias(0, question) + threadVars[ID].overall_bias + (trainKC.row(question) * threadVars[ID].KMat * U.row(user).transpose())(0,0) / trainKC.row(question).sum();
			double err = est - answer;
			
			//cout << est << endl << endl;
			
			squaredErr += err*err;
			
			U.row(user)     -= t_gamma * (err * Q.row(question) + lambda * U.row(user));
			Q.row(question) -= t_gamma * (err * U.row(user) + lambda * Q.row(question));
			QuestionBias(0, question) -= t_gamma * (err + lambda * QuestionBias(0,question));
			threadVars[ID].overall_bias -= t_gamma * (err + lambda * threadVars[ID].overall_bias);
			
			MatrixD left = MatrixD((trainKC.row(question).transpose() * U.row(user))) * err / trainKC.row(question).sum();
			MatrixD right = (threadVars[ID].KMat) * lambda;
			threadVars[ID].KMat -= left + right;
			
			/*
			cout << threadVars[ID].KMat.maxCoeff() << " " << threadVars[ID].KMat.minCoeff();
			
			std::string s;
			std::getline(std::cin, s);
			*/
		}
	}
	//cout << squaredErr << endl;
	//cout << "DONE\n" << endl;
}

void training_harness()//, const blaze::CompressedMatrix<int8_t> &actual)
{
	U.resize(train.rows(),numFeatures);
	Q.resize(train.cols(),numFeatures);
	//1 row by Q columns for QuestionBias
	QuestionBias.resize(1, Q.rows());
	//KCBias.resize(1, trainKC.cols());
	
	U = MatrixXd::Random(U.rows(), U.cols());
	Q = MatrixXd::Random(Q.rows(), Q.cols());
	QuestionBias = MatrixXd::Random(QuestionBias.rows(), QuestionBias.cols());
	//KCBias = MatrixXd::Random(KCBias.rows(), KCBias.cols());
	
	for(int c = 0; c < THREADCOUNT; c++)
	{
		threadVars[c].KMat.resize(trainKC.cols(), numFeatures);
		threadVars[c].KMat = MatrixXd::Random(threadVars[c].KMat.rows(), threadVars[c].KMat.cols());
	}
	
	
	double squaredErr = 0;
	
	size_t user_total = train.rows();
	size_t question_total = train.cols();
	
	size_t numU = ceil(double(user_total) / THREADCOUNT);
	size_t numQ = ceil(double(question_total) / THREADCOUNT);
	
	map<size_t, size_t> q_strat;
	map<size_t, size_t> u_strat;
	
	//iterations through the data
	for(int c = 0; c < 15 * THREADCOUNT; c++)
	{
		double t_gamma = _gamma / (1 + _gamma * c * _gammalearn );
		
		double squaredErrVec[THREADCOUNT];
		squaredErr = 0;
		boost::thread_group sgdThreads;
		
		q_strat.clear();
		u_strat.clear();
		
		for(int i = 0; i < THREADCOUNT; i++)
		{
			size_t u, q;
			while(u_strat.count(u = rand() % THREADCOUNT))
			{}
			while(q_strat.count(q = rand() % THREADCOUNT))
			{}
			u_strat[u] = 1;
			q_strat[q] = 1;
			
			sgdThreads.add_thread(new boost::thread(SGD, t_gamma, numU * u, numU, numQ * q, numQ, boost::ref(squaredErrVec[i]), i));
		}
		
		sgdThreads.join_all();
		aggregateVars();
		
		for(int i = 0; i < THREADCOUNT; i++)
		{
			squaredErr += squaredErrVec[i];
		}
		//cout << lastErr - squaredErr << endl;
		//lastErr = squaredErr;
		if(logtrain)
			trainlogfile << c << "," << squaredErr << ",\n";
			
		//if(c % 25 == 0)
		{
			cout << "Iteration: " << c << endl;
			cout << squaredErr << " " << t_gamma << endl;
		}
	}

	
	//do the testing here too....
	cout << "testing\n";

	ifstream file;
	file.open("../data/validation_test_data.csv");
	string line;
	string token;
	int result;
	int user;
	int question;
	
	int correct = 0;
	int total = 0;
	
	double estimation;
	
	double SE = 0;
	
	while(file.good())
	{
		getline(file,line);
		istringstream ss(line);
		getline(ss, token, ',');
		result = atoi(token.c_str());
		getline(ss, token, ',');
		user = atoi(token.c_str());
		user = studentMap[user];
		getline(ss, token, ',');
		question = atoi(token.c_str());
		
		//cout << result << " " << user << " " << question << endl;

		//estimation = (U.row(user) * Q.row(question).transpose())(0,0) + QuestionBias(0, question) + avgVars.overall_bias;
		
		estimation = (U.row(user) * Q.row(question).transpose())(0,0) + QuestionBias(0, question) + avgVars.overall_bias + (trainKC.row(question) * avgVars.KMat * U.row(user).transpose())(0,0) / trainKC.row(question).sum();

		if(result == 0)
			result -= 1;
		
		SE += (estimation - result) * (estimation - result);
		
		if(boost::math::sign(estimation) - result == 0)
		{
			//cout << "correct!\n";
			correct ++;
		}
		
		total++;
	}
	SE /= total;
	SE = sqrt(SE);
	
	if(loglambda)
		lambdalogfile << correct / ( (double) total) << "," << SE << endl;
		
	if(loggamma)
		gammalearnlogfile << correct / ( (double) total) << "," << SE << endl;
		
	if(logfeats)
		featurelogfile << correct / ( (double) total) << "," << SE << endl;

	cout << "Final accuracy: " << correct / ( (double) total);
	cout << endl << correct << " out of " << total << endl;
	
	//cout << QuestionBias << endl;
	//int wut;
	//cin >> wut;
} 

int main(int argc, char**argv)
{	
	if (FILE *file = fopen("../data/realmats_t.eigen", "r")) {
        fclose(file);
		try{
			ifstream gg( "../data/realmats_t.eigen" );
			boost::archive::text_iarchive ar( gg );
			getMap( "../data/validation_student_responses.csv" );
			ar >> train;
		}catch(boost::archive::archive_exception e)
		{
			cout << "Archive Exception: " << e.what() << endl;
			return 0;
		}
    }
	else
	{
		read_results_file("../data/validation_student_responses.csv");
		ofstream gg( "../data/realmats_t.eigen" );
		boost::archive::text_oarchive ar( gg );
		ar << train;
	}
	
	if( FILE *file = fopen("../data/realKCmats_t.eigen", "r")) {
		fclose(file);
		try{
			ifstream gg( "../data/realKCmats_t.eigen" );
			boost::archive::text_iarchive ar( gg );
			ar >> trainKC;
		}catch(boost::archive::archive_exception e)
		{
			cout << "Archive Exception: " << e.what() << endl;
			return 0;
		}
    }
	else
	{
		read_question_features("../data/validation_question_features.csv");
		ofstream gg( "../data/realKCmats_t.eigen" );
		boost::archive::text_oarchive ar( gg );
		ar << trainKC;
	}
	
	if( FILE *file = fopen("../data/realKCmatsPCA_t.eigen", "r")) {
		fclose(file);
		try{
			ifstream gg( "../data/realKCmatsPCA_t.eigen" );
			boost::archive::text_iarchive ar( gg );
			ar >> trainKC_PCA;
		}catch(boost::archive::archive_exception e)
		{
			cout << "Archive Exception: " << e.what() << endl;
			return 0;
		}
    }
	else
	{
		read_question_features("../data/question_feature30.csv");
		ofstream gg( "../data/realKCmatsPCA_t.eigen" );
		boost::archive::text_oarchive ar( gg );
		ar << trainKC_PCA;
	}
	
	trainSubtracks = trainKC.block(0, 14, trainKC.rows(), 16);
	//for now just try PCA
	//trainKC = trainKC_PCA;
	
	cout << "DATA LOADED\n";
	
	srand(time(NULL));
	
	logtrain = true;
	//loglambda = true;
	//loggamma = true;
	//logfeats = true;
	
	if(logtrain)
		trainlogfile.open("../data/real_train_log.csv");
	if(loggamma || logfeats || loglambda)
	{
		if(loggamma)
			gammalearnlogfile.open("../data/real_gammalearn_log.csv");
		if(logfeats)
			featurelogfile.open("../data/real_feature_log.csv");
		if(loglambda)
			lambdalogfile.open("../data/real_lambda_log.csv");
	}
	
	if(logtrain)
	{
		time_t then = time(NULL);
		trainlogfile << "Rounds through data, Total squared training error (Features = " << numFeatures << " Gamma_0 = " << _gamma << " Gamma' = " << _gammalearn << " Lambda = " << lambda << " )\n";
		training_harness();
		cout << "\n\nTime: " << difftime(time(NULL), then) << " seconds.\n";
	}
	if(loglambda)
	{
		lambdalogfile << "Lambda, Test accuracy, Test RMSE   (Features = " << numFeatures << " Gamma_0 = " << _gamma << " Gamma' = " << _gammalearn << " )\n";
		for(lambda = 0.01; lambda < 0.31; lambda += 0.01)
		{
			lambdalogfile << lambda << ",";
			training_harness();
		}
	}
	lambda = LAMBDA;
	if(logfeats)
	{
		featurelogfile << "Features, Test accuracy, Test RMSE   (Lambda = " << lambda << " Gamma_0 = " << _gamma << " Gamma' = " << _gammalearn << " )\n";
		for(numFeatures = 10; numFeatures < 50; numFeatures+=5)
		{
			featurelogfile << numFeatures << ",";
			training_harness();
		}
	}
	numFeatures = NUMFEATURES;
	if(loggamma)
	{
		gammalearnlogfile << "Gamma', Test accuracy, Test RMSE   (Features = " << numFeatures << " Gamma_0 = " << _gamma << " Lambda = " << lambda << " )\n";
		for(_gammalearn = 0.05; _gammalearn < 0.35; _gammalearn += 0.05)
		{
			gammalearnlogfile << _gammalearn << ",";
			training_harness();
		}
	}
	
	
	//dump biases to file
	{
		ofstream gg( "../data/question_bias_t.eigen" );
		boost::archive::text_oarchive ar( gg );
		ar << QuestionBias;
	}
	
	
	return 0;
}

