//
//  main.cpp
//  beat-seg
//
//  Created by Piaras Hoban on 30/09/2014.
//  Copyright (c) 2014 Piaras Hoban. All rights reserved.
//

#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "essentia/essentia.h"
#include "essentia/essentiamath.h"
#include "essentia/algorithm.h"
#include "essentia/algorithmfactory.h"
#include "essentia/streaming/algorithms/poolstorage.h"
#include "essentia/scheduler/network.h"

#include "mlpack/core.hpp"
#include "mlpack/methods/neighbor_search/neighbor_search.hpp"
#include "armadillo"

#include "beat-seg.h"

using namespace std;
using namespace essentia;
using namespace essentia::standard;
using namespace arma;
using namespace mlpack::neighbor;

//Methods
mat runAnalysis(string audiopath);
void medianFilter(mat& M, int k = 8);
//void gaussianFilter();
//void gaussianKernel();
mat recurrence_matrix(mat& M);
mat downsample(mat& X,int v);
void noveltyCurve();
void pickPeaks();
void circularShift();
void embed(mat& M);
void segment();

void normalizeMatrix(mat& m);
void write_matrix(mat& m);

int frameSize = 4096;
int hopSize = 2048;

//Analysis Output Storage
std::vector<Real> audioBuffer,startTimes,endTimes,peakM,peakF,frame,spectrum,chroma;
std::vector<Real> onsets = {0};
std::vector<std::vector<Real>> slicedFrames;
Real rate,dur;

int main(int argc, const char * argv[]) {
	
	string audioFilename = argv[1];

	mat R = runAnalysis(audioFilename);
	
//	R = downsample(R, 2);

	cout << R.size() << endl;

//	embed(R);
	
	mat RP = recurrence_matrix(R);
	
	cout << "Writing" << endl;
	
	write_matrix(RP);
	
	
	return 0;
}

mat runAnalysis(string audiopath)
{
	
	
	essentia::init();
	
	//Algorithms
	AlgorithmFactory& factory = standard::AlgorithmFactory::instance();
	Algorithm* audio = factory.create("EasyLoader","filename",audiopath);
	Algorithm* duration = factory.create("Duration");
	Algorithm* onsetDetection = factory.create("OnsetRate");
	Algorithm* framecutter = factory.create("FrameCutter","frameSize",frameSize,"hopSize",hopSize);
	Algorithm* fft = factory.create("Spectrum","size",frameSize);
	Algorithm* peaks = factory.create("SpectralPeaks");
	Algorithm* hpcp = factory.create("HPCP");
	
	//Processes
	audio->output("audio").set(audioBuffer);
	
	duration->input("signal").set(audioBuffer);
	duration->output("duration").set(dur);
	
	framecutter->input("signal").set(audioBuffer);
	framecutter->output("frame").set(frame);
	
	fft->input("frame").set(frame);
	fft->output("spectrum").set(spectrum);
	
	peaks->input("spectrum").set(frame);
	peaks->output("frequencies").set(peakF);
	peaks->output("magnitudes").set(peakM);
	
	hpcp->input("frequencies").set(peakF);
	hpcp->input("magnitudes").set(peakM);
	hpcp->output("hpcp").set(chroma);
	
	onsetDetection->input("signal").set(audioBuffer);
	onsetDetection->output("onsets").set(onsets);
	onsetDetection->output("onsetRate").set(rate);
	
	audio->compute();
	duration->compute();
	onsetDetection->compute();
	onsets.push_back(dur);
	
	int c = 0;
	
	mat M;
	
	while (true) {
		framecutter->compute();
		
		if (!frame.size()) {
			break;
		}
		
		fft->compute();
		peaks->compute();
		hpcp->compute();
		M.insert_cols(c,conv_to<colvec>::from(chroma));
		c++;
		
	}
	
	//CleanUp
	delete audio;
	delete duration;
	delete onsetDetection;
	delete framecutter;
	delete fft;
	delete peaks;
	delete hpcp;

	essentia::shutdown();
	
	return M;
}

void write_matrix(mat& m)
{
	ofstream myfile("chroma.txt");
	
	myfile << m;
	
	myfile.close();

	system("gnuplot plotchroma.gnuplot");
}

void medianFilter(mat& M, int k)
{
	cout << M.n_rows << endl;
	for (int i = 0 ; i < M.n_rows; ++i) {
		for (int j = 0 ; j < M.n_cols - k/2; ++j) {
			if (j < k/2) {
				M.at(i,j) = median(M.cols(j,j+k/2)(i));
			} else {
				M.at(i,j) = median(M.cols(j-k/2,j+k/2)(i));
			}
	 	}
	}
}

void normalizeMatrix(mat& M)
{
	double min = M.min();
	double max = M.max();
	
	M = M + min;
	M = M / max;
}

int col_includes(const Col<size_t> x,const int j) {
	int val = 0;
	
	for (int i = 0; i < x.size(); ++i) {
		if (x(i) == j) {
			val = 1;
			break;
		}
	}
	
	return val;
}

mat recurrence_matrix(mat& M)
{
	int n = M.n_cols;
	int K = n * 0.001;
	mat RP(n,n);

	for  (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			RP(i,j) = sqrt(sum((M.col(j) - M.col(i))));
		}
	}
	
//	RP = cor(M,M.t());
//	AllkNN a(M);
//	Mat<size_t> resultingNeighbors;
//	mat resultingDistances;
//
//	a.Search(K,resultingNeighbors,resultingDistances);
//	int n2 = resultingNeighbors.n_cols;
//
//	for (int i = 0; i < n; ++i) {
//		for (int j = 0; j < n; ++j) {
//			RP(i,j) =	1 - (col_includes(resultingNeighbors.col(i),j)
//						&&
//						col_includes(resultingNeighbors.col(j),i));
//		}
//		cout << "-------> " << (i/(float)n2)*100 << "%...\r";
//		cout.flush();
//	}
	return RP;
}

void embed(mat& M)
{
	int t = 1;
	int m = 10;
	int n = (int)M.n_cols - (m-1);
	
	mat H;
	
	for (int i = 0; i < n;++i) {
		colvec x = M.col(i);
		for (int j = 0; j < m;++j) {
			x = join_cols(x,M.col(i+j));
		}
		H.insert_cols(i,x);
	}
	M = H;
}

mat downsample(mat& X,int v)
{
	int n = X.n_cols;
	mat H;
	for (int i=0; i < n/v;++i) {
		Mat<double> Y = X.cols(i*v,((i+1)*v)-1);
		colvec sumy = sum(Y,1);
		H.insert_cols(i,sumy);
	}
	return H;
}

