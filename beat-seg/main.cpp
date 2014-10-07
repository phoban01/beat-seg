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

//functions
mat runAnalysis(string audiopath);
void medianFilter(mat& M, int k = 8);
mat recurrence_matrix(mat& M,int metric = 1);
mat downsample(mat& X,int v);
mat beatsync(mat& X,std::vector<Real> p);
mat ckernel(int n);
void embed(mat& M,int m = 10);

void noveltyCurve();
void pickPeaks();
void circularShift();
void segment();

void normalizeMatrix(mat& m);
void write_matrix(mat& m);

const int SAMPLERATE = 44100;
const int frameSize = 4096;
const int hopSize = 4096;
const double FRAMEDUR = SAMPLERATE/(double)frameSize;

//Analysis Output Storage
std::vector<Real> audioBuffer,startTimes,endTimes,peakM,peakF,frame,wframe,spectrum,chroma;
std::vector<Real> onsets = {0};
std::vector<std::vector<Real>> slicedFrames;
Real rate,dur,confidence;


int main(int argc, const char * argv[]) {

	string audioFilename = argv[1];

	mat R = runAnalysis(audioFilename);
	
	R = beatsync(R,onsets);
	
//	R = downsample(R, 1);
	
	embed(R,10);
	
	mat RP = recurrence_matrix(R,atoi(argv[2]));
	
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
	Algorithm* window = factory.create("Windowing","size",frameSize,"type","blackmanharris62");
	Algorithm* duration = factory.create("Duration");
	Algorithm* onsetDetection = factory.create("BeatTrackerDegara","maxTempo",160);
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
	
	window->input("frame").set(frame);
	window->output("frame").set(wframe);
	
	fft->input("frame").set(wframe);
	fft->output("spectrum").set(spectrum);
	
	peaks->input("spectrum").set(spectrum);
	peaks->output("frequencies").set(peakF);
	peaks->output("magnitudes").set(peakM);
	
	hpcp->input("frequencies").set(peakF);
	hpcp->input("magnitudes").set(peakM);
	hpcp->output("hpcp").set(chroma);
	
	onsetDetection->input("signal").set(audioBuffer);
	onsetDetection->output("ticks").set(onsets);
//	onsetDetection->output("confidence").set(confidence);
	
	audio->compute();
	duration->compute();
	onsetDetection->compute();

	int counter = 0;
	
	mat M;
	
	while (true) {
		framecutter->compute();
		
		if (!frame.size()) {
			break;
		}

		window->compute();
		fft->compute();
		peaks->compute();
		hpcp->compute();
		M.insert_rows(counter,conv_to<rowvec>::from(chroma));
		counter++;
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
	
	for (int i = 0; i < x.size() - 1; ++i) {
		if (x(i) == j) {
			val = 1;
			break;
		}
	}
	
	return val;
}

mat recurrence_matrix(mat& M,int metric)
{
	int n = M.n_rows;
	mat RP(n,n);

	for (int i = 0; i < M.n_rows; ++i) {
		for (int j = 0; j < M.n_rows; ++j) {
			if (metric == 0)
				RP(i,j) = mlpack::metric::SquaredEuclideanDistance::Evaluate(M.row(i),M.row(j));
			else
				RP(i,j) = mlpack::kernel::CosineDistance::Evaluate(M.row(i),M.row(j));
		}
	}
	
//	AllkNN a(M);
//	Mat<size_t> resultingNeighbors;
//	mat resultingDistances;
//	int K = M.size() * 0.8;
//	a.Search(K,resultingNeighbors,resultingDistances);
//	int n2 = resultingNeighbors.n_cols;
//	RP.resize(n2,n2);
//
//	for (int i = 0; i < n2; ++i) {
//		for (int j = 0; j < n2; ++j) {
//			RP(i,j) =	1 - (col_includes(resultingNeighbors.col(i),j)
//						&&
//						col_includes(resultingNeighbors.col(j),i));
//		}
//	}
	return RP;
}

void embed(mat& M, int m)
{
	int t = 1;
	int n = (int)M.n_rows - (m-1);
	
	mat H;
	
	for (int i = 0; i < n;++i) {
		rowvec x = M.row(i);
		for (int j = 0; j < m;++j) {
			x = join_rows(x,M.row(i+j));
		}
		H.insert_rows(i,x);
	}
	M = H;
}

mat downsample(mat& X,int v)
{
	int n = X.n_rows;
	mat H;
	for (int i=0; i < n/v;++i) {
		Mat<double> Y = X.rows(i*v,((i+1)*v)-1);
		rowvec sumy = mean(Y);
		H.insert_rows(i,sumy);
	}
	return H;
}

mat beatsync(mat& X,std::vector<Real> p)
{
//	int n = X.n_rows;
	mat H;
	int x,y;
	p.push_back((X.n_rows-1) / FRAMEDUR);

	for (int i=1; i < p.size();++i) {
		x = p[i-1] * FRAMEDUR;
		y = p[i] * FRAMEDUR;
//		cout << x << " " << y << endl;
		Mat<double> Y = X.rows(x,y);
		rowvec sumy = mean(Y);
		H.insert_rows(i-1,sumy);
	}
	
	return H;
}

mat ckernel(int n)
{
	double sigma = 1;
	double r, s = 2.0 * sigma * sigma;
	double sum = 0.0;
	
	mat kernel(n,n);
	
	int halfn = n / 2;
	
	for (int i = halfn * -1;i < halfn; ++i) {
		for (int j = halfn * -1;j < halfn; ++j) {
			r = sqrt(i*i + j*j);
			kernel(i+halfn,j+halfn) = (exp(-(r*r)/s)) / (M_PI * s);
			sum += kernel(i+halfn,j+halfn);
		}
	}
	
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < kernel.n_rows; ++j) {
			if (((i < halfn) && (j < halfn)) || ((i >= halfn) && (j >= halfn))) {
				kernel(i,j) *= 1;
			} else {
				kernel(i,j) *= -1;
			}
		}
	}
	
	
	return kernel;
}