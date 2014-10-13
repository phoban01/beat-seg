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
mat recurrence_matrix(mat& M,float thresh = 0.1);
mat similarity_matrix(mat& M,int metric = 1);

mat downsample(mat& X,int v);
mat beatsync(mat& X,std::vector<Real> p);
mat ckernel(int n,double sigma);
std::vector<double> correlate(Mat<double> matrix,Mat<double> kernel);

void embed(mat& M,int m = 10);

mat circularShift(mat& M);
void gaussian_blur(mat& M,float size,float sigma);

void plotCorrelation(std::vector<double> v);
std::vector<double> noveltycurve(mat& M);
void pickPeaks();
void segment();

std::vector<double> peak_detection(std::vector<double>data,double thresh);
double auto_threshold(std::vector<double>data,double thresh);
double mean(std::vector<double>x);
void write_audacity_labels(std::vector<double> peaks,std::vector<Real> onset_times,int offset = 1);

void normalizeMatrix(mat& m);
void write_matrix(mat& m);
std::vector<double> normalize(std::vector<double> input);

rowvec slice(mat& M,int row_index,int start,int stop);

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

	int embedDimension = atoi(argv[2]);
	
    cout << "--> Running Analysis" << endl;
	mat R = runAnalysis(audioFilename);
	
	cout << "--> Beat-syncing" << endl;
	R = beatsync(R,onsets);
	
//	R = downsample(R,10);
	
	embed(R,embedDimension);
	
	cout << "--> Calculating Distance Matrix" << endl;
//	mat RP = similarity_matrix(R,1);
	mat RP = recurrence_matrix(R,atof(argv[3]));
	
	cout << "--> Circular Shifting" << endl;
	RP = circularShift(RP).t();

	gaussian_blur(RP,0.2,10);
	
	cout << "---> Writing Matrix" << endl;
	write_matrix(RP);
	
	std::vector<double> nc = normalize(noveltycurve(RP));

    plotCorrelation(nc);

//    cout << "---> Peak Detection" << endl;
//
//    std::vector<double> peaks = peak_detection(nc,3);
//
//	for (int i = 0; i < peaks.size();++i) {
//		peaks[i] += embedDimension / 2;
//	}
//	
//	cout << peaks << endl;
//	
//    write_audacity_labels(peaks, onsets,1);

	return 0;
}

int wrap(int value,int max)
{
	if (value < 0) {
		return max + value;
	} else if (value >= max) {
		return value % max;
	} else {
		return value;
	}
}

mat circularShift(mat& M)
{
	int n = M.n_rows;
	int k;
	mat L(n,n);
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			k = wrap(i+j,n)+1;
			L(i,j) = M(i,wrap(k,n));
		}
	}
	
	return L;
	
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

	system("gnuplot plotmatrix.gnuplot");
}

void medianFilter(mat& M, int k)
{
	int n = M.n_rows - 1;
	for (int i = 0 ; i < n; ++i) {
		for (int j = 0 ; j < n; ++j) {
			if (j+k <= n) {
				M(i,j) = median(slice(M,i,j,j+k));
			} else {
				M(i,j) = median(slice(M,i,j,n));
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

mat recurrence_matrix(mat& M,float thresh)
{
	AllkNN a(M.t());
	Mat<size_t> resultingNeighbors;
	mat resultingDistances;
	int K = M.n_rows * thresh;
	a.Search(K,resultingNeighbors,resultingDistances);
	int n = resultingNeighbors.n_cols;
	mat RP(n,n);
	
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			RP(i,j) =	1 - (col_includes(resultingNeighbors.col(i),j)
						&&
						col_includes(resultingNeighbors.col(j),i));
		}
	}
	return RP;
}

mat similarity_matrix(mat& M,int metric)
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
	return RP;
}

std::vector<double> noveltycurve(mat& M)
{
	std::vector<double> nc;
	int n = M.n_rows - 1;
	for (int i = 0; i < n; ++i) {
		nc.push_back(mlpack::metric::SquaredEuclideanDistance::Evaluate(M.row(i),M.row(i+1)));
	}
	
	return nc;
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

mat ckernel(int n,double sigma)
{
    mat checker_kernel(n,n);
    
    int t;
    
    double half_n = n / 2.0;
    // double sigma = 4.25;
    double r,s = (n * 0.5) * sigma * sigma;
    double sum = 0.0;
    
    //checkboard kernel with gaussian smoothing
    for (int i = 0;  i < n;  ++i)
    {
        double x = i - half_n;
        for (int j = 0;  j < n;  ++j)
        {
            double y = j - half_n;
            r = sqrt(x*x + y*y);
            if (i<half_n) {
                if (j<half_n) {
                    t=1;
                } else {
                    t=-1;
                }
            } else {
                if (j<half_n) {
                    t=-1;
                } else {
                    t=1;
                }
            }
            checker_kernel.at(i,j) = (exp(-(r*r)/s))/(M_PI * s) * t;
            sum += checker_kernel.at(i,j);
        }
    }
    
//    for (int i=0; i < n; ++i) {
//        for (int j=0; j < n; ++j) {
//            checker_kernel.at(i,j) /= sum;
//        }
//    }
	
//    ofstream kernelfile("kernel.txt");
//    
//    for (int i=0; i < n;++i) {
//        for (int j=0;j < n;++j) {
//            kernelfile << i << " " << j << " " << checker_kernel(i,j) << endl;
//        }
//        kernelfile << endl;
//    }
//    kernelfile.close();
    
    return checker_kernel;
}

std::vector<double> correlate(Mat<double> matrix,Mat<double> kernel)
{
    int n = kernel.n_rows - 1;
    std::vector<double> novelty;
    
    for (int i = n * -0.5 ; i != matrix.n_rows ; ++i) {
        double corr = 0.0;
        //zero padding
        if (i < 0) {
            for (int j=0 ; j < n  ; ++j) {
                for (int k=0 ; k < n ; ++k) {
                    if ((i+j >= 0) && (i+k >= 0)) {
                        corr += matrix(i+j,i+k) * kernel(j,k);
                    }
                }
            }
        }
        else if ((i > 0) && (i < matrix.n_rows - n)) {
            for (int j=0 ; j < n  ; ++j) {
                for (int k=0 ; k < n ; ++k) {
                    corr += matrix(i+j,i+k) * kernel(j,k);
                }
            }
        } else if (i >= matrix.n_rows - n) {
            for (int j=0 ; j < n  ; ++j) {
                for (int k=0 ; k < n ; ++k) {
                    if ((i+j < (matrix.n_rows-1)) &&
                        (i+k < (matrix.n_rows-1))) {
                        corr += matrix(i+j,i+k) * kernel(j,k);
                    }
                }
            }
        }
        novelty.push_back(corr);
    }
    //novelty curve
    return novelty;
}

void plotCorrelation(std::vector<double> v)
{
    ofstream novelty("novelty.txt");
    
    for (int i = 0 ; i < v.size();++i) {
        novelty << i << " " << v[i] << endl;
    }
    
    novelty.close();
    system("gnuplot 'novelty.gnu'");
}

std::vector<double> normalize(std::vector<double> input)
{
    std::vector<double> output;
    double max = *max_element(input.begin(),input.end());
    double min = *min_element(input.begin(),input.end());
    
    for (int i=1;i<input.size();++i) {
        output.push_back((input[i] - min) / (max-min));
    }
    return output;
}

double mean(std::vector<double>x)
{
    double sum = std::accumulate(x.begin(),x.end(),0.0);
    return sum / x.size();
}

double auto_threshold(std::vector<double>data,double thresh)
{
    double e = thresh;
    double c1 = data[0];
    double c2 = data[1];
    double lastc1 = c1;
    double lastc2 = c2;
    while (true) {
        std::vector<double> class1,class2;
        for (int i = 0; i < data.size(); ++i) {
            if (abs(c1 - data[i]) < abs(c2 - data[i])) {
                class1.push_back(data[i]);
            } else {
                class2.push_back(data[i]);
            }
        }
        c2 = mean(class2);
        c1 = mean(class1);
        if ((abs(lastc2 - c2) < e) and (abs(lastc1 - c1) < e)) {
            if (class1.size() > class2.size()) {
                return c1;
            } else {
                return c2;
            }
        }
        lastc2 = c2;
        lastc1 = c1;
    }
}
std::vector<double> peak_detection(std::vector<double>data,double thresh)
{
    double e = auto_threshold(data,thresh);
    std::vector<double> p,t;
    int a = 0,b = 0,d = 0;
    int i = -1;
    int xl = ((int)data.size() - 1);
    while (i != xl) {
        ++i;
        if (d == 0) {
            if (data[a] >= (data[i] + e)) {
                d = 2;
            } else if (data[i] >= (data[b] + e)) {
                d = 1;
            }
            if (data[a] <= data[i]) {
                a = i;
            } else if (data[i] <= data[b]) {
                b = i;
            }
        } else if (d == 1) {
            if (data[a] <= data[i]) {
                a = i;
            } else if (data[a] >= (data[i] + e)) {
                p.push_back(a);
                b = i;
                d = 2;
            }
        } else if (d == 2) {
            if (data[i] <= data[b]) {
                b = i;
            } else if (data[i] >= (data[b] + e)) {
                t.push_back(b);
                a = i;
                d = 1;
            }
        }
    }
    //    cout << p << endl;
    return p;
}

void write_audacity_labels(std::vector<double> peaks,std::vector<Real> onset_times,int offset)
{
    ofstream audacity_labels;
    
    audacity_labels.open("audacity_labels.txt");
    
    for (int i = 0; i < peaks.size() ; ++i) {
        audacity_labels << onset_times[peaks[i]*offset] << "\tEvent\n";
    }
    audacity_labels.close();
}


rowvec slice(mat& M,int row_index,int start,int stop)
{
	rowvec xr(stop - start);
	int counter = 0;
	for (int i = start; i < stop-1; ++i) {
		xr(counter) = M.row(row_index)(i);
		counter++;
	}
	
	return xr;
	
}

double gaussian(double x, double mu, double sigma) {
	return exp( -(((x-mu)/(sigma))*((x-mu)/(sigma)))/2.0 );
}

std::vector<float>  gausswin(int n,double sigma)
{
	std::vector<float> window;
	window.resize(n);
	int half_n = n / 2;
	double sum = 0;
	for (int i= -(half_n);i <= half_n;++i) {
		window[i+half_n] = gaussian(i,0,sigma);
		sum += window[i+half_n];
	}
	
	for (int i=0;i<n;++i) {
		window[i] /= sum;
	}
	
	return window;
}

int reflect(int M, int x)
{
	if(x < 0)
	{
		return -x - 1;
	}
	if(x >= M)
	{
		return 2*M - x - 1;
	}
	
	return x;
}

void gaussian_blur(mat& M,float size,float sigma)
{
	int n = M.n_rows;
	int win_size = n * size;
	int halfw = win_size / 2;
//	double sigma = 20;
	std::vector<float> coeffs = gausswin(win_size,sigma);
	
	Mat<float> temp(n,n);
	float sum,x1,y1;
	
	//    // along y - direction
	for(int y = 0; y < n; y++){
		for(int x = 0; x < n; x++){
			sum = 0.0;
			for(int i = halfw*-1; i <= halfw; i++){
				y1 = reflect(n, y - i);
				sum = sum + coeffs[i + halfw]*M(y1, x);
			}
			temp(y,x) = sum;
		}
	}
	
	// along x - direction
	for(int y = 0; y < n; y++){
		for(int x = 0; x < n; x++){
			sum = 0.0;
			for(int i =halfw*-1; i <= halfw; i++){
				x1 = reflect(n, x - i);
				sum = sum + coeffs[i + halfw]*M(y, x1);
			}
			M(y,x) = sum;
		}
	}
}
