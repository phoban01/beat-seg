//
//  main.cpp
//  beat-seg
//
//  Created by Piaras Hoban on 30/09/2014.
//  Copyright (c) 2014 Piaras Hoban. All rights reserved.
//

#include <iostream>

#include "essentia/essentia.h"
#include "essentia/essentiamath.h"
#include "essentia/algorithm.h"
#include "essentia/algorithmfactory.h"
#include "essentia/streaming/algorithms/poolstorage.h"
#include "essentia/scheduler/network.h"

#include "mlpack/core.hpp"
#include "mlpack/methods/neighbor_search/neighbor_search.hpp"
#include "armadillo"

using namespace std;
using namespace essentia;
using namespace essentia::standard;
using namespace arma;
using namespace mlpack::neighbor;


Mat<double> RP(std::vector<std::vector<float>>input);
double cos_distance(std::vector<float> v1, std::vector<float> v2);


int main(int argc, const char * argv[]) {
	
	string audioFilename = argv[1];
	
	essentia::init();
	
	AlgorithmFactory& factory = standard::AlgorithmFactory::instance();
	Algorithm* audio = factory.create("EasyLoader","filename",audioFilename);
	Algorithm* duration = factory.create("Duration");
	
	Algorithm* onsetDetection = factory.create("OnsetRate");
	
	std::vector<Real> audioBuffer,onsets,startTimes,endTimes,peakM,peakF,frame,chroma;
	std::vector<std::vector<Real>> slicedFrames,chromaVector;
	Real rate,dur;
	audio->output("audio").set(audioBuffer);
	
	duration->input("signal").set(audioBuffer);
	duration->output("duration").set(dur);
	
	onsetDetection->input("signal").set(audioBuffer);
	onsetDetection->output("onsets").set(onsets);
	onsetDetection->output("onsetRate").set(rate);

	audio->compute();
	duration->compute();
	onsetDetection->compute();
	
//	std::cout << onsets << std::endl;
	startTimes.push_back(0);endTimes.push_back(onsets[0]);
	
	for (int i = 0; i < onsets.size() - 1; ++i) {
		startTimes.push_back(onsets[i]);
		endTimes.push_back(onsets[i+1]);
	}
	
	Algorithm* slicer = factory.create("Slicer","endTimes",endTimes,"startTimes",startTimes);
	slicer->input("audio").set(audioBuffer);
	slicer->output("frame").set(slicedFrames);
	
	audio->compute();
	slicer->compute();
	
	std::cout << slicedFrames.size() << std::endl;

	Algorithm* spectrum = factory.create("Spectrum");
	Algorithm* peaks = factory.create("SpectralPeaks");
	spectrum->output("spectrum").set(frame);
	peaks->input("spectrum").set(frame);
	peaks->output("frequencies").set(peakF);
	peaks->output("magnitudes").set(peakM);

	Algorithm* hpcp = factory.create("HPCP");
	
	hpcp->input("frequencies").set(peakF);
	hpcp->input("magnitudes").set(peakM);
	hpcp->output("hpcp").set(chroma);
	
	ofstream plotc("/users/piarashoban/documents/codingProjects/beat-seg/chroma.txt");

	
	for (int i = 0; i < slicedFrames.size();++i) {
		if (slicedFrames[i].size() % 2 != 0) {slicedFrames[i].push_back(0);}
		spectrum->input("frame").set(slicedFrames[i]);
		spectrum->compute();
		peaks->compute();
		hpcp->compute();
		
		chromaVector.push_back(chroma);
		
		for (int j = 0;j < 12;++j) {
			plotc << chroma[j] << " ";
		}
		plotc << endl;
	}
	
	plotc.close();
	
//	mat sim = RP(chromaVector);
	
	
	return 0;
}

Mat<double> RP(std::vector<std::vector<float>>input)
{
	
	int n = (int)input.size();

	Mat<double> R(n,n);
	
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
				R(i,j) = cos_distance(input[i], input[j]);
		}
		cout << "-------> " << (i/(float)n)*100 << "%...\r";
		cout.flush();
	}
	
	return R;
}

double cos_distance(std::vector<float> v1, std::vector<float> v2)
{
	int N = (int)v1.size();
	float dot = 0.0;
	float mag1 = 0.0;
	float mag2 = 0.0;
	int n;
	for (n = 0; n < N; ++n)
	{
		dot += v1[n] * v2[n];
		mag1 += pow(v1[n], 2);
		mag2 += pow(v2[n], 2);
	}
	return exp((dot / (sqrt(mag1) * sqrt(mag2)))-1);
}