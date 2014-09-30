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

#include "armadillo"

using namespace std;
using namespace essentia;
using namespace essentia::standard;

int main(int argc, const char * argv[]) {
	
	string audioFilename = argv[1];
	
	essentia::init();
	
	AlgorithmFactory& factory = standard::AlgorithmFactory::instance();
	Algorithm* audio = factory.create("EasyLoader","filename",audioFilename);
	
	Algorithm* onsetDetection = factory.create("OnsetRate");
	
	std::vector<Real> audioBuffer,onsets,startTimes,endTimes,peakM,peakF,frame,chroma;
	std::vector<std::vector<Real>> slicedFrames,chromaVector;
	Real rate;
	audio->output("audio").set(audioBuffer);
	
	onsetDetection->input("signal").set(audioBuffer);
	onsetDetection->output("onsets").set(onsets);
	onsetDetection->output("onsetRate").set(rate);

	audio->compute();
	onsetDetection->compute();
	
	std::cout << onsets << std::endl;
	
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
	
	for (int i = 0; i < slicedFrames.size();++i) {
		spectrum->input("frame").set(slicedFrames[i]);
		spectrum->compute();
		peaks->compute();
		hpcp->compute();
		
		chromaVector.push_back(chroma);
	}
	
	std::cout << chromaVector[0] << std::endl;
	
	return 0;
}
