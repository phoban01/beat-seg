//
//  beat-seg.h
//  beat-seg
//
//  Created by Piaras Hoban on 05/10/2014.
//  Copyright (c) 2014 Piaras Hoban. All rights reserved.
//

#ifndef beat_seg_beat_seg_h
#define beat_seg_beat_seg_h

template <typename T>
cv::Mat
arma2cv(const arma::Mat<T>& from, bool copy = true) {
	cv::Mat mat(from.n_cols, from.n_rows, cv::DataType<T>::type, const_cast<T*>(from.memptr()));
	mat = mat.t();
	return copy ? mat.clone() : mat;
}

/// @brief Convert OpenCV matrix to Armadillo matrix.
template <typename T>
arma::Mat<T>
cv2arma(const cv::Mat& from, bool copy = true) {
	cv::Mat temp(from.t());
	return arma::Mat<T>(temp.ptr<T>(), from.rows, from.cols, copy);
}

#endif
