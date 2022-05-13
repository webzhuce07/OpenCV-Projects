#include "KcgMatch.h"
#include <math.h>
 
#define KCG_EPS 0.00001f
#define KCG_PI	3.1415926535897932384626433832795f
#define KCG_MODEL_SUFFUX string(".yaml")
 
const float AngleRegionTable[16][2] = {
 
	0.f		, 22.5f	,
	22.5f	, 45.f	,
	45.f	, 67.5f	,
	67.5f	, 90.f	,
	90.f	, 112.5f,
	112.5f	, 135.f	,
	135.f	, 157.5f,
	157.5f	, 180.f,
	180.f	, 202.5f,
	202.5f	, 225.f,
	225.f	, 247.5f,
	247.5f	, 270.f,
	270.f	, 292.5f,
	292.5f	, 315.f,
	315.f	, 337.5f,
	337.5f	, 360.f
};
 
namespace cv_dnn_nms {
 
template <typename T>
static inline bool SortScorePairDescend(const std::pair<float, T>& pair1, const std::pair<float, T>& pair2) {
	
	return pair1.first > pair2.first;
}
 
inline void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
	std::vector<std::pair<float, int> >& score_index_vec) {
 
	for (size_t i = 0; i < scores.size(); ++i)
	{
		if (scores[i] > threshold)
		{
			//score_index_vec.push_back(std::make_pair(scores[i], i));
			std::pair<float, int> psi;
			psi.first = scores[i];
			psi.second = (int)i;
			score_index_vec.push_back(psi);
		}
	}
	std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
		SortScorePairDescend<int>);
	if (top_k > 0 && top_k < (int)score_index_vec.size())
	{
		score_index_vec.resize(top_k);
	}
}
 
template <typename BoxType>
inline void NMSFast_(const std::vector<BoxType>& bboxes,
	const std::vector<float>& scores, const float score_threshold,
	const float nms_threshold, const float eta, const int top_k,
	std::vector<int>& indices, float(*computeOverlap)(const BoxType&, const BoxType&)) {
 
	CV_Assert(bboxes.size() == scores.size());
	std::vector<std::pair<float, int> > score_index_vec;
	GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);
 
	float adaptive_threshold = nms_threshold;
	indices.clear();
	for (size_t i = 0; i < score_index_vec.size(); ++i) {
		const int idx = score_index_vec[i].second;
		bool keep = true;
		for (int k = 0; k < (int)indices.size() && keep; ++k) {
			const int kept_idx = indices[k];
			float overlap = computeOverlap(bboxes[idx], bboxes[kept_idx]);
			keep = overlap <= adaptive_threshold;
		}
		if (keep)
			indices.push_back(idx);
		if (keep && eta < 1 && adaptive_threshold > 0.5) {
			adaptive_threshold *= eta;
		}
	}
}
 
template<typename _Tp> static inline
	double jaccardDistance__(const Rect_<_Tp>& a, const Rect_<_Tp>& b) {
	_Tp Aa = a.area();
	_Tp Ab = b.area();
 
	if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon()) {
		// jaccard_index = 1 -> distance = 0
		return 0.0;
	}
 
	double Aab = (a & b).area();
	// distance = 1 - jaccard_index
	return 1.0 - Aab / (Aa + Ab - Aab);
}
 
template <typename T>
static inline float rectOverlap(const T& a, const T& b) {
 
	return 1.f - static_cast<float>(jaccardDistance__(a, b));
}
 
void NMSBoxes(const std::vector<Rect>& bboxes, const std::vector<float>& scores,
	const float score_threshold, const float nms_threshold,
	std::vector<int>& indices, const float eta = 1, const int top_k = 0) {
 
	NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, rectOverlap);
}
 
} // end namespace cv_dnn_nms
 
namespace kcg_matching{
 
KcgMatch::KcgMatch(string model_root, string class_name) {
 
	assert(!model_root.empty() && "model_root should not empty.");
	assert(!class_name.empty() && "class_name should not empty.");
	if (model_root[model_root.length() - 1] != '/') {
 
		model_root.push_back('/');
	}
	model_root_ = model_root;
	class_name_ = class_name;
 
	/// Create 180*180 table
	for (int i = 0; i < 180; i++) {
 
		for (int j = 0; j < 180; j++) {
 
			float rad = (i - j) * KCG_PI / 180.f;
			score_table_[i][j] = fabs(cosf(rad));
		}
	}
 
	/// Create 8*8 table
	ATTR_ALIGN(8) unsigned char score_table_8d[8][8];
	for (int i = 0; i < 8; i++) {
 
		for (int j = 0; j < 8; j++) {
 
			float rad = (i - j) * (180.f / 8.f) * KCG_PI / 180.f;
			score_table_8d[i][j] = (unsigned char)(fabs(cosf(rad))*100.f);
		}
	}
 
	/// Create 8*256 table
	for (int i = 0; i < 8; i++) {
 
		for (int j = 0; j < 256; j++) {
 
			unsigned char max_score = 0;
			for (int shift_time = 0; shift_time < 8; shift_time++) {
 
				unsigned char flg = (j >> shift_time) & 0b00000001;
				if (flg) {
 
					if (score_table_8d[i][shift_time] > max_score) {
 
						max_score = score_table_8d[i][shift_time];
					}
				}
			}
			score_table_8map_[i][j] = max_score;
		}
	}
}
 
KcgMatch::~KcgMatch() {
 
}
 
void KcgMatch::MakingTemplates(Mat model, AngleRange angle_range, ScaleRange scale_range,
	int num_features,float weak_thresh, float strong_thresh, Mat mask) {
 
	ClearModel();
	PaddingModelAndMask(model, mask, scale_range.end);
	angle_range_ = angle_range;
	scale_range_ = scale_range;
	vector<ShapeInfo> shape_infos = ProduceShapeInfos(angle_range, scale_range);
	vector<Mat> l0_mdls; l0_mdls.clear();
	vector<Mat> l0_msks; l0_msks.clear();
	for (int s = 0; s < shape_infos.size(); s++) {
 
		l0_mdls.push_back(MdlOf(model, shape_infos[s]));
		l0_msks.push_back(MskOf(mask, shape_infos[s]));
	}
	for (int p = 0; p <= PyramidLevel_7; p++) {
 
		for (int s = 0; s < shape_infos.size(); s++) {
 
			Mat mdl_pyrd = l0_mdls[s];
			Mat msk_pyrd = l0_msks[s];
			if (p > 0) {
 
				Size sz = Size(l0_mdls[s].cols >> 1, l0_mdls[s].rows >> 1);
				pyrDown(l0_mdls[s], mdl_pyrd, sz);
				pyrDown(l0_msks[s], msk_pyrd, sz);
			}
			erode(msk_pyrd, msk_pyrd, Mat(), Point(-1, -1), 1, BORDER_REPLICATE);
			l0_mdls[s] = mdl_pyrd;
			l0_msks[s] = msk_pyrd;
 
			int features_pyrd = (int)((num_features >> p) * shape_infos[s].scale);
 
			Mat mag8, angle8, quantized_angle8;
			QuantifyEdge(mdl_pyrd, angle8, quantized_angle8, mag8, weak_thresh, false);
			Template templ = ExtractTemplate(	angle8, quantized_angle8, mag8, 
												shape_infos[s], PyramidLevel(p),
												weak_thresh, strong_thresh,
												features_pyrd, msk_pyrd);
			templ_all_[p].push_back(templ);
 
			Mat mag180, angle180, quantized_angle180;
			QuantifyEdge(mdl_pyrd, angle180, quantized_angle180, mag180, weak_thresh, true);
			templ = ExtractTemplate(	angle180, quantized_angle180, mag180,
										shape_infos[s], PyramidLevel(p),
										weak_thresh, strong_thresh,
										features_pyrd, msk_pyrd);
			templ_all_[p + 8].push_back(templ);
 
			/// draw
			/*Mat draw_mask;
			msk_pyrd.copyTo(draw_mask);
			DrawTemplate(draw_mask, templ, Scalar(0));
			imshow("draw_mask", draw_mask);
			waitKey(1);*/
		}
		cout << "train pyramid level " << p << " complete." << endl;
	}
	SaveModel();
}
 
vector<Match> KcgMatch::Matching(Mat source, float score_thresh, float overlap,
	float mag_thresh, float greediness, PyramidLevel pyrd_level, int T, int top_k,
	MatchingStrategy strategy, const Mat mask) {
 
	InitMatchParameter(score_thresh, overlap, mag_thresh, greediness, T, top_k, strategy);
	GetAllPyramidLevelValidSource(source, pyrd_level);
 
	vector<Match> matches; 
	matches = MatchingPyrd8(sources_[pyrd_level], pyrd_level, region8_idxes_);
	matches = GetTopKMatches(matches);
 
	matches = ReconfirmMatches(matches, pyrd_level);
	matches = GetTopKMatches(matches);
 
	matches = MatchingFinal(matches, pyrd_level);
	matches = GetTopKMatches(matches);
 
	return matches;
}
 
void KcgMatch::DrawMatches(Mat &image, vector<Match> matches, Scalar color) {
 
	//#pragma omp parallel for
	for (int i = 0; i < matches.size(); i++) {
 
		auto match = matches[i];
		auto templ = templ_all_[8][match.template_id];
		int w = match.x + templ.w;
		int h = match.y + templ.h;
		for (int i = 0; i < (int)templ.features.size(); i++) {
 
			auto feature = templ.features[i];
			//circle(image, cv::Point(match.x + feature.x, match.y + feature.y), 1, color, 1);
			line(image,
				Point(match.x + feature.x, match.y + feature.y),
				Point(match.x + feature.x, match.y + feature.y),
				color, 1);
		}
		cv::rectangle(image, { match.x, match.y }, { w, h }, color, 1);
		char info[128];
		sprintf(info,
				"%.2f%% [%.2f, %.2f]",
				match.similarity * 100,
				templ.shape_info.angle,
				templ.shape_info.scale);
		cv::putText(image,
					info,
					Point(match.x, match.y), FONT_HERSHEY_PLAIN, 1.f, color, 1);
	}
}
 
void KcgMatch::PaddingModelAndMask(Mat &model, Mat &mask, float max_scale) {
 
	CV_Assert(!model.empty() && "model is empty.");
	if (mask.empty()) 
		mask = Mat(model.size(), CV_8UC1, { 255 });
	else 
		CV_Assert(model.size() == mask.size());
	int min_side_length = std::min(model.rows, model.cols);
	int diagonal_line_length =
		(int)ceil(std::sqrt(model.rows*model.rows + model.cols*model.cols)*max_scale);
	int padding = ((diagonal_line_length - min_side_length) >> 1) + 16;
	int double_padding = (padding << 1);
	Mat model_padded = Mat(model.rows + double_padding, model.cols + double_padding, model.type(), Scalar::all(0));
	model.copyTo(model_padded(Rect(padding, padding, model.cols, model.rows)));
	Mat mask_padded = Mat(mask.rows + double_padding, mask.cols + double_padding, mask.type(), Scalar::all(0));
	mask.copyTo(mask_padded(Rect(padding, padding, mask.cols, mask.rows)));
	model = model_padded;
	mask = mask_padded;
}
 
vector<ShapeInfo> KcgMatch::ProduceShapeInfos(AngleRange angle_range, ScaleRange scale_range) {
 
	assert(scale_range.begin > KCG_EPS && scale_range.end > KCG_EPS);
	assert(angle_range.end >= angle_range.begin);
	assert(scale_range.end >= scale_range.begin);
	assert(angle_range.step > KCG_EPS);
	assert(scale_range.step > KCG_EPS);
	vector<ShapeInfo> shape_infos;
	shape_infos.clear();
	for (float scale = scale_range.begin; scale <= scale_range.end + KCG_EPS; scale += scale_range.step) {
 
		for (float angle = angle_range.begin; angle <= angle_range.end + KCG_EPS; angle += angle_range.step) {
 
			ShapeInfo info;
			info.angle = angle;
			info.scale = scale;
			shape_infos.push_back(info);
		}
	}
	return shape_infos;
}
 
Mat KcgMatch::Transform(Mat src, float angle, float scale) {
 
	Mat dst;
	Point center(src.cols / 2, src.rows / 2);
	Mat rot_mat = cv::getRotationMatrix2D(center, angle, scale);
	warpAffine(src, dst, rot_mat, src.size());
	return dst;
}
 
Mat KcgMatch::MdlOf(Mat model, ShapeInfo info) {
 
	return Transform(model, info.angle, info.scale);
}
 
Mat KcgMatch::MskOf(Mat mask, ShapeInfo info) {
 
	return (Transform(mask, info.angle, info.scale) > 0);
}
 
void KcgMatch::DrawTemplate(Mat &image, Template templ, Scalar color) {
 
	for (int i = 0; i < templ.features.size(); i++) {
 
		auto feature = templ.features[i];
		line(image,
			Point(templ.x + feature.x, templ.y + feature.y),
			Point(templ.x + feature.x, templ.y + feature.y),
			color, 1);
	}
}
 
void KcgMatch::QuantifyEdge(Mat image, Mat &angle, Mat &quantized_angle, Mat &mag, float mag_thresh, bool calc_180) {
 
	Mat dx, dy;
	//Sobel(image, dx, CV_32F, 1, 0, 3, 1.0, 0.0, BORDER_REPLICATE);
	//Sobel(image, dy, CV_32F, 0, 1, 3, 1.0, 0.0, BORDER_REPLICATE);
	float mask_x[3][3] = { { -1,0,1 },{ -2,0,2 },{ -1,0,1 } };
	float mask_y[3][3] = { { 1,2,1 },{ 0,0,0 },{ -1,-2,-1 } };
	Mat kernel_x = Mat(3, 3, CV_32F, mask_x);
	Mat kernel_y = Mat(3, 3, CV_32F, mask_y);
	filter2D(image, dx, CV_32F, kernel_x);
	filter2D(image, dy, CV_32F, kernel_y);
	//dx = abs(dx);
	//dy = abs(dy);
	mag = dx.mul(dx) + dy.mul(dy);
	phase(dx, dy, angle, true);
 
	if(calc_180)
		Quantify180(angle, quantized_angle, mag, mag_thresh);
	else
		Quantify8(angle, quantized_angle, mag, mag_thresh);
}
 
void KcgMatch::Quantify8(Mat angle, Mat &quantized_angle, Mat mag, float mag_thresh) {
 
	Mat_<unsigned char> quantized_unfiltered;
	angle.convertTo(quantized_unfiltered, CV_8U, 16.0f / 360.0f);
	for (int r =0 ; r < angle.rows; ++r)
	{
		unsigned char *quant_ptr = quantized_unfiltered.ptr<unsigned char>(r);
		for (int c = 0; c < angle.cols; ++c)
		{
			quant_ptr[c] &= 7;
		}
	}
	//quantized_unfiltered.copyTo(quantized_angle);
	quantized_angle = Mat::zeros(angle.size(), CV_8U);
	for (int r = 0; r < quantized_angle.rows; ++r) {
 
		quantized_angle.ptr<unsigned char>(r)[0] = 255;
		quantized_angle.ptr<unsigned char>(r)[quantized_angle.cols - 1] = 255;
	}
	for (int c = 0; c < quantized_angle.cols; ++c) {
 
		quantized_angle.ptr<unsigned char>(0)[c] = 255;
		quantized_angle.ptr<unsigned char>(quantized_angle.rows - 1)[c] = 255;
	}
		
	for (int r = 1; r < angle.rows - 1; ++r)
	{
		float *mag_ptr= mag.ptr<float>(r);
		for (int c = 1; c < angle.cols - 1; ++c)
		{
			if (mag_ptr[c] >= (mag_thresh * mag_thresh))
			{
				int histogram[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
 
				unsigned char *patch3x3_row = &quantized_unfiltered(r - 1, c - 1);
				histogram[patch3x3_row[0]]++;
				histogram[patch3x3_row[1]]++;
				histogram[patch3x3_row[2]]++;
 
				patch3x3_row += quantized_unfiltered.step1();
				histogram[patch3x3_row[0]]++;
				histogram[patch3x3_row[1]]++;
				histogram[patch3x3_row[2]]++;
 
				patch3x3_row += quantized_unfiltered.step1();
				histogram[patch3x3_row[0]]++;
				histogram[patch3x3_row[1]]++;
				histogram[patch3x3_row[2]]++;
 
				// Find bin with the most votes from the patch
				int max_votes = 0;
				int index = -1;
				for (int i = 0; i < 8; ++i)
				{
					if (max_votes < histogram[i])
					{
						index = i;
						max_votes = histogram[i];
					}
				}
 
				// Only accept the quantization if majority of pixels in the patch agree
				static const int NEIGHBOR_THRESHOLD = 5;
				if (max_votes >= NEIGHBOR_THRESHOLD)
					quantized_angle.at<unsigned char>(r, c) = index;
				else
					quantized_angle.at<unsigned char>(r, c) = 255;
			}
			else
			{
				quantized_angle.at<unsigned char>(r, c) = 255;
			}
		}
	}
}
 
void KcgMatch::Quantify180(Mat angle, Mat &quantized_angle, Mat mag, float mag_thresh) {
 
	quantized_angle = Mat::zeros(angle.size(), CV_8U);
	#pragma omp parallel for
	for (int r = 0; r < angle.rows; ++r)
	{
		unsigned char *quantized_angle_ptr = quantized_angle.ptr<unsigned char>(r);
		float *angle_ptr = angle.ptr<float>(r);
		float *mag_ptr = mag.ptr<float>(r);
		for (int c = 0; c < angle.cols; ++c)
		{
			if (mag_ptr[c] >= (mag_thresh * mag_thresh))
				quantized_angle_ptr[c] = (int)round(angle_ptr[c]) % 180;
			else
				quantized_angle_ptr[c] = 255;
		}
	}
}
 
Template KcgMatch::ExtractTemplate(Mat angle, Mat quantized_angle, Mat mag, ShapeInfo shape_info, 
	PyramidLevel pl, float weak_thresh, float strong_thresh, int num_features, Mat mask) {
 
	Mat local_angle = Mat(angle.size(), angle.type());
	for (int r = 0; r < angle.rows; ++r) {
 
		float *angle_ptr = angle.ptr<float>(r);
		float *local_angle_ptr = local_angle.ptr<float>(r);
		for (int c = 0; c < angle.cols; ++c) {
 
			float dir = angle_ptr[c];
			if ((dir > 0. && dir < 22.5) || (dir > 157.5 && dir < 202.5) || (dir > 337.5 && dir < 360.))
				local_angle_ptr[c] = 0.f;
			else if ((dir > 22.5 && dir < 67.5) || (dir > 202.5 && dir < 247.5))
				local_angle_ptr[c] = 45.f;
			else if ((dir > 67.5 && dir < 112.5) || (dir > 247.5 && dir < 292.5))
				local_angle_ptr[c] = 90.f;
			else if ((dir > 112.5 && dir < 157.5) || (dir > 292.5 && dir < 337.5))
				local_angle_ptr[c] = 135.f;
			else
				local_angle_ptr[c] = 0.f;
		}
	}
 
	vector<Candidate> candidates;
	candidates.clear();
	bool no_mask = mask.empty();
	float weak_sq = weak_thresh * weak_thresh;
	float strong_sq = strong_thresh * strong_thresh;
	float pre_grad, lst_grad;
	for (int r = 1; r < mag.rows - 1; ++r)
	{
		const unsigned char *mask_ptr = no_mask ? NULL : mask.ptr<unsigned char>(r);
		const float* pre_ptr = mag.ptr<float>(r - 1);
		const float* cur_ptr = mag.ptr<float>(r);
		const float* lst_ptr = mag.ptr<float>(r + 1);
		float *local_angle_ptr = local_angle.ptr<float>(r);
 
		for (int c = 1; c < mag.cols - 1; ++c)
		{
			if (no_mask || mask_ptr[c])
			{
				switch ((int)local_angle_ptr[c]) {
 
				case 0:
					pre_grad = cur_ptr[c - 1];
					lst_grad = cur_ptr[c + 1];
					break;
				case 45:
					pre_grad = pre_ptr[c + 1];
					lst_grad = lst_ptr[c - 1];
					break;
				case 90:
					pre_grad = pre_ptr[c];
					lst_grad = lst_ptr[c];
					break;
				case 135:
					pre_grad = pre_ptr[c - 1];
					lst_grad = lst_ptr[c + 1];
					break;
				}
				if ((cur_ptr[c] > pre_grad) && (cur_ptr[c] > lst_grad)) {
 
					float score = cur_ptr[c];
					bool validity = false;
					if (score >= weak_sq) {
 
						if (score >= strong_sq) {
 
							validity = true;
						}
						else {
 
							if (((pre_ptr[c - 1])	>= strong_sq) ||
								((pre_ptr[c])		>= strong_sq) ||
								((pre_ptr[c + 1])	>= strong_sq) ||
								((cur_ptr[c - 1])	>= strong_sq) ||
								((cur_ptr[c + 1])	>= strong_sq) ||
								((lst_ptr[c - 1])	>= strong_sq) ||
								((lst_ptr[c])		>= strong_sq) ||
								((lst_ptr[c + 1])	>= strong_sq))
							{
								validity = true;
							}
						}
					}
					if (validity == true && 
						quantized_angle.at<unsigned char>(r, c) != 255) {
 
						Candidate cd;
						cd.score = score;
						cd.feature.x = c;
						cd.feature.y = r;
						cd.feature.lbl = quantized_angle.at<unsigned char>(r, c);
						candidates.push_back(cd);
					}
				}
 
			}
		}
	}
 
	Template templ;
	templ.shape_info.angle = shape_info.angle;
	templ.shape_info.scale = shape_info.scale;
	templ.pyramid_level = pl;
	templ.is_valid = 0;
	templ.features.clear();
 
	if (candidates.size() >= num_features && num_features > 0) {
 
		std::stable_sort(candidates.begin(), candidates.end());
		float distance = static_cast<float>(candidates.size() / num_features + 1);
		templ = SelectScatteredFeatures(candidates, num_features, distance);
	}
	else {
 
		for (int c = 0; c < candidates.size(); c++) {
 
			templ.features.push_back(candidates[c].feature);
		}
	}
	
	if (templ.features.size() > 0) {
 
		templ.is_valid = 1;
		CropTemplate(templ);
	}
 
	return templ;
}
 
Template KcgMatch::SelectScatteredFeatures(vector<Candidate> candidates, int num_features, float distance) {
 
	Template templ;
	templ.features.clear();
	float distance_sq = distance * distance;
	int i = 0;
	while (templ.features.size() < num_features) {
 
		Candidate c = candidates[i];
		// Add if sufficient distance away from any previously chosen feature
		bool keep = true;
		for (int j = 0; (j < (int)templ.features.size()) && keep; ++j)
		{
			Feature f = templ.features[j];
			keep = ((c.feature.x - f.x) * (c.feature.x - f.x) + (c.feature.y - f.y) * (c.feature.y - f.y) >= distance_sq);
		}
		if (keep)
			templ.features.push_back(c.feature);
 
		if (++i == (int)candidates.size())
		{
			// Start back at beginning, and relax required distance
			i = 0;
			distance -= 1.0f;
			distance_sq = distance * distance;
			// if (distance < 3)
			// {
			//     // we don't want two features too close
			//     break;
			// }
		}
	}
	return templ;
}
 
Rect KcgMatch::CropTemplate(Template &templ) {
 
	int min_x = std::numeric_limits<int>::max();
	int min_y = std::numeric_limits<int>::max();
	int max_x = std::numeric_limits<int>::min();
	int max_y = std::numeric_limits<int>::min();
 
	// First pass: find min/max feature x,y 
	for (int i = 0; i < (int)templ.features.size(); ++i)
	{
		int x = templ.features[i].x;
		int y = templ.features[i].y;
		min_x = std::min(min_x, x);
		min_y = std::min(min_y, y);
		max_x = std::max(max_x, x);
		max_y = std::max(max_y, y);
	}
	
	/// @todo Why require even min_x, min_y?
	if (min_x % 2 == 1)
		--min_x;
	if (min_y % 2 == 1)
		--min_y;
 
	// Second pass: set width/height and shift all feature positions
	templ.w = (max_x - min_x);
	templ.h = (max_y - min_y);
	templ.x = min_x;
	templ.y = min_y;
 
	for (int i = 0; i < (int)templ.features.size(); ++i)
	{
		templ.features[i].x -= templ.x;
		templ.features[i].y -= templ.y;
	}
	return Rect(min_x, min_y, max_x - min_x, max_y - min_y);
}
 
void KcgMatch::LoadRegion8Idxes() {
 
	int keys[16] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
	region8_idxes_.clear();
	int angle_region = (int)((angle_range_.end - angle_range_.begin) / angle_range_.step) + 1;
	int scale_region = (int)((scale_range_.end - scale_range_.begin) / scale_range_.step) + 1;
	for (int ar = 0; ar < angle_region; ar++) {
 
		float cur_agl = templ_all_[PyramidLevel_0][ar].shape_info.angle;
		if (cur_agl < 0.f) cur_agl += 360.f;
		int idx = 0;
		for (int i = 0; i < 16; i++) {
 
			if (cur_agl >= AngleRegionTable[i][0] &&
				cur_agl < AngleRegionTable[i][1]) {
 
				idx = i;
				break;
			}
		}
		if (keys[idx] == 0) {
 
			for (int sr = 0; sr < scale_region; sr++) {
 
				region8_idxes_.push_back(ar + sr * angle_region);
			}
		}
		keys[idx] = 1;
	}
}
 
void KcgMatch::SaveModel() {
 
	int total_templ = 0;
	for (int i = 0; i < PyramidLevel_TabooUse; i++) {
 
		total_templ += (int)templ_all_[i].size();
	}
	assert((total_templ / PyramidLevel_TabooUse) == templ_all_[0].size());
	int match_range_size = (int)templ_all_[0].size();
	string model_name = model_root_ + class_name_ + KCG_MODEL_SUFFUX;
	FileStorage fs(model_name, FileStorage::WRITE);
	fs << "class_name" << class_name_;
	fs << "total_pyramid_levels" << PyramidLevel_7;
	fs << "angle_range_bgin" << angle_range_.begin;
	fs << "angle_range_end" << angle_range_.end;
	fs << "angle_range_step" << angle_range_.step;
	fs << "scale_range_bgin" << scale_range_.begin;
	fs << "scale_range_end" << scale_range_.end;
	fs << "scale_range_step" << scale_range_.step;
	fs << "templates"
	<< "[";
	{
		for (int i = 0; i < match_range_size; i++) {
 
			fs << "{";
			fs << "template_id" << int(i);
			fs << "template_pyrds"
			<< "[";
			{
				for (int j = 0; j < PyramidLevel_TabooUse; j++) {
 
					auto templ = templ_all_[j][i];
					fs << "{";
					fs << "id" << int(i);
					fs << "pyramid_level" << templ.pyramid_level;
					fs << "is_valid" << templ.is_valid;
					fs << "x" << templ.x;
					fs << "y" << templ.y;
					fs << "w" << templ.w;
					fs << "h" << templ.h;
					fs << "shape_scale" << templ.shape_info.scale;
					fs << "shape_angle" << templ.shape_info.angle;
					fs << "feature_size" << (int)templ.features.size();
					fs << "features"
					<< "[";
					{
						for (int k = 0; k < (int)templ.features.size(); k++) {
 
							auto feat = templ.features[k];
							fs << "[:" << feat.x << feat.y << feat.lbl << "]";
						}
					}
					fs << "]";
					fs << "}";
				}
			}
			fs << "]";
			fs << "}";
		}
	}
	fs << "]";
}
 
void KcgMatch::LoadModel() {
 
	ClearModel();
	string model_name = model_root_ + class_name_ + KCG_MODEL_SUFFUX;
	FileStorage fs(model_name, FileStorage::READ);
	assert(fs.isOpened() && "load model failed.");
	FileNode fn = fs.root();
	angle_range_.begin = fn["angle_range_bgin"];
	angle_range_.end = fn["angle_range_end"];
	angle_range_.step = fn["angle_range_step"];
	scale_range_.begin = fn["scale_range_bgin"];
	scale_range_.end = fn["scale_range_end"];
	scale_range_.step = fn["scale_range_step"];
	FileNode tps_fn = fn["templates"];
	FileNodeIterator tps_it = tps_fn.begin(), tps_it_end = tps_fn.end();
	for (; tps_it != tps_it_end; ++tps_it)
	{
		int template_id = (*tps_it)["template_id"];
		FileNode pyrds_fn = (*tps_it)["template_pyrds"];
		FileNodeIterator pyrd_it = pyrds_fn.begin(), pyrd_it_end = pyrds_fn.end();
		int pl = 0;
		for (; pyrd_it != pyrd_it_end; ++pyrd_it)
		{
			FileNode pyrd_fn = (*pyrd_it);
			Template templ;
			templ.id = pyrd_fn["id"];
			templ.pyramid_level = pyrd_fn["pyramid_level"];
			templ.is_valid = pyrd_fn["is_valid"];
			templ.x = pyrd_fn["x"];
			templ.y = pyrd_fn["y"];
			templ.w = pyrd_fn["w"];
			templ.h = pyrd_fn["h"];
			templ.shape_info.scale = pyrd_fn["shape_scale"];
			templ.shape_info.angle = pyrd_fn["shape_angle"];
			FileNode features_fn = pyrd_fn["features"];
			FileNodeIterator feature_it = features_fn.begin(), feature_it_end = features_fn.end();
			for (; feature_it != feature_it_end; ++feature_it)
			{
				FileNode feature_fn = (*feature_it);
				FileNodeIterator feature_info = feature_fn.begin();
				Feature feat;
				feature_info >> feat.x >> feat.y >> feat.lbl;
				templ.features.push_back(feat);
			}
			templ_all_[pl].push_back(templ);
			pl++;
		}
	}
 
	LoadRegion8Idxes();
}
 
void KcgMatch::ClearModel() {
 
	for (int i = 0; i < PyramidLevel_TabooUse; i++) {
 
		templ_all_[i].clear();
	}
}
 
void KcgMatch::InitMatchParameter(float score_thresh, float overlap, float mag_thresh, float greediness, int T, int top_k, MatchingStrategy strategy) {
 
	score_thresh_ = score_thresh;
	overlap_ = overlap;
	mag_thresh_ = mag_thresh;
	greediness_ = greediness;
	T_ = T;
	top_k_ = top_k;
	strategy_ = strategy;
}
 
void KcgMatch::GetAllPyramidLevelValidSource(cv::Mat &source, PyramidLevel pyrd_level) {
 
	sources_.clear();
	for (int pl = 0; pl <= pyrd_level; pl++) {
 
		Mat source_pyrd;
		if (pl == 0) source_pyrd = source;
		else pyrDown(source, source_pyrd, Size(source.cols >> 1, source.rows >> 1));
		source = source_pyrd;
		sources_.push_back(source_pyrd);
	}
}
 
vector<Match> KcgMatch::GetTopKMatches(vector<Match> matches) {
 
	vector<Match> top_k_matches;
	top_k_matches.clear();
	if (top_k_ > 0 && (top_k_ < matches.size()) && (matches.size() > 0)) {
 
		int k = 0;
		top_k_matches.push_back(matches[0]);
		for (int m = 1; m < matches.size(); m++) {
 
			if (matches[m].similarity < matches[m - 1].similarity) {
 
				++k;
				if(k >= top_k_) break;
			}
			top_k_matches.push_back(matches[m]);
		}
	}
	else
	{
		top_k_matches = matches;
	}
	return top_k_matches;
}
 
vector<Match> KcgMatch::DoNmsMatches(vector<Match> matches, PyramidLevel pl, float overlap) {
 
	vector<Rect> boxes; boxes.clear();
	vector<float> scores; scores.clear();
	vector<int> indices; indices.clear();
	for (int m = 0; m < matches.size(); m++) {
 
		auto templ = templ_all_[pl][matches[m].template_id];
		Rect box = Rect(matches[m].x, matches[m].y, templ.w, templ.h);
		boxes.insert(boxes.end(), box);
		scores.insert(scores.end(), matches[m].similarity);
	}
	cv_dnn_nms::NMSBoxes(boxes, scores, overlap, overlap, indices);
	vector<Match> final_matches; final_matches.clear();
	for (auto index : indices) {
 
		final_matches.push_back(matches[index]);
	}
	return final_matches;
}
 
vector<Match> KcgMatch::MatchingPyrd180(Mat src, PyramidLevel pl, vector<int> region_idxes) {
 
	pl = PyramidLevel(pl + 8);
	vector<Match> matches; matches.clear();
	Mat angle, quantized_angle, mag;
	QuantifyEdge(src, angle, quantized_angle, mag, mag_thresh_, true);
	#pragma omp parallel 
	{
		int tlsz = region_idxes.empty() ? ((int)templ_all_[pl].size()) : ((int)region_idxes.size());
		#pragma omp for nowait
		for (int t = 0; t < tlsz; t++) {
 
			Template templ = region_idxes.empty() ? (templ_all_[pl][t]) : (templ_all_[pl][region_idxes[t]]);
			for (int r = 0; r < quantized_angle.rows - templ.h; r++) {
 
				for (int c = 0; c < quantized_angle.cols - templ.w; c++) {
 
					int fsz = (int)templ.features.size();
					float partial_sum = 0.f;
					bool valid = true;
					for (int f = 0; f < fsz; f++) {
 
						Feature feat = templ.features[f];
						int sidx = quantized_angle.ptr<unsigned char>(r + feat.y)[c + feat.x];
						int tidx = feat.lbl;
						if (sidx != 255) {
 
							partial_sum += score_table_[sidx][tidx];
						}
						if (partial_sum + (fsz - f) * greediness_ < score_thresh_ * fsz) {
 
							valid = false;
							break;
						}
					}
					if (valid) {
 
						float score = partial_sum / fsz;
						if (score >= score_thresh_) {
 
							Match match;
							match.x = c;
							match.y = r;
							match.similarity = score;
							match.template_id = templ.id;
							#pragma omp critical
							matches.insert(matches.end(), match);
						}
					}
 
				}
			}
		}
	}
	matches = DoNmsMatches(matches, pl, overlap_);
	return matches;
}
 
vector<Match> KcgMatch::MatchingPyrd8(Mat src, PyramidLevel pl, vector<int> region_idxes) {
 
	vector<Match> matches; matches.clear();
	Mat angle, quantized_angle, mag;
	QuantifyEdge(src, angle, quantized_angle, mag, mag_thresh_, false);
	Mat spread_angle;
	Spread(quantized_angle, spread_angle, T_);
	vector<Mat> response_maps;
	ComputeResponseMaps(spread_angle, response_maps);
	#pragma omp parallel 
	{
		int tlsz = region_idxes.empty() ? ((int)templ_all_[pl].size()) : ((int)region_idxes.size());
		#pragma omp for nowait
		for (int t = 0; t < tlsz; t++) {
 
			Template templ = region_idxes.empty() ? (templ_all_[pl][t]) : (templ_all_[pl][region_idxes[t]]);
			for (int r = 0; r < quantized_angle.rows - templ.h; r += T_) {
 
				for (int c = 0; c < quantized_angle.cols - templ.w; c += T_) {
 
					int fsz = (int)templ.features.size();
					int partial_sum = 0;
					bool valid = true;
					for (int f = 0; f < fsz; f++) {
 
						Feature feat = templ.features[f];
						int label = feat.lbl;
						partial_sum += 
							response_maps[label].ptr<unsigned char>(r + feat.y)[c + feat.x];
						if (partial_sum + (fsz - f) * greediness_ < score_thresh_ * fsz) {
 
							valid = false;
							break;
						}
					}
					if (valid) {
 
						float score = partial_sum / (100.f * fsz);
						if (score >= score_thresh_) {
 
							Match match;
							match.x = c;
							match.y = r;
							match.similarity = score;
							match.template_id = templ.id;
							#pragma omp critical
							matches.insert(matches.end(), match);
						}
					}
				}
			}
		}
	}
	matches = DoNmsMatches(matches, pl, overlap_);
	return matches;
}
 
void KcgMatch::Spread(const Mat quantized_angle, Mat &spread_angle, int T) {
 
	spread_angle = Mat::zeros(quantized_angle.size(), CV_8U);
	int cols = quantized_angle.cols;
	int rows = quantized_angle.rows;
	int half_T = 0;
	if (T != 1) half_T = T / 2;
	#pragma omp parallel for
	for (int r = half_T; r < rows - half_T; r++) {
 
		for (int c = half_T; c < cols - half_T; c++) {
 
			for (int i = -half_T; i <= half_T; i++) {
 
				for (int j = -half_T; j <= half_T; j++) {
 
					unsigned char shift_bits =
						quantized_angle.ptr<unsigned char>(r + i)[c + j];
					if (shift_bits < 8) {
 
						spread_angle.ptr<unsigned char>(r)[c] |=
							(unsigned char)(1 << shift_bits);
					}
				}
			}
		}
	}
}
 
void KcgMatch::ComputeResponseMaps(const Mat spread_angle, vector<Mat> &response_maps) {
 
	response_maps.clear();
	for (int i = 0; i < 8; i++) {
 
		Mat rm;
		rm.create(spread_angle.size(), CV_8U);
		response_maps.push_back(rm);
	}
	int cols = spread_angle.cols;
	int rows = spread_angle.rows;
	#pragma omp parallel for
	for (int i = 0; i < 8; i++) {
 
		for (int r = 0; r < rows; r++) {
 
			for (int c = 0; c < cols; c++) {
 
				response_maps[i].ptr<unsigned char>(r)[c] =
					score_table_8map_[i][spread_angle.ptr<unsigned char>(r)[c]];
			}
		}
	}
}
 
bool KcgMatch::CalcPyUpRoiAndStartPoint(PyramidLevel cur_pl, PyramidLevel obj_pl, Match match,
	Mat &r, Point &p, bool is_padding) {
 
	auto templ = templ_all_[cur_pl][match.template_id];
	int padding = 0;
	if (is_padding) {
 
		int min_side = std::min(templ.w, templ.h);
		int diagonal_line_length = (int)ceil(sqrt(templ.w*templ.w + templ.h*templ.h));
		padding = diagonal_line_length - min_side;
	}
	int err_pl = cur_pl - obj_pl;
	int T = 2 * T_;
	int extend_pixel = 1;
	cv::Point bp, ep;
	int multiple = (1 << err_pl);
	match.x -= (T + padding) / 2;
	match.y -= (T + padding) / 2;
	templ.w += (T + padding);
	templ.h += (T + padding);
	bp.x = (match.x - extend_pixel) * multiple;
	bp.y = (match.y - extend_pixel) * multiple;
	ep.x = (match.x + templ.w + extend_pixel) * multiple;
	ep.y = (match.y + templ.h + extend_pixel) * multiple;
	if (bp.x < 0) bp.x = 0;
	if (bp.y < 0) bp.y = 0;
	if (ep.x < 0) ep.x = 0;
	if (ep.y < 0) ep.y = 0;
	if (bp.x >= sources_[obj_pl].cols) bp.x = sources_[obj_pl].cols - 1;
	if (bp.y >= sources_[obj_pl].rows) bp.y = sources_[obj_pl].rows - 1;
	if (ep.x >= sources_[obj_pl].cols) ep.x = sources_[obj_pl].cols - 1;
	if (ep.y >= sources_[obj_pl].rows) ep.y = sources_[obj_pl].rows - 1;
	if (bp.x != ep.x || bp.y != ep.y) {
 
		Rect rect = Rect(bp, ep);
		Mat roi(sources_[obj_pl], rect);
		r = roi;
		p = bp;
		return true;
	}
	else
	{
		return false;
	}
}
 
void KcgMatch::CalcRegionIndexes(vector<int> &region_idxes, Match match, MatchingStrategy strategy) {
 
	region_idxes.clear();
	Template templ = templ_all_[PyramidLevel_0][match.template_id];
	float match_agl = templ.shape_info.angle;
	float match_sal = templ.shape_info.scale;
	int angle_region = (int)((angle_range_.end - angle_range_.begin) / angle_range_.step) + 1;
	int scale_region = (int)((scale_range_.end - scale_range_.begin) / scale_range_.step) + 1;
	if (strategy <= Strategy_Middling) {
 
		if (match_agl < 0.f) match_agl += 360.f;
		int key = (int)floor(match_agl / 22.5f);
		float left_agl = match_agl - key * 22.5f;
		for (int ar = 0; ar < angle_region; ar++) {
 
			float cur_agl = templ_all_[PyramidLevel_0][ar].shape_info.angle;
			if (cur_agl < 0.f) cur_agl += 360.f;
			int k = key;
			if (cur_agl >= AngleRegionTable[k][0] && cur_agl < AngleRegionTable[k][1]) {
 
				for (int sr = 0; sr < scale_region; sr++) {
 
					region_idxes.push_back(ar + sr * angle_region);
				}
			}
			if (strategy == Strategy_Accurate) {
 
				if (left_agl < 11.25f) {
 
					k = key - 1;
					if (k < 0) k = 15;
					if (cur_agl >= AngleRegionTable[k][0] && cur_agl < AngleRegionTable[k][1]) {
 
						for (int sr = 0; sr < scale_region; sr++) {
 
							region_idxes.push_back(ar + sr * angle_region);
						}
					}
				}
				else
				{
					k = key + 1;
					if (k > 15) k = 0;
					if (cur_agl >= AngleRegionTable[k][0] && cur_agl < AngleRegionTable[k][1]) {
 
						for (int sr = 0; sr < scale_region; sr++) {
 
							region_idxes.push_back(ar + sr * angle_region);
						}
					}
				}
			}
		}
	}
	else if(strategy == Strategy_Rough) {
 
		float err_range = 3.f;
		for (int ar = 0; ar < angle_region; ar++) {
 
			float cur_agl = templ_all_[PyramidLevel_0][ar].shape_info.angle;
			if (cur_agl >= (match_agl - angle_range_.step * err_range) && 
				cur_agl <= (match_agl + angle_range_.step * err_range)) {
 
				for (int sr = 0; sr < scale_region; sr++) {
 
					float cur_sal = templ_all_[PyramidLevel_0][ar + sr * angle_region].shape_info.scale;
					if (cur_sal >= (match_sal - scale_range_.step * err_range) &&
						cur_sal <= (match_sal + scale_range_.step * err_range)) {
 
						region_idxes.push_back(ar + sr * angle_region);
					}
				}
			}
		}
	}
}
 
vector<Match> KcgMatch::ReconfirmMatches(vector<Match> matches, PyramidLevel pl) {
 
	vector<Match> rf_matches;
	rf_matches.clear();
	for (int i = 0; i < matches.size(); i++) {
 
		Mat roi;
		Point sp;
		CalcPyUpRoiAndStartPoint(pl, pl, matches[i], roi, sp, true);
		vector<int> region_idxes;
		CalcRegionIndexes(region_idxes, matches[i], Strategy_Accurate);
		auto tmp_matches = MatchingPyrd8(roi, pl, region_idxes);
		if (tmp_matches.size() > 0) {
 
			tmp_matches[0].x += sp.x;
			tmp_matches[0].y += sp.y;
			rf_matches.push_back(tmp_matches[0]);
		}
	}
	rf_matches = DoNmsMatches(rf_matches, pl, overlap_);
	return rf_matches;
}
 
vector<Match> KcgMatch::MatchingFinal(vector<Match> matches, PyramidLevel pl) {
 
	vector<Match> final_matches;
	final_matches.clear();
	for (int i = 0; i < matches.size(); i++) {
 
		Mat roi;
		Point sp;
		CalcPyUpRoiAndStartPoint(pl, PyramidLevel_0, matches[i], roi, sp, false);
		vector<int> region_idxes;
		CalcRegionIndexes(region_idxes, matches[i], strategy_);
		auto tmp_matches = MatchingPyrd180(roi, PyramidLevel_0, region_idxes);
		if (tmp_matches.size() > 0) {
 
			tmp_matches[0].x += sp.x;
			tmp_matches[0].y += sp.y;
			final_matches.push_back(tmp_matches[0]);
		}
	}
	final_matches = DoNmsMatches(final_matches, pl, overlap_);
	return final_matches;
}
 
} // end namespace kcg_matching
 
 